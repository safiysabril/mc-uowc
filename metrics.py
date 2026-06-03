"""
metrics.py
==========
Compute all channel metrics from accumulated Parquet data and save results
for paper analysis — no simulation, no figures.

Outputs
-------
metrics_homogeneous.csv       — all scalar metrics, one row per (medium, beam, range)
metrics_inhomogeneous.csv     — same for inhomogeneous data
cir_homogeneous.npz           — CIR + frequency-response arrays (--arrays flag)
cir_inhomogeneous.npz         — same for inhomogeneous data

CSV columns
-----------
  medium_name, beam_name, link_range_m

  -- Power --
  power_dB            Monte Carlo received power  (dB)
  beer_lambert_dB     Beer-Lambert reference power  (dB)
  power_excess_dB     MC − Beer-Lambert  (scattering gain/loss, dB)

  -- Timing --
  delay_spread_ns     RMS delay spread  (ns)
  bandwidth_mhz       3 dB bandwidth  (MHz)

  -- Receiver --
  snr_db              Electrical SNR  (dB); depends on noise_power and responsivity
  ber_ook             OOK bit-error rate; sigma_R2=0 unless --sigma-r2 is given

  -- Propagation statistics (from per-photon data) --
  capture_rate        n_captured / n_launched
  mean_n_scatters     Mean real-collision count per captured photon
  std_n_scatters      Std of real-collision count
  mean_excess_path_m  Mean extra path beyond straight-line distance  (m)
  std_excess_path_m   Std of extra path  (m)
  mean_r_m            Mean radial offset at receiver plane  (m)

  -- Counts --
  n_launched          Total photons launched (power normalisation denominator)
  n_captured          Photons captured
  cir_quality         good / sparse / very sparse / TOO FEW

Notes on BER
------------
sigma_R2 (Rytov variance) drives turbulence-fading mode selection:
  sigma_R2 = 0          → AWGN only  (default)
  0 < sigma_R2 < 0.3    → log-normal fading
  0.3 ≤ sigma_R2 < 5    → Gamma-Gamma
  sigma_R2 ≥ 5          → negative-exponential (fully developed speckle)
Turbulence parameters are not stored in Parquet; supply --sigma-r2 manually
if you know the Rytov variance for your channel scenario.

Usage
-----
  python metrics.py                           # default ./outputs directory
  python metrics.py --out ./results
  python metrics.py --arrays                  # also save CIR/FR as .npz
  python metrics.py --noise-power 1e-8 --responsivity 0.6
  python metrics.py --sigma-r2 0.15          # weak turbulence BER
  python metrics.py --hom path/to/file.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from uowc.config import SIM
from uowc.simulation import RunKey
from uowc.metrics import (
    compute_all_metrics,
    compute_snr,
    compute_ber_ook,
)
from uowc.analysis import (
    reconstruct_sweep_results,
    c_ref_map_from_df,
)


# ─────────────────────────────────────────────────────────────────────────────
# Quality label
# ─────────────────────────────────────────────────────────────────────────────

_CIR_THRESHOLDS = [(2_000, "good"), (250, "sparse"), (50, "very sparse"), (0, "TOO FEW")]


def _quality(n: int) -> str:
    return next(label for thr, label in _CIR_THRESHOLDS if n >= thr)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_from_df(
    df:           pd.DataFrame,
    *,
    noise_power:  float = 1e-7,
    responsivity: float = 0.5,
    sigma_r2:     float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute all channel metrics from a photon DataFrame.

    Parameters
    ----------
    df           : per-photon DataFrame loaded from Parquet
    noise_power  : receiver noise variance σ²_noise (A²) for SNR/BER
    responsivity : photodetector responsivity R (A/W) for SNR/BER
    sigma_r2     : Rytov variance σ²_R for turbulence BER (0 = AWGN only)

    Returns
    -------
    summary      : tidy DataFrame, one row per (medium, beam, range)
    array_store  : dict keyed by RunKey with CIR/FR arrays for .npz export
    """
    raw   = reconstruct_sweep_results(df)
    c_map = c_ref_map_from_df(df)

    rows        = []
    array_store = {}

    group_cols = ["medium_name", "beam_name", "link_range_m"]

    for (medium, beam, Z), grp in df.groupby(group_cols, observed=True):
        key    = RunKey(str(medium), str(beam), float(Z))
        result = raw[key]
        c      = c_map.get(key, float("nan"))

        # ── Physics metrics ──────────────────────────────────────────────────
        m = compute_all_metrics(result, SIM, c, float(Z))

        # ── SNR and BER ──────────────────────────────────────────────────────
        snr_lin = compute_snr(result.weights, result.n_launched,
                              noise_power=noise_power, responsivity=responsivity)
        snr_db  = float(10.0 * np.log10(snr_lin)) if snr_lin > 0 else float("-inf")
        ber     = compute_ber_ook(snr_lin, sigma_R2=sigma_r2)

        # ── Per-photon propagation statistics ───────────────────────────────
        n_captured = len(grp)
        n_launched = int(grp["n_launched"].iloc[0]) if n_captured > 0 else 0

        mean_scatters = float(grp["n_scatters"].mean()) if "n_scatters" in grp else float("nan")
        std_scatters  = float(grp["n_scatters"].std())  if "n_scatters" in grp else float("nan")

        if "excess_path_m" in grp.columns:
            mean_excess = float(grp["excess_path_m"].mean())
            std_excess  = float(grp["excess_path_m"].std())
        else:
            excess_col  = grp["path_length_m"].astype(float) - float(Z)
            mean_excess = float(excess_col.mean())
            std_excess  = float(excess_col.std())

        mean_r = float(np.hypot(grp["x_m"], grp["y_m"]).mean()) \
                 if ("x_m" in grp.columns and "y_m" in grp.columns) else float("nan")

        # ── Assemble row ─────────────────────────────────────────────────────
        ds_ns = m["delay_spread_s"] * 1e9 if np.isfinite(m["delay_spread_s"]) else float("nan")

        rows.append({
            "medium_name":        str(medium),
            "beam_name":          str(beam),
            "link_range_m":       float(Z),
            "power_dB":           round(m["power_dB"],                    4),
            "beer_lambert_dB":    round(m["beer_lambert_dB"],              4),
            "power_excess_dB":    round(m["power_dB"] - m["beer_lambert_dB"], 4),
            "delay_spread_ns":    round(ds_ns,                             6) if np.isfinite(ds_ns) else float("nan"),
            "bandwidth_mhz":      round(m["bandwidth_hz"] / 1e6,          4),
            "snr_db":             round(snr_db,                            4) if np.isfinite(snr_db) else float("-inf"),
            "ber_ook":            float(f"{ber:.6e}"),
            "capture_rate":       round(n_captured / n_launched, 8)       if n_launched > 0 else float("nan"),
            "mean_n_scatters":    round(mean_scatters, 3),
            "std_n_scatters":     round(std_scatters,  3),
            "mean_excess_path_m": round(mean_excess,   6),
            "std_excess_path_m":  round(std_excess,    6),
            "mean_r_m":           round(mean_r,        6),
            "n_launched":         n_launched,
            "n_captured":         n_captured,
            "cir_quality":        _quality(n_captured),
        })

        array_store[key] = {
            "t_axis_ns": m["t_axis"] * 1e9,
            "cir":       m["cir"],
            "freqs_mhz": m["freqs"] / 1e6,
            "fr":        m["fr"],
        }

    summary = pd.DataFrame(rows)
    return summary, array_store


# ─────────────────────────────────────────────────────────────────────────────
# .npz export
# ─────────────────────────────────────────────────────────────────────────────

def save_arrays(array_store: dict, path: str) -> None:
    """
    Save CIR and frequency-response arrays to a compressed .npz file.

    Arrays from different runs have variable length (adaptive binning),
    so each run is stored as a set of indexed arrays:
        label_{i}      run identifier string
        t_axis_ns_{i}  time axis in nanoseconds
        cir_{i}        normalised CIR
        freqs_mhz_{i}  frequency axis in MHz
        fr_{i}         |H(f)| normalised to DC

    Load example
    ------------
        data   = np.load("cir_homogeneous.npz", allow_pickle=True)
        n_runs = sum(1 for k in data if k.startswith("label_"))
        for i in range(n_runs):
            label  = str(data[f"label_{i}"])
            t_ns   = data[f"t_axis_ns_{i}"]
            cir    = data[f"cir_{i}"]
    """
    payload = {}
    for i, (key, arrays) in enumerate(sorted(
        array_store.items(),
        key=lambda kv: (kv[0].medium_name, kv[0].beam_name, kv[0].link_range),
    )):
        label = f"{key.medium_name} | {key.beam_name} | {key.link_range:.1f} m"
        payload[f"label_{i}"]     = np.array(label)
        payload[f"t_axis_ns_{i}"] = arrays["t_axis_ns"]
        payload[f"cir_{i}"]       = arrays["cir"]
        payload[f"freqs_mhz_{i}"] = arrays["freqs_mhz"]
        payload[f"fr_{i}"]        = arrays["fr"]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute all UOWC channel metrics from Parquet data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", default="./outputs",
        help="Directory containing Parquet files and where outputs are written "
             "(default: ./outputs).",
    )
    parser.add_argument(
        "--hom", default=None, metavar="PATH",
        help="Explicit path to a homogeneous Parquet file.",
    )
    parser.add_argument(
        "--inh", default=None, metavar="PATH",
        help="Explicit path to an inhomogeneous Parquet file.",
    )
    parser.add_argument(
        "--arrays", action="store_true",
        help="Also save CIR and frequency-response arrays as .npz files.",
    )
    parser.add_argument(
        "--noise-power", type=float, default=1e-7, metavar="W²",
        help="Receiver noise variance σ²_noise in A² for SNR/BER (default: 1e-7).",
    )
    parser.add_argument(
        "--responsivity", type=float, default=0.5, metavar="A/W",
        help="Photodetector responsivity R in A/W for SNR/BER (default: 0.5).",
    )
    parser.add_argument(
        "--sigma-r2", type=float, default=0.0, metavar="σ²_R",
        help="Rytov variance σ²_R for turbulence BER. "
             "0 = AWGN only (default). See module docstring for regime thresholds.",
    )
    args = parser.parse_args()

    hom_path = args.hom or os.path.join(args.out, "photons_homogeneous.parquet")
    inh_path = args.inh or os.path.join(args.out, "photons_inhomogeneous.parquet")

    os.makedirs(args.out, exist_ok=True)
    found_any = False
    t0        = time.perf_counter()

    kw = dict(
        noise_power  = args.noise_power,
        responsivity = args.responsivity,
        sigma_r2     = args.sigma_r2,
    )

    if os.path.exists(hom_path):
        found_any = True
        print(f"\nLoading {hom_path} ...", flush=True)
        df      = pd.read_parquet(hom_path)
        summary, arrays = compute_metrics_from_df(df, **kw)

        csv_path = os.path.join(args.out, "metrics_homogeneous.csv")
        summary.to_csv(csv_path, index=False)
        print(f"  Saved  {csv_path}  ({len(summary)} rows)")
        print(summary.to_string(index=False))

        if args.arrays:
            npz_path = os.path.join(args.out, "cir_homogeneous.npz")
            save_arrays(arrays, npz_path)
            print(f"  Saved  {npz_path}")

    if os.path.exists(inh_path):
        found_any = True
        print(f"\nLoading {inh_path} ...", flush=True)
        df      = pd.read_parquet(inh_path)
        summary, arrays = compute_metrics_from_df(df, **kw)

        csv_path = os.path.join(args.out, "metrics_inhomogeneous.csv")
        summary.to_csv(csv_path, index=False)
        print(f"  Saved  {csv_path}  ({len(summary)} rows)")
        print(summary.to_string(index=False))

        if args.arrays:
            npz_path = os.path.join(args.out, "cir_inhomogeneous.npz")
            save_arrays(arrays, npz_path)
            print(f"  Saved  {npz_path}")

    if not found_any:
        print(
            f"\nNo Parquet files found.\n"
            f"  Expected: {hom_path}\n"
            f"         or {inh_path}\n\n"
            f"Run the simulation first:\n"
            f"  python main.py homogeneous --out {args.out}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
