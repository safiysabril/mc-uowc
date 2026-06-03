"""
main.py
=======

Simulation entry point — runs photon transport, computes channel metrics,
prints console summary tables, and saves per-photon data to Parquet.

No figures are generated here.  Run ``plot.py`` once you have accumulated
enough photons across one or more runs.

Three comparison models
-----------------------
  Model 1  (homogeneous)   constant a, b, c — no turbulence.
  Model 2  (inhomogeneous) depth-varying a, b, c with layered/gradient
                           boundaries (Woodcock delta-tracking) — no turbulence.
  Model 3  (turbulent)     Model 2's IOP structure + ocean optical turbulence
                           (phase-screen angular kicks + scintillation fading).

Each model writes its own Parquet file so the three scenarios stay separate:
  photons_homogeneous.parquet · photons_inhomogeneous.parquet · photons_turbulent.parquet

Examples
--------
Run all three models:
    python main.py all

One model at a time:
    python main.py homogeneous     # Model 1
    python main.py inhomogeneous   # Model 2
    python main.py turbulent       # Model 3

Custom output directory:
    python main.py all --out ./results

Each run *accumulates* photons into the existing Parquet file in --out.
Run the command multiple times until the photon-count table printed at
the end of each run shows "good" for all combinations you care about,
then generate figures with:
    python plot.py --out ./results

Pipeline (per run)
------------------
  1. Run adaptive Monte Carlo simulation
  2. Compute channel metrics (power, CIR, delay spread, bandwidth)
  3. Print per-run summary table to stdout
  4. Convert RunResults → Pandas DataFrame (with n_launched and c_ref)
  5. Accumulate photons into Parquet (append if file exists, else create)
  6. Print per-run photon count table so you can judge when to plot
"""

from __future__ import annotations

import argparse
import os
import secrets
import time
from dataclasses import replace

from uowc.config import SIM, ALL_WATERS, ALL_BEAMS
# Model 2 — inhomogeneous, turbulence-free: varying a,b,c with layered/gradient
# boundaries, transported by Woodcock delta-tracking.  These plain MediumProfile
# objects carry no turbulence, so the dispatch runs the turbulence-free worker.
from uowc.medium import ALL_INHOMOGENEOUS_MEDIA
# Model 3 — inhomogeneous + turbulence: CoupledOceanMedium profiles whose depth
# layers bundle aligned IOP and turbulence (ε, χ_T).  The dispatch detects the
# turbulence and runs the turbulent Woodcock worker (phase-screen angular kicks
# along the path + scintillation fading at the receiver).
from uowc.turbulence import ALL_COUPLED_MEDIA as ALL_TURBULENT_MEDIA

from uowc.reporting import (
    print_run_header,
    print_summary_tables,
    print_inhomogeneous_header,
    print_inhomogeneous_summary,
)

from uowc.simulation import (
    RunKey,
    run_sweep_adaptive,
    run_sweep_inhomogeneous_adaptive,
)

from uowc.metrics import compute_all_metrics

from uowc.analysis import (
    to_dataframe,
    append_to_parquet,
)


# ─────────────────────────────────────────────────────────────────────────────
# Photon count summary  (printed after each run so you know when to plot)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

_CIR_THRESHOLDS = [(2_000, "good"), (250, "sparse"), (50, "very sparse"), (0, "TOO FEW")]


def _photon_count_table(df: pd.DataFrame) -> None:
    """Print a per-run photon count table with CIR quality indicators."""
    print(f"\n{'Medium':<28} {'Beam':<24} {'Range':>7}  {'Captured':>10}  Status")
    print("-" * 82)
    for (medium, beam, Z), grp in df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        n      = len(grp)
        status = next(label for thr, label in _CIR_THRESHOLDS if n >= thr)
        flag   = "  <-- run more" if n < 250 else ""
        print(f"{str(medium):<28} {str(beam):<24} {Z:>7.1f}  {n:>10,}  {status}{flag}")
    print()


# ============================================================================
# HOMOGENEOUS PIPELINE
# ============================================================================

def run_homogeneous(out_dir: str, cfg) -> dict:
    print_run_header(cfg)

    raw = run_sweep_adaptive(cfg, verbose=True)

    metrics = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in cfg.link_ranges_m:
                key        = RunKey(water.name, beam.name, float(Z))
                metrics[key] = compute_all_metrics(raw[key], cfg, water.c, Z)

    print_summary_tables(metrics, cfg)

    c_ref_map = {
        RunKey(water.name, beam.name, float(Z)): water.c
        for water in ALL_WATERS
        for beam in ALL_BEAMS
        for Z in cfg.link_ranges_m
    }

    df           = to_dataframe(raw, c_ref_map=c_ref_map)
    parquet_path = os.path.join(out_dir, "photons_homogeneous.parquet")
    append_to_parquet(df, parquet_path)

    print(f"\nAccumulated Parquet: {parquet_path}")
    accumulated = pd.read_parquet(parquet_path)
    _photon_count_table(accumulated)

    return {"raw": raw, "metrics": metrics, "dataframe": df}


# ============================================================================
# INHOMOGENEOUS PIPELINE  (shared by Model 2 and Model 3)
# ============================================================================
#
# Model 2 and Model 3 use the *same* Woodcock delta-tracking machinery — they
# differ only in (a) the media set and (b) whether those media carry turbulence.
# The dispatch layer decides turbulence per medium, so this one helper serves
# both; a distinct seed_offset keeps their RNG sub-streams independent within a
# single ``main.py all`` invocation.

def _run_inhomogeneous_family(
    out_dir:      str,
    cfg,
    media,
    parquet_name: str,
    seed_offset:  int,
) -> dict:
    print_inhomogeneous_header(cfg, media)

    raw = run_sweep_inhomogeneous_adaptive(
        cfg, media=media, seed_offset=seed_offset, verbose=True,
    )

    metrics = {}
    for medium in media:
        for beam in ALL_BEAMS:
            for Z in cfg.link_ranges_m:
                key        = RunKey(medium.name, beam.name, float(Z))
                metrics[key] = compute_all_metrics(raw[key], cfg, medium.c_max, Z)

    print_inhomogeneous_summary(metrics, cfg, media)

    c_ref_map = {
        RunKey(medium.name, beam.name, float(Z)): medium.c_max
        for medium in media
        for beam in ALL_BEAMS
        for Z in cfg.link_ranges_m
    }

    df           = to_dataframe(raw, c_ref_map=c_ref_map)
    parquet_path = os.path.join(out_dir, parquet_name)
    append_to_parquet(df, parquet_path)

    print(f"\nAccumulated Parquet: {parquet_path}")
    accumulated = pd.read_parquet(parquet_path)
    _photon_count_table(accumulated)

    return {"raw": raw, "metrics": metrics, "dataframe": df}


def run_inhomogeneous(out_dir: str, cfg) -> dict:
    """Model 2 — inhomogeneous IOPs, turbulence-free (seed offset 1)."""
    return _run_inhomogeneous_family(
        out_dir, cfg, ALL_INHOMOGENEOUS_MEDIA,
        parquet_name="photons_inhomogeneous.parquet", seed_offset=1,
    )


def run_turbulent(out_dir: str, cfg) -> dict:
    """Model 3 — inhomogeneous IOPs + turbulence (seed offset 2)."""
    return _run_inhomogeneous_family(
        out_dir, cfg, ALL_TURBULENT_MEDIA,
        parquet_name="photons_turbulent.parquet", seed_offset=2,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "UOWC Monte Carlo simulation — computes channel metrics and "
            "accumulates photon data into Parquet.  Run plot.py separately "
            "to generate figures once enough photons are captured."
        ),
    )
    parser.add_argument(
        "mode",
        choices=["all", "homogeneous", "inhomogeneous", "turbulent"],
        help="Which scenario to run.  "
             "homogeneous = Model 1 (constant IOPs, no turbulence); "
             "inhomogeneous = Model 2 (layered/gradient IOPs via Woodcock, no "
             "turbulence); "
             "turbulent = Model 3 (Model 2's IOP structure + turbulence); "
             "all = run all three.",
    )
    parser.add_argument(
        "--out",
        default="./outputs",
        help="Output directory (default: ./outputs).  Parquet files in this "
             "directory are appended to on every run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed.  IMPORTANT for the accumulation workflow: every "
             "run uses a *different* seed so that appending to the Parquet file "
             "adds statistically independent photons.  If omitted, a fresh random "
             "seed is drawn from system entropy and printed (so you can reproduce "
             "a run later with --seed).  Passing the same seed twice reproduces "
             "the same photons exactly — do NOT reuse a seed across accumulation "
             "runs or you will stack duplicate photons (false convergence).",
    )

    args    = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # ── Per-run seed ──────────────────────────────────────────────────────────
    # The default SIM.master_seed is fixed, which would make every invocation
    # produce byte-identical photons; appending those to the Parquet file just
    # duplicates samples (the power stays correct but the CIR / delay-spread
    # gain no real statistics).  Draw a unique seed per run unless the user pins
    # one explicitly, and always report it for reproducibility.
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    cfg  = replace(SIM, master_seed=seed)
    if args.seed is None:
        print(f"Master seed: {seed}  (random — pass --seed {seed} to reproduce this run)")
    else:
        print(f"Master seed: {seed}  (user-specified)")

    t0 = time.perf_counter()

    if args.mode in ("all", "homogeneous"):
        run_homogeneous(out_dir, cfg)        # Model 1

    if args.mode in ("all", "inhomogeneous"):
        run_inhomogeneous(out_dir, cfg)      # Model 2

    if args.mode in ("all", "turbulent"):
        run_turbulent(out_dir, cfg)          # Model 3

    print(f"Total runtime: {time.perf_counter() - t0:.1f} s")
    print("Done.  Generate figures with:  python plot.py --out", out_dir)


if __name__ == "__main__":
    main()
