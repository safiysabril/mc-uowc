"""
uowc/main.py
============

Flexible application entry point.

Examples
--------
Run everything:
    python -m uowc.main all

Run only homogeneous simulation:
    python -m uowc.main homogeneous

Run only inhomogeneous simulation:
    python -m uowc.main inhomogeneous

Generate only plots:
    python -m uowc.main plots

Generate only diagnostics:
    python -m uowc.main diagnostics

Custom output directory:
    python -m uowc.main all --out ./outputs

Pipeline
--------
  1. Run adaptive simulations (homogeneous + inhomogeneous)
  2. Compute channel metrics (CIR, delay spread, bandwidth)
  3. Convert RunResults → Pandas DataFrame
  4. Compute capture statistics (with correct n_launched denominators)
  5. Save DataFrame to Parquet (Strategy A — full in-memory then write)
  6. Generate channel figures (plotting module)
  7. Generate diagnostic figures (analysis.plots module)

"""

from __future__ import annotations

import argparse
import os
import time

from uowc.config import SIM, ALL_WATERS, ALL_BEAMS, RECEIVER
from uowc.medium import ALL_INHOMOGENEOUS_MEDIA

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

from uowc.plotting import (
    plot_all,
    plot_all_inhomogeneous,
)

from uowc.analysis import (
    to_dataframe,
    to_parquet,
    append_to_parquet,
    capture_statistics_with_launched,
    reconstruct_sweep_results,
    launched_map_from_df,
    c_ref_map_from_df,
)
import pandas as pd

_CIR_THRESHOLDS = [(2_000, "good"), (250, "sparse"), (50, "very sparse"), (0, "TOO FEW")]


def _photon_count_table(df: pd.DataFrame) -> None:
    print(f"\n{'Medium':<28} {'Beam':<24} {'Range':>7}  {'Captured':>10}  Status")
    print("-" * 82)
    for (medium, beam, Z), grp in df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        n = len(grp)
        status = next(label for thr, label in _CIR_THRESHOLDS if n >= thr)
        flag   = " <-- run more" if n < 250 else ""
        print(f"{str(medium):<28} {str(beam):<24} {Z:>7.1f}  {n:>10,}  {status}{flag}")
    print()

from uowc.analysis.plots import plot_all_diagnostics


# ============================================================================
# HOMOGENEOUS PIPELINE
# ============================================================================

def run_homogeneous(out_dir: str):
    print_run_header(SIM)

    raw_hom = run_sweep_adaptive(SIM, verbose=True)

    metrics_hom = {}

    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:

                key = RunKey(water.name, beam.name, float(Z))

                metrics_hom[key] = compute_all_metrics(
                    raw_hom[key],
                    SIM,
                    water.c,
                    Z,
                )

    print_summary_tables(metrics_hom, SIM)

    plot_all(
        metrics_hom,
        SIM,
        save_dir=out_dir,
    )

    c_ref_hom = {
        RunKey(water.name, beam.name, float(Z)): water.c
        for water in ALL_WATERS
        for beam in ALL_BEAMS
        for Z in SIM.link_ranges_m
    }

    df_hom = to_dataframe(raw_hom, c_ref_map=c_ref_hom)

    parquet_path = os.path.join(
        out_dir,
        "photons_homogeneous.parquet",
    )

    append_to_parquet(df_hom, parquet_path)

    launched_hom = launched_map_from_df(df_hom)

    stats_hom = capture_statistics_with_launched(
        df_hom,
        launched_hom,
    )

    diag_dir = os.path.join(out_dir, "diagnostics")

    plot_all_diagnostics(
        df_hom,
        stats_hom,
        diag_dir,
        aperture_radius_m=RECEIVER.aperture_radius_m,
    )

    return {
        "raw": raw_hom,
        "metrics": metrics_hom,
        "dataframe": df_hom,
        "stats": stats_hom,
    }


# ============================================================================
# INHOMOGENEOUS PIPELINE
# ============================================================================

def run_inhomogeneous(out_dir: str):
    print_inhomogeneous_header(
        SIM,
        ALL_INHOMOGENEOUS_MEDIA,
    )

    raw_inh = run_sweep_inhomogeneous_adaptive(
        SIM,
        media=ALL_INHOMOGENEOUS_MEDIA,
        verbose=True,
    )

    metrics_inh = {}

    for medium in ALL_INHOMOGENEOUS_MEDIA:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:

                key = RunKey(
                    medium.name,
                    beam.name,
                    float(Z),
                )

                metrics_inh[key] = compute_all_metrics(
                    raw_inh[key],
                    SIM,
                    medium.c_max,
                    Z,
                )

    print_inhomogeneous_summary(
        metrics_inh,
        SIM,
        ALL_INHOMOGENEOUS_MEDIA,
    )

    plot_all_inhomogeneous(
        metrics_inh,
        SIM,
        ALL_INHOMOGENEOUS_MEDIA,
        save_dir=out_dir,
    )

    c_ref_inh = {
        RunKey(medium.name, beam.name, float(Z)): medium.c_max
        for medium in ALL_INHOMOGENEOUS_MEDIA
        for beam in ALL_BEAMS
        for Z in SIM.link_ranges_m
    }

    df_inh = to_dataframe(raw_inh, c_ref_map=c_ref_inh)

    parquet_path = os.path.join(
        out_dir,
        "photons_inhomogeneous.parquet",
    )

    append_to_parquet(df_inh, parquet_path)

    launched_inh = launched_map_from_df(df_inh)

    stats_inh = capture_statistics_with_launched(
        df_inh,
        launched_inh,
    )

    diag_dir = os.path.join(out_dir, "diagnostics")

    plot_all_diagnostics(
        df_inh,
        stats_inh,
        diag_dir,
        medium_name=ALL_INHOMOGENEOUS_MEDIA[0].name,
        aperture_radius_m=RECEIVER.aperture_radius_m,
    )

    return {
        "raw": raw_inh,
        "metrics": metrics_inh,
        "dataframe": df_inh,
        "stats": stats_inh,
    }


# ============================================================================
# PLOTS-ONLY PIPELINE  (no simulation — load from saved Parquet)
# ============================================================================

def plot_from_parquet(out_dir: str) -> None:
    """
    Regenerate all figures from previously saved Parquet files.

    Looks for:
      {out_dir}/photons_homogeneous.parquet
      {out_dir}/photons_inhomogeneous.parquet

    At least one must exist.  For each file found, reconstructs the
    RunResult dicts, recomputes channel metrics, and calls the same
    plotting functions used in the full simulation pipeline.
    """
    hom_path = os.path.join(out_dir, "photons_homogeneous.parquet")
    inh_path = os.path.join(out_dir, "photons_inhomogeneous.parquet")
    diag_dir = os.path.join(out_dir, "diagnostics")

    if not os.path.exists(hom_path) and not os.path.exists(inh_path):
        raise FileNotFoundError(
            f"No Parquet files found in {out_dir!r}. "
            "Run the simulation first with 'homogeneous', 'inhomogeneous', or 'all'."
        )

    if os.path.exists(hom_path):
        print(f"Loading {hom_path} ...", flush=True)
        df_hom  = pd.read_parquet(hom_path)
        _photon_count_table(df_hom)
        raw_hom = reconstruct_sweep_results(df_hom)
        c_map   = c_ref_map_from_df(df_hom)

        metrics_hom = {
            key: compute_all_metrics(result, SIM, c_map.get(key, 1.0), key.link_range)
            for key, result in raw_hom.items()
        }

        print_summary_tables(metrics_hom, SIM)
        plot_all(metrics_hom, SIM, save_dir=out_dir)

        stats_hom = capture_statistics_with_launched(df_hom, launched_map_from_df(df_hom))
        plot_all_diagnostics(
            df_hom, stats_hom, diag_dir,
            aperture_radius_m=RECEIVER.aperture_radius_m,
        )
        print("Homogeneous figures done.", flush=True)

    if os.path.exists(inh_path):
        print(f"Loading {inh_path} ...", flush=True)
        df_inh  = pd.read_parquet(inh_path)
        _photon_count_table(df_inh)
        raw_inh = reconstruct_sweep_results(df_inh)
        c_map   = c_ref_map_from_df(df_inh)

        metrics_inh = {
            key: compute_all_metrics(result, SIM, c_map.get(key, 1.0), key.link_range)
            for key, result in raw_inh.items()
        }

        print_inhomogeneous_summary(metrics_inh, SIM, ALL_INHOMOGENEOUS_MEDIA)
        plot_all_inhomogeneous(metrics_inh, SIM, ALL_INHOMOGENEOUS_MEDIA, save_dir=out_dir)

        stats_inh = capture_statistics_with_launched(df_inh, launched_map_from_df(df_inh))
        plot_all_diagnostics(
            df_inh, stats_inh, diag_dir,
            medium_name=ALL_INHOMOGENEOUS_MEDIA[0].name,
            aperture_radius_m=RECEIVER.aperture_radius_m,
        )
        print("Inhomogeneous figures done.", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UOWC Simulation Runner",
    )

    parser.add_argument(
        "mode",
        choices=[
            "all",
            "homogeneous",
            "inhomogeneous",
            "plots",
        ],
        help=(
            "Which pipeline to run. "
            "'plots' regenerates all figures from saved Parquet without re-simulating."
        ),
    )

    parser.add_argument(
        "--out",
        default="./outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()

    if args.mode == "all":
        run_homogeneous(out_dir)
        run_inhomogeneous(out_dir)

    elif args.mode == "homogeneous":
        run_homogeneous(out_dir)

    elif args.mode == "inhomogeneous":
        run_inhomogeneous(out_dir)

    elif args.mode == "plots":
        plot_from_parquet(out_dir)

    elapsed = time.perf_counter() - t0

    print(f"\nTotal runtime: {elapsed:.1f} s")
    print("Done.\n")


if __name__ == "__main__":
    main()