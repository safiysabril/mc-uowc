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

from uowc.config import SIM, ALL_WATERS, ALL_BEAMS
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
    capture_statistics_with_launched,
)

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

    df_hom = to_dataframe(raw_hom)

    parquet_path = os.path.join(
        out_dir,
        "photons_homogeneous.parquet",
    )

    to_parquet(df_hom, parquet_path)

    launched_hom = {
        k: v.n_launched
        for k, v in raw_hom.items()
    }

    stats_hom = capture_statistics_with_launched(
        df_hom,
        launched_hom,
    )

    diag_dir = os.path.join(out_dir, "diagnostics")

    plot_all_diagnostics(
        df_hom,
        stats_hom,
        diag_dir,
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

    df_inh = to_dataframe(raw_inh)

    parquet_path = os.path.join(
        out_dir,
        "photons_inhomogeneous.parquet",
    )

    to_parquet(df_inh, parquet_path)

    launched_inh = {
        k: v.n_launched
        for k, v in raw_inh.items()
    }

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
    )

    return {
        "raw": raw_inh,
        "metrics": metrics_inh,
        "dataframe": df_inh,
        "stats": stats_inh,
    }


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
        ],
        help="Which pipeline to run",
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

    elapsed = time.perf_counter() - t0

    print(f"\nTotal runtime: {elapsed:.1f} s")
    print("Done.\n")


if __name__ == "__main__":
    main()