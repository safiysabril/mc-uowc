"""
main.py
=======

Simulation entry point — runs photon transport, computes channel metrics,
prints console summary tables, and saves per-photon data to Parquet.

No figures are generated here.  Run ``plot.py`` once you have accumulated
enough photons across one or more runs.

Examples
--------
Run everything (homogeneous + inhomogeneous):
    python main.py all

Homogeneous only (Clear Water / Coastal Water, two beams, five ranges):
    python main.py homogeneous

Inhomogeneous only (Woodcock delta-tracking, stratified profiles):
    python main.py inhomogeneous

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

def run_homogeneous(out_dir: str) -> dict:
    print_run_header(SIM)

    raw = run_sweep_adaptive(SIM, verbose=True)

    metrics = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key        = RunKey(water.name, beam.name, float(Z))
                metrics[key] = compute_all_metrics(raw[key], SIM, water.c, Z)

    print_summary_tables(metrics, SIM)

    c_ref_map = {
        RunKey(water.name, beam.name, float(Z)): water.c
        for water in ALL_WATERS
        for beam in ALL_BEAMS
        for Z in SIM.link_ranges_m
    }

    df           = to_dataframe(raw, c_ref_map=c_ref_map)
    parquet_path = os.path.join(out_dir, "photons_homogeneous.parquet")
    append_to_parquet(df, parquet_path)

    print(f"\nAccumulated Parquet: {parquet_path}")
    accumulated = pd.read_parquet(parquet_path)
    _photon_count_table(accumulated)

    return {"raw": raw, "metrics": metrics, "dataframe": df}


# ============================================================================
# INHOMOGENEOUS PIPELINE
# ============================================================================

def run_inhomogeneous(out_dir: str) -> dict:
    print_inhomogeneous_header(SIM, ALL_INHOMOGENEOUS_MEDIA)

    raw = run_sweep_inhomogeneous_adaptive(
        SIM, media=ALL_INHOMOGENEOUS_MEDIA, verbose=True,
    )

    metrics = {}
    for medium in ALL_INHOMOGENEOUS_MEDIA:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key        = RunKey(medium.name, beam.name, float(Z))
                metrics[key] = compute_all_metrics(raw[key], SIM, medium.c_max, Z)

    print_inhomogeneous_summary(metrics, SIM, ALL_INHOMOGENEOUS_MEDIA)

    c_ref_map = {
        RunKey(medium.name, beam.name, float(Z)): medium.c_max
        for medium in ALL_INHOMOGENEOUS_MEDIA
        for beam in ALL_BEAMS
        for Z in SIM.link_ranges_m
    }

    df           = to_dataframe(raw, c_ref_map=c_ref_map)
    parquet_path = os.path.join(out_dir, "photons_inhomogeneous.parquet")
    append_to_parquet(df, parquet_path)

    print(f"\nAccumulated Parquet: {parquet_path}")
    accumulated = pd.read_parquet(parquet_path)
    _photon_count_table(accumulated)

    return {"raw": raw, "metrics": metrics, "dataframe": df}


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
        choices=["all", "homogeneous", "inhomogeneous"],
        help="Which simulation to run.",
    )
    parser.add_argument(
        "--out",
        default="./outputs",
        help="Output directory (default: ./outputs).  Parquet files in this "
             "directory are appended to on every run.",
    )

    args    = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()

    if args.mode in ("all", "homogeneous"):
        run_homogeneous(out_dir)

    if args.mode in ("all", "inhomogeneous"):
        run_inhomogeneous(out_dir)

    print(f"Total runtime: {time.perf_counter() - t0:.1f} s")
    print("Done.  Generate figures with:  python plot.py --out", out_dir)


if __name__ == "__main__":
    main()
