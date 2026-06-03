"""
plot.py
=======
Standalone figure generator — regenerate all figures from saved Parquet
data without re-running the simulation.

Usage
-----
# Regenerate figures from the default output directory:
    python plot.py

# Specify a different output directory:
    python plot.py --out ./my_outputs

# Merge two Parquet files first, then plot (accumulate photons across runs):
    python plot.py --merge outputs/photons_homogeneous.parquet \\
                           outputs2/photons_homogeneous.parquet \\
                   --merge-out outputs/merged_homogeneous.parquet

# Plot only a specific Parquet file:
    python plot.py --hom outputs/photons_homogeneous.parquet
    python plot.py --inh outputs/photons_inhomogeneous.parquet

Why merge?
----------
Turbid water at long range captures very few photons per run.  Running
the simulation multiple times and merging the Parquet files accumulates
more photons without increasing per-run RAM.  The merged file tracks the
total ``n_launched`` correctly so power normalization remains exact.

IMPORTANT — each run must use a different RNG seed, otherwise you merge
identical photons (the power stays correct but the CIR / delay-spread gain
no real statistics).  ``main.py`` draws a fresh random seed per run by
default, so separate invocations are already independent.  Note that
appending to one Parquet file with ``main.py`` (the default workflow) is
usually simpler than this explicit merge.

Workflow example
----------------
  # Run 3 times — main.py uses a different random seed each time by default:
  python main.py homogeneous --out outputs_run1
  python main.py homogeneous --out outputs_run2
  python main.py homogeneous --out outputs_run3

  # Merge and plot:
  python plot.py --merge outputs_run1/photons_homogeneous.parquet \\
                         outputs_run2/photons_homogeneous.parquet \\
                         outputs_run3/photons_homogeneous.parquet \\
                 --merge-out outputs_merged/photons_homogeneous.parquet \\
                 --out outputs_merged
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from uowc.config import SIM, RECEIVER
# Must match the media simulated by main.py — the turbulence-coupled profiles.
from uowc.turbulence import ALL_COUPLED_MEDIA as ALL_INHOMOGENEOUS_MEDIA
from uowc.metrics import compute_all_metrics
from uowc.plotting import plot_all, plot_all_inhomogeneous
from uowc.reporting import print_summary_tables, print_inhomogeneous_summary
from uowc.analysis import (
    reconstruct_sweep_results,
    launched_map_from_df,
    c_ref_map_from_df,
    capture_statistics_with_launched,
    merge_parquet_files,
)
from uowc.analysis.plots import plot_all_diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CIR_THRESHOLDS = [(2_000, "good"), (250, "sparse"), (50, "very sparse"), (0, "TOO FEW")]


def _photon_count_table(df: pd.DataFrame) -> None:
    """Print a per-run photon count table with quality indicators."""
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


def _plot_one(df: pd.DataFrame, out_dir: str, medium_type: str) -> None:
    """Reconstruct metrics from ``df`` and write all figures to ``out_dir``."""
    _photon_count_table(df)
    raw   = reconstruct_sweep_results(df)
    c_map = c_ref_map_from_df(df)

    metrics = {
        key: compute_all_metrics(result, SIM, c_map.get(key, 1.0), key.link_range)
        for key, result in raw.items()
    }

    diag_dir = os.path.join(out_dir, "diagnostics")
    stats    = capture_statistics_with_launched(df, launched_map_from_df(df))

    if medium_type == "homogeneous":
        print_summary_tables(metrics, SIM)
        plot_all(metrics, SIM, save_dir=out_dir)
    else:
        print_inhomogeneous_summary(metrics, SIM, ALL_INHOMOGENEOUS_MEDIA)
        plot_all_inhomogeneous(metrics, SIM, ALL_INHOMOGENEOUS_MEDIA, save_dir=out_dir)

    plot_all_diagnostics(
        df, stats, diag_dir,
        aperture_radius_m=RECEIVER.aperture_radius_m,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate UOWC figures from saved Parquet data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", default="./outputs",
        help="Directory containing Parquet files and where figures are written "
             "(default: ./outputs).",
    )
    parser.add_argument(
        "--hom", default=None, metavar="PATH",
        help="Explicit path to a homogeneous Parquet file. "
             "Overrides the default <out>/photons_homogeneous.parquet.",
    )
    parser.add_argument(
        "--inh", default=None, metavar="PATH",
        help="Explicit path to an inhomogeneous Parquet file. "
             "Overrides the default <out>/photons_inhomogeneous.parquet.",
    )
    parser.add_argument(
        "--merge", nargs="+", metavar="PATH",
        help="Two or more Parquet files to merge before plotting. "
             "Requires --merge-out.",
    )
    parser.add_argument(
        "--merge-out", default=None, metavar="PATH",
        help="Output path for the merged Parquet file.",
    )
    args = parser.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        if not args.merge_out:
            parser.error("--merge requires --merge-out to specify the output path.")
        if len(args.merge) < 2:
            parser.error("--merge requires at least two input files.")
        print(f"Merging {len(args.merge)} Parquet files → {args.merge_out} ...")
        merge_parquet_files(args.merge, args.merge_out)
        print("Merge done.")
        # After merging, plot from the merged file if it matches a known name
        merged_name = os.path.basename(args.merge_out)
        if "homogeneous" in merged_name and not args.hom:
            args.hom = args.merge_out
        elif "inhomogeneous" in merged_name and not args.inh:
            args.inh = args.merge_out

    # ── Resolve Parquet paths ─────────────────────────────────────────────────
    hom_path = args.hom or os.path.join(args.out, "photons_homogeneous.parquet")
    inh_path = args.inh or os.path.join(args.out, "photons_inhomogeneous.parquet")

    found_any = False
    t0 = time.perf_counter()

    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(hom_path):
        found_any = True
        print(f"\nLoading homogeneous data: {hom_path}")
        df = pd.read_parquet(hom_path)
        n_photons = len(df)
        runs = df.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ).size()
        print(f"  {n_photons:,} captured photons across {len(runs)} run combinations")
        _plot_one(df, args.out, "homogeneous")
        print(f"  Homogeneous figures written to {args.out}/")

    if os.path.exists(inh_path):
        found_any = True
        print(f"\nLoading inhomogeneous data: {inh_path}")
        df = pd.read_parquet(inh_path)
        n_photons = len(df)
        runs = df.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ).size()
        print(f"  {n_photons:,} captured photons across {len(runs)} run combinations")
        _plot_one(df, args.out, "inhomogeneous")
        print(f"  Inhomogeneous figures written to {args.out}/")

    if not found_any:
        print(
            f"\nNo Parquet files found.\n"
            f"  Expected: {hom_path}\n"
            f"         or {inh_path}\n\n"
            f"Run the simulation first:\n"
            f"  python main.py homogeneous --out {args.out}\n"
            f"  python main.py inhomogeneous --out {args.out}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
