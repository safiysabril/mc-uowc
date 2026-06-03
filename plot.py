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
from uowc.medium import ALL_INHOMOGENEOUS_MEDIA          # Model 2 media
from uowc.turbulence import ALL_COUPLED_MEDIA            # Model 3 media
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


def _plot_one(df: pd.DataFrame, fig_dir: str, kind: str, media=None) -> None:
    """Reconstruct metrics from ``df`` and write all figures to ``fig_dir``.

    ``kind`` is "homogeneous" (Model 1), "inhomogeneous" (Model 2) or
    "turbulent" (Model 3).  Models 2 and 3 share the inhomogeneous figure family
    (fig_inh*), so Model 3 is written to its own ``fig_dir`` (a ``turbulent/``
    subdirectory) to avoid overwriting Model 2's figures.  ``media`` is the list
    of media profiles for the inhomogeneous families (Model 2: layered/gradient;
    Model 3: coupled-ocean).
    """
    os.makedirs(fig_dir, exist_ok=True)
    _photon_count_table(df)
    raw   = reconstruct_sweep_results(df)
    c_map = c_ref_map_from_df(df)

    metrics = {
        key: compute_all_metrics(result, SIM, c_map.get(key, 1.0), key.link_range)
        for key, result in raw.items()
    }

    diag_dir = os.path.join(fig_dir, "diagnostics")
    stats    = capture_statistics_with_launched(df, launched_map_from_df(df))

    if kind == "homogeneous":
        print_summary_tables(metrics, SIM)
        plot_all(metrics, SIM, save_dir=fig_dir)
    else:
        print_inhomogeneous_summary(metrics, SIM, media)
        plot_all_inhomogeneous(metrics, SIM, media, save_dir=fig_dir)

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
        help="Explicit path to an inhomogeneous (Model 2) Parquet file. "
             "Overrides the default <out>/photons_inhomogeneous.parquet.",
    )
    parser.add_argument(
        "--turb", default=None, metavar="PATH",
        help="Explicit path to a turbulent (Model 3) Parquet file. "
             "Overrides the default <out>/photons_turbulent.parquet. "
             "Model 3 figures are written to <out>/turbulent/.",
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
        # After merging, plot from the merged file if it matches a known name.
        # Check "turbulent" and "inhomogeneous" before "homogeneous" since the
        # latter is a substring of the former.
        merged_name = os.path.basename(args.merge_out)
        if "turbulent" in merged_name and not args.turb:
            args.turb = args.merge_out
        elif "inhomogeneous" in merged_name and not args.inh:
            args.inh = args.merge_out
        elif "homogeneous" in merged_name and not args.hom:
            args.hom = args.merge_out

    # ── Resolve Parquet paths ─────────────────────────────────────────────────
    hom_path  = args.hom  or os.path.join(args.out, "photons_homogeneous.parquet")
    inh_path  = args.inh  or os.path.join(args.out, "photons_inhomogeneous.parquet")
    turb_path = args.turb or os.path.join(args.out, "photons_turbulent.parquet")

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
        print(f"  Homogeneous (Model 1) figures written to {args.out}/")

    if os.path.exists(inh_path):
        found_any = True
        print(f"\nLoading inhomogeneous (Model 2) data: {inh_path}")
        df = pd.read_parquet(inh_path)
        n_photons = len(df)
        runs = df.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ).size()
        print(f"  {n_photons:,} captured photons across {len(runs)} run combinations")
        _plot_one(df, args.out, "inhomogeneous", media=ALL_INHOMOGENEOUS_MEDIA)
        print(f"  Inhomogeneous (Model 2) figures written to {args.out}/")

    if os.path.exists(turb_path):
        found_any = True
        turb_dir = os.path.join(args.out, "turbulent")
        print(f"\nLoading turbulent (Model 3) data: {turb_path}")
        df = pd.read_parquet(turb_path)
        n_photons = len(df)
        runs = df.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ).size()
        print(f"  {n_photons:,} captured photons across {len(runs)} run combinations")
        _plot_one(df, turb_dir, "turbulent", media=ALL_COUPLED_MEDIA)
        print(f"  Turbulent (Model 3) figures written to {turb_dir}/")

    if not found_any:
        print(
            f"\nNo Parquet files found.\n"
            f"  Expected: {hom_path}\n"
            f"         or {inh_path}\n"
            f"         or {turb_path}\n\n"
            f"Run the simulation first:\n"
            f"  python main.py homogeneous   --out {args.out}   # Model 1\n"
            f"  python main.py inhomogeneous --out {args.out}   # Model 2\n"
            f"  python main.py turbulent     --out {args.out}   # Model 3\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
