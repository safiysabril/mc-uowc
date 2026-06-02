"""
uowc.analysis.plots
===================
Diagnostic plots generated from Pandas DataFrames.

Separation-of-Concern role
---------------------------
  This module sits strictly downstream of `uowc.analysis`.
  It receives DataFrames and produces figures — no simulation logic,
  no NumPy transport, no metrics.

  The Pandas/Matplotlib boundary:
    Pandas  — groupby, agg, filtering, pivot_table (done before plotting)
    Matplotlib/Seaborn — rendering only; no data transformation

Figures produced
----------------
  fig_diag1_capture_rate.png      — capture rate % vs depth (log-y)
  fig_diag2_power_vs_depth.png    — received power dB vs depth with 95% CI
  fig_diag3_tof_histograms.png    — overlaid ToF histograms per depth
  fig_diag4_scatter_profile.png   — mean scattering events vs depth
  fig_diag5_spatial_spread.png    — (x, y) scatter at each depth
  fig_diag6_excess_path.png       — excess path length vs depth
"""

from __future__ import annotations
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from uowc.analysis import (
    capture_statistics_with_launched,
    tof_histograms,
    scattering_profile,
)
from uowc.simulation import RunKey


_GRID_KW  = dict(alpha=0.22, linestyle="--")
_TICK_SZ  = 8
_LABEL_SZ = 9
_TITLE_SZ = 9


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=_LABEL_SZ)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SZ)
    ax.set_title(title,   fontsize=_TITLE_SZ, fontweight="bold")
    ax.grid(True, **_GRID_KW)
    ax.tick_params(labelsize=_TICK_SZ)


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag1 — Capture rate vs depth
# ─────────────────────────────────────────────────────────────────────────────

def plot_capture_rate(
    stats_df:  pd.DataFrame,
    save_dir:  str,
) -> None:
    """
    Log-scale capture rate (%) vs link range for every (medium, beam) combo.

    `stats_df` must come from capture_statistics_with_launched() so it has
    n_captured, n_launched, capture_rate_pct columns.

    This is the primary diagnostic for "not enough photons at depth":
    the exponential drop is expected — the question is how steep it is and
    where it falls below your statistical threshold.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    mediums = stats_df["medium_name"].unique()
    beams   = stats_df["beam_name"].unique()
    colours = plt.colormaps["tab10"](np.linspace(0, 0.9, len(mediums)))
    markers = ["o", "s", "D", "^", "v"]

    for ci, medium in enumerate(mediums):
        for bi, beam in enumerate(beams):
            sub = stats_df[
                (stats_df["medium_name"] == medium) &
                (stats_df["beam_name"]   == beam)
            ].sort_values("link_range_m")
            if sub.empty:
                continue
            label = f"{medium} | {beam}"
            ax.semilogy(
                sub["link_range_m"],
                sub["capture_rate_pct"].clip(lower=1e-6),
                marker=markers[bi % len(markers)],
                color=colours[ci],
                linestyle="-" if bi == 0 else "--",
                linewidth=1.6, label=label,
            )

    ax.axhline(0.001, color="red", linestyle=":", linewidth=1.0,
               label="0.001% threshold")
    _style(ax, "Link Range (m)", "Capture Rate (%)",
           "Photon Capture Rate vs Depth  (log scale)")
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4g}"))
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_diag1_capture_rate.png"))


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag2 — Received power with 95% CI
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_with_ci(
    stats_df: pd.DataFrame,
    save_dir: str,
) -> None:
    """
    Received power (dB) vs depth with shaded 95% confidence band.

    CI comes from the delta-method columns ci_95_low_dB / ci_95_high_dB
    in stats_df produced by capture_statistics_with_launched().  Wide bands
    at deep ranges visually communicate where you have too few photons for
    reliable estimates.
    """
    mediums = stats_df["medium_name"].unique()
    beams   = stats_df["beam_name"].unique()
    n_beams = len(beams)

    fig, axes = plt.subplots(1, n_beams, figsize=(7 * n_beams, 5), sharey=False)
    if n_beams == 1:
        axes = [axes]
    fig.suptitle("Received Power vs Depth  |  with 95% Confidence Interval",
                 fontsize=11, fontweight="bold")

    colours = plt.colormaps["tab10"](np.linspace(0, 0.9, len(mediums)))
    for ax, beam in zip(axes, beams):
        for ci, medium in enumerate(mediums):
            sub = stats_df[
                (stats_df["medium_name"] == medium) &
                (stats_df["beam_name"]   == beam)
            ].sort_values("link_range_m")
            if sub.empty:
                continue
            x  = sub["link_range_m"].values
            y  = sub["power_dB"].values
            lo = sub["ci_95_low_dB"].values
            hi = sub["ci_95_high_dB"].values
            col = colours[ci]
            ax.plot(x, y, color=col, marker="o", linewidth=1.8, label=medium)
            # Only shade where CI is valid (finite)
            valid = np.isfinite(lo) & np.isfinite(hi)
            if valid.any():
                ax.fill_between(x[valid], lo[valid], hi[valid],
                                alpha=0.15, color=col)
        _style(ax, "Link Range (m)", "Normalised Power (dB)", beam)
        ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_diag2_power_vs_depth.png"))


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag3 — ToF histograms per depth
# ─────────────────────────────────────────────────────────────────────────────

def plot_tof_histograms(
    df:       pd.DataFrame,
    save_dir: str,
    *,
    medium_name: Optional[str] = None,
    beam_name:   Optional[str] = None,
    max_depths:  int = 5,
) -> None:
    """
    Overlaid normalised ToF histograms for each link range.

    Uses a sequential colourmap so deeper ranges appear in progressively
    darker colours — the rightward spread and longer tail at depth is
    immediately visible.

    `medium_name` and `beam_name` filter to one (medium, beam) pair.
    """
    sub = df.copy()
    if medium_name:
        sub = sub[sub["medium_name"] == medium_name]
    if beam_name:
        sub = sub[sub["beam_name"] == beam_name]
    if sub.empty:
        return

    depths = sorted(sub["link_range_m"].unique())[:max_depths]
    cmap   = plt.colormaps["plasma"].resampled(len(depths))
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, Z in enumerate(depths):
        grp = sub[sub["link_range_m"] == Z]
        if len(grp) < 10:
            continue
        tof_hist = tof_histograms(grp, n_bins=200, group_by="link_range_m")
        if tof_hist.empty:
            continue
        ax.plot(tof_hist["tof_bin_ns"], tof_hist["density"],
                color=cmap(i), linewidth=1.4, label=f"{Z:.0f} m")

    title = "ToF Histograms by Depth"
    if medium_name:
        title += f"  |  {medium_name}"
    if beam_name:
        title += f"  |  {beam_name}"
    _style(ax, "Time of Flight (ns)", "Normalised Weight Density", title)
    ax.legend(fontsize=8, title="Depth")
    plt.tight_layout()
    fname = "fig_diag3_tof_histograms"
    if medium_name:
        fname += f"_{medium_name.replace(' ', '_')[:20]}"
    _save(fig, os.path.join(save_dir, fname + ".png"))


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag4 — Scattering profile vs depth
# ─────────────────────────────────────────────────────────────────────────────

def plot_scattering_profile(
    df:       pd.DataFrame,
    save_dir: str,
) -> None:
    """
    Mean and P95 scattering event count vs depth for each medium.

    The number of scattering events is the primary driver of:
      (a) photon loss (each scatter can deflect photon out of FOV)
      (b) delay spread (each scatter adds path length)
      (c) Woodcock null-collision overhead

    Plotting this confirms whether scattering differences between media
    explain the capture-rate differences seen in fig_diag1.
    """
    sp  = scattering_profile(df)
    if sp.empty:
        return

    mediums = sp["medium_name"].unique()
    colours = plt.colormaps["tab10"](np.linspace(0, 0.9, len(mediums)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scattering Profile vs Depth", fontsize=11, fontweight="bold")

    for ci, medium in enumerate(mediums):
        sub = sp[sp["medium_name"] == medium].sort_values("link_range_m")
        col = colours[ci]
        axes[0].plot(sub["link_range_m"], sub["mean_scatters"],
                     color=col, marker="o", linewidth=1.6, label=medium)
        axes[0].fill_between(sub["link_range_m"],
                             sub["mean_scatters"], sub["p95_scatters"],
                             alpha=0.12, color=col)
        axes[1].plot(sub["link_range_m"], sub["mean_excess_path"],
                     color=col, marker="s", linewidth=1.6, label=medium)
        axes[1].fill_between(sub["link_range_m"],
                             sub["mean_excess_path"], sub["p95_excess_path"],
                             alpha=0.12, color=col)

    _style(axes[0], "Link Range (m)", "Scattering Events",
           "Mean (± P95) Scattering Events vs Depth")
    _style(axes[1], "Link Range (m)", "Excess Path Length (m)",
           "Mean (± P95) Excess Path Length vs Depth")
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_diag4_scatter_profile.png"))


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag5 — Spatial spread (x, y) at each depth
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_spread(
    df:       pd.DataFrame,
    save_dir: str,
    *,
    medium_name: Optional[str] = None,
    beam_name:   Optional[str] = None,
    max_depths:  int = 4,
    max_points:  int = 3000,
) -> None:
    """
    Scatter plot of (x, y) capture positions at each link range.

    Shows whether the beam broadens with depth and whether captured photons
    cluster near the beam axis or spread to the aperture edge.
    The receiver aperture boundary is shown as a dashed circle.
    """
    from uowc.config import RECEIVER

    sub = df.copy()
    if medium_name:
        sub = sub[sub["medium_name"] == medium_name]
    if beam_name:
        sub = sub[sub["beam_name"] == beam_name]
    if sub.empty:
        return

    depths  = sorted(sub["link_range_m"].unique())[:max_depths]
    n_cols  = min(len(depths), 4)
    n_rows  = (len(depths) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    title = "Spatial Spread (x, y) at Receiver Plane"
    if medium_name:
        title += f"  |  {medium_name}"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    rx = RECEIVER.aperture_radius_m
    for idx, Z in enumerate(depths):
        row, col = divmod(idx, n_cols)
        ax  = axes[row][col]
        grp = sub[sub["link_range_m"] == Z]
        if len(grp) > max_points:
            grp = grp.sample(max_points, random_state=0)

        sc = ax.scatter(grp["x_m"], grp["y_m"],
                        c=grp["weight"], cmap="viridis",
                        s=3, alpha=0.5, vmin=0)
        theta   = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rx * np.cos(theta), rx * np.sin(theta),
                "r--", linewidth=1.0, label=f"Aperture r={rx:.3f} m")
        ax.set_xlim(-rx * 1.2, rx * 1.2)
        ax.set_ylim(-rx * 1.2, rx * 1.2)
        ax.set_aspect("equal")
        _style(ax, "x (m)", "y (m)", f"{Z:.0f} m  (N={len(grp):,})")
        ax.legend(fontsize=6)
        plt.colorbar(sc, ax=ax, label="weight", shrink=0.7)

    # Hide unused panels
    for idx in range(len(depths), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    fname = "fig_diag5_spatial_spread"
    if medium_name:
        fname += f"_{medium_name.replace(' ', '_')[:20]}"
    _save(fig, os.path.join(save_dir, fname + ".png"))


# ─────────────────────────────────────────────────────────────────────────────
# fig_diag6 — Excess path length distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_excess_path_distribution(
    df:       pd.DataFrame,
    save_dir: str,
    *,
    medium_name: Optional[str] = None,
) -> None:
    """
    Box-plot of excess path length (path_length_m - link_range_m) per depth.

    Excess path is the total extra distance a photon travelled due to
    scattering.  A long-tailed distribution at deep ranges explains the
    long delay-spread tail in the CIR.
    """
    sub = df.copy()
    if medium_name:
        sub = sub[sub["medium_name"] == medium_name]
    if sub.empty:
        return

    depths = sorted(sub["link_range_m"].unique())
    data   = [sub[sub["link_range_m"] == Z]["excess_path_m"].values
               for Z in depths]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, positions=depths, widths=[d * 0.15 for d in depths],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#a8d8ea", alpha=0.7))
    title = "Excess Path Length Distribution vs Depth"
    if medium_name:
        title += f"  |  {medium_name}"
    _style(ax, "Link Range (m)", "Excess Path Length (m)", title)
    ax.set_xticks(depths)
    ax.set_xticklabels([f"{Z:.0f}" for Z in depths])
    plt.tight_layout()
    fname = "fig_diag6_excess_path"
    if medium_name:
        fname += f"_{medium_name.replace(' ', '_')[:20]}"
    _save(fig, os.path.join(save_dir, fname + ".png"))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_diagnostics(
    df:          pd.DataFrame,
    stats_df:    pd.DataFrame,
    save_dir:    str,
    *,
    medium_name: Optional[str] = None,
    beam_name:   Optional[str] = None,
) -> None:
    """Render and save all six diagnostic figures."""
    os.makedirs(save_dir, exist_ok=True)
    plot_capture_rate(stats_df, save_dir)
    plot_power_with_ci(stats_df, save_dir)
    plot_tof_histograms(df, save_dir,
                        medium_name=medium_name, beam_name=beam_name)
    plot_scattering_profile(df, save_dir)
    plot_spatial_spread(df, save_dir,
                        medium_name=medium_name, beam_name=beam_name)
    plot_excess_path_distribution(df, save_dir, medium_name=medium_name)