"""
uowc.plotting
=============
Publication-quality figure generation.

Separation-of-Concern role
---------------------------
  This module receives pre-computed metric dicts and turns them into figures.
  It knows about axes, labels, colours, and line styles — nothing else.
  Changing a plot style never touches physics, transport, or metrics code.

  All public functions accept a `save_dir` argument and return nothing;
  figures are written to disk and closed to avoid memory accumulation.

Figure inventory
----------------
  Homogeneous sweep  (plot_all)
  ─────────────────────────────
    fig1_received_power.png      — received power vs range (2 panels: water type)
    fig2_cir.png                 — CIR at near/far range  (2×4 grid)
    fig3_frequency_response.png  — |H(f)| at near/far range (2×4 grid)
    fig4_delay_spread.png        — RMS delay spread vs range (2 panels: water type)
    fig5_bandwidth.png           — 3 dB bandwidth vs range  (2 panels: water type)

  Inhomogeneous sweep  (plot_all_inhomogeneous)   ← mirrors fig1–fig5 exactly
  ─────────────────────────────────────────────
    fig_inh1_received_power.png      — received power vs range (2 panels: beam type)
    fig_inh2_cir.png                 — CIR at near/far range  (n_media×4 grid)
    fig_inh3_frequency_response.png  — |H(f)| at near/far range (n_media×4 grid)
    fig_inh4_delay_spread.png        — RMS delay spread vs range (2 panels: beam type)
    fig_inh5_bandwidth.png           — 3 dB bandwidth vs range  (2 panels: beam type)

Design principles
-----------------
  Homogeneous panels are keyed by *water type*  (one panel per water, lines = beams).
  Inhomogeneous panels are keyed by *beam type*  (one panel per beam, lines = media).
  This is the only structural difference — axis labels, marker conventions, and
  grid layouts are identical between the two families so figures can be compared
  side-by-side without confusion.

  Shared rendering helpers (_render_cir_grid, _render_fr_grid, etc.) eliminate
  the copy-paste that previously existed between the two families.  Adding a new
  metric figure requires only: (a) one shared renderer, (b) two thin wrappers
  that pass the right entities and colour scheme.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Sequence

_log = logging.getLogger(__name__)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from uowc.config import (
    CLEAR_WATER, COASTAL_WATER, COLLIMATED, DIFFUSED,
    ALL_WATERS, ALL_BEAMS, SimConfig,
)
from uowc.simulation import RunKey


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

# Homogeneous: colour keyed by (water_name, beam_name)
_HOM_COLOUR: Dict[str, Dict[str, str]] = {
    CLEAR_WATER.name  : {COLLIMATED.name: "#1f77b4", DIFFUSED.name: "#7ab8e8"},
    COASTAL_WATER.name: {COLLIMATED.name: "#d62728", DIFFUSED.name: "#f4a261"},
}
_HOM_MARKER: Dict[str, str] = {
    COLLIMATED.name: "o",
    DIFFUSED.name:   "s",
}

# Inhomogeneous: colour/marker indexed by medium position in the list
_INH_COLOURS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
_INH_MARKERS = ["D", "^", "v", "p", "<", ">"]

_LINE_KW = dict(linewidth=1.8)
_BEER_KW = dict(color="black", linestyle="--", linewidth=1.2)
_GRID_KW = dict(alpha=0.25, linestyle="--")
_TICK_SZ  = 8
_LABEL_SZ = 9
_TITLE_SZ = 9

# Columns shown in CIR / frequency-response grids
def _z_show(cfg: SimConfig) -> List[float]:
    """Near and far range shown in the 2D grid figures."""
    r = list(cfg.link_ranges_m)
    return [r[0], r[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=_LABEL_SZ)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SZ)
    ax.set_title(title,   fontsize=_TITLE_SZ, fontweight="bold")
    ax.grid(True, **_GRID_KW)
    ax.tick_params(labelsize=_TICK_SZ)


def _save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info("Saved → %s", path)


def _semilogy_finite(ax, x, y, **kw) -> None:
    """semilogy that silently skips NaN/inf values (e.g. zero-capture runs)."""
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(ya) & (ya > 0)
    if mask.any():
        ax.semilogy(xa[mask], ya[mask], **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Empty-metric sentinel
# ─────────────────────────────────────────────────────────────────────────────
# A (medium, beam, range) combination that captured zero photons is absent
# from the Parquet file (to_dataframe skips empty groups), so the reconstructed
# metrics dict has no entry for it.  Plotting must degrade gracefully — show a
# blank panel / NaN gap — rather than KeyError.  All consumers use
# ``metrics.get(key, _EMPTY_METRIC)``.
_EMPTY_METRIC: Dict = {
    "power_dB":        float("nan"),
    "beer_lambert_dB": float("nan"),
    "delay_spread_s":  float("nan"),
    "bandwidth_hz":    float("nan"),
    "t_axis":          np.array([]),
    "cir":             np.array([]),
    "freqs":           np.array([]),
    "fr":              np.array([]),
    "n_launched":      0,
    "n_captured":      0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Photon-count quality helpers
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds for CIR/frequency-response reliability
_CIR_THRESHOLDS = [
    (2_000, "good",      None),
    (250,   "sparse",    "orange"),
    (50,    "very sparse","red"),
    (0,     "too few",   "red"),
]


def _cir_quality(n_captured: int):
    """Return (label, colour) for a photon count."""
    for threshold, label, colour in _CIR_THRESHOLDS:
        if n_captured >= threshold:
            return label, colour
    return "too few", "red"


def _annotate_photon_count(ax, n_captured: int) -> None:
    """Add a photon-count badge in the top-right corner of a CIR/FR subplot."""
    label, colour = _cir_quality(n_captured)
    text  = f"n={n_captured:,}\n({label})"
    props = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor=colour or "grey", alpha=0.85)
    ax.text(
        0.97, 0.97, text,
        transform=ax.transAxes,
        fontsize=6.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        color=colour or "black",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared grid renderers  (used by both homogeneous and inhomogeneous families)
# ─────────────────────────────────────────────────────────────────────────────

def _render_cir_grid(
    axes:      np.ndarray,          # shape (n_rows, n_beams * 2)
    metrics:   Dict[RunKey, dict],
    entity_names: List[str],        # water names OR medium names (row labels)
    colours:   List[str],           # one colour per row
    cfg:       SimConfig,
) -> None:
    """
    Fill a (n_rows × n_beams×2) axes grid with normalised CIR plots.

    Columns cycle: beam₀@near, beam₀@far, beam₁@near, beam₁@far, …
    Each row corresponds to one entity (water type or medium).
    """
    z_near, z_far = _z_show(cfg)
    for row, (entity_name, colour) in enumerate(zip(entity_names, colours)):
        col = 0
        for beam in ALL_BEAMS:
            for Z in (z_near, z_far):
                key  = RunKey(entity_name, beam.name, float(Z))
                m    = metrics.get(key, _EMPTY_METRIC)
                ax   = axes[row, col]
                _style_ax(ax, "Time (ns)", "Norm. amplitude",
                          f"{entity_name}\n{beam.name} — {Z} m")
                t_ns = np.asarray(m["t_axis"]) * 1e9
                if t_ns.size:
                    ax.plot(t_ns, m["cir"], color=colour, linewidth=1.2)
                    ax.set_xlim([0, t_ns[-1]])
                _annotate_photon_count(ax, m.get("n_captured", 0))
                col += 1


def _render_fr_grid(
    axes:         np.ndarray,
    metrics:      Dict[RunKey, dict],
    entity_names: List[str],
    colours:      List[str],
    cfg:          SimConfig,
) -> None:
    """
    Fill a (n_rows × n_beams×2) axes grid with |H(f)| frequency-response plots.

    Each subplot shows:
      • |H(f)| curve on a linear scale
      • dashed grey line at 1/√2  (−3 dB threshold)
      • dotted red  line at the 3 dB bandwidth
    """
    z_near, z_far = _z_show(cfg)
    for row, (entity_name, colour) in enumerate(zip(entity_names, colours)):
        col = 0
        for beam in ALL_BEAMS:
            for Z in (z_near, z_far):
                key    = RunKey(entity_name, beam.name, float(Z))
                m      = metrics.get(key, _EMPTY_METRIC)
                ax     = axes[row, col]
                _style_ax(ax, "Frequency (MHz)", "|H(f)|",
                          f"{entity_name}\n{beam.name} — {Z} m")

                f_MHz  = np.asarray(m["freqs"]) / 1e6
                if f_MHz.size:
                    bw_MHz = m["bandwidth_hz"] / 1e6
                    ax.plot(f_MHz, m["fr"], color=colour, linewidth=1.2)
                    ax.axhline(1.0 / np.sqrt(2.0), color="grey",
                               linestyle="--", linewidth=0.8, label="−3 dB")
                    ax.axvline(bw_MHz, color="red", linestyle=":",
                               linewidth=1.0, label=f"BW={bw_MHz:.1f} MHz")
                    # Clip x-axis to 5× the bandwidth so low-BW channels don't
                    # compress all interesting content into the left edge.
                    x_max = min(f_MHz[-1], max(bw_MHz * 5, 1.0))
                    ax.set_xlim([0, x_max])
                    ax.set_ylim([0, 1.1])
                    ax.legend(fontsize=7)
                _annotate_photon_count(ax, m.get("n_captured", 0))
                col += 1


# ─────────────────────────────────────────────────────────────────────────────
# ── HOMOGENEOUS FIGURES  (fig1 – fig5) ──────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_received_power(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    fig1 — Received power vs link range.

    Two panels (one per water type).  Each panel shows MC power for both beam
    types and a Beer-Lambert reference.  Lines are coloured by (water, beam).
    """
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "Normalised Received Power vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )
    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            pwr = [metrics.get(RunKey(water.name, beam.name, float(Z)), _EMPTY_METRIC)["power_dB"]
                   for Z in ranges]
            ax.plot(ranges, pwr,
                    marker=_HOM_MARKER[beam.name],
                    color=_HOM_COLOUR[water.name][beam.name],
                    label=f"MC — {beam.name}", **_LINE_KW)
        r_arr = np.array(ranges)
        bl    = 10.0 * np.log10(np.exp(-water.c * r_arr) + 1e-300)
        ax.plot(r_arr, bl, **_BEER_KW, label="Beer-Lambert")
        _style_ax(ax, "Link Range (m)", "Normalised Power (dB)", water.name)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig1_received_power.png"))


def plot_cir(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    fig2 — Channel impulse response at near and far range.

    Grid layout (rows = water type, cols = beam × {near, far}):
      Clear   | Coll@near | Coll@far | Diff@near | Diff@far
      Coastal | Coll@near | Coll@far | Diff@near | Diff@far
    """
    z_near, z_far = _z_show(cfg)
    n_rows  = len(ALL_WATERS)
    n_cols  = len(ALL_BEAMS) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Channel Impulse Response  |  Homogeneous Medium\n"
        f"Cols: {ALL_BEAMS[0].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[0].name} @{z_far:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_far:.0f} m",
        fontsize=11, fontweight="bold",
    )
    entity_names = [w.name for w in ALL_WATERS]
    colours      = [_HOM_COLOUR[w.name][ALL_BEAMS[0].name] for w in ALL_WATERS]
    _render_cir_grid(axes, metrics, entity_names, colours, cfg)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig2_cir.png"))


def plot_frequency_response(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    fig3 — Channel frequency response |H(f)| at near and far range.

    Same grid layout as fig2.  Each subplot shows the normalised magnitude
    response, the −3 dB threshold, and the 3 dB bandwidth marker.
    """
    z_near, z_far = _z_show(cfg)
    n_rows  = len(ALL_WATERS)
    n_cols  = len(ALL_BEAMS) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Channel Frequency Response |H(f)|  |  Homogeneous Medium\n"
        f"Cols: {ALL_BEAMS[0].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[0].name} @{z_far:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_far:.0f} m",
        fontsize=11, fontweight="bold",
    )
    entity_names = [w.name for w in ALL_WATERS]
    colours      = [_HOM_COLOUR[w.name][ALL_BEAMS[0].name] for w in ALL_WATERS]
    _render_fr_grid(axes, metrics, entity_names, colours, cfg)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig3_frequency_response.png"))


def plot_delay_spread(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    fig4 — RMS delay spread vs link range.

    Two panels (one per water type).  Semi-log Y axis.
    """
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "RMS Delay Spread vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )
    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            ds = [metrics.get(RunKey(water.name, beam.name, float(Z)), _EMPTY_METRIC)["delay_spread_s"]
                  for Z in ranges]
            _semilogy_finite(ax, ranges, ds,
                             marker=_HOM_MARKER[beam.name],
                             color=_HOM_COLOUR[water.name][beam.name],
                             label=beam.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "RMS Delay Spread (s)", water.name)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.2e}")
        )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig4_delay_spread.png"))


def plot_bandwidth(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    fig5 — 3 dB channel bandwidth vs link range.

    Two panels (one per water type).  Semi-log Y axis.
    """
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "3 dB Channel Bandwidth vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )
    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            bw = [metrics.get(RunKey(water.name, beam.name, float(Z)), _EMPTY_METRIC)["bandwidth_hz"] / 1e6
                  for Z in ranges]
            _semilogy_finite(ax, ranges, bw,
                             marker=_HOM_MARKER[beam.name],
                             color=_HOM_COLOUR[water.name][beam.name],
                             label=beam.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "3 dB Bandwidth (MHz)", water.name)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.1f}")
        )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig5_bandwidth.png"))


# ─────────────────────────────────────────────────────────────────────────────
# ── INHOMOGENEOUS FIGURES  (fig_inh1 – fig_inh5) ────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#
# Structural mirror of the homogeneous family:
#   Homogeneous panels → one panel per water type  (2 water types)
#   Inhomogeneous panels → one panel per beam type (2 beam types)
#
# Within each panel:
#   Homogeneous lines → one line per beam
#   Inhomogeneous lines → one line per medium  (colour = _INH_COLOURS[i])
#
# ─────────────────────────────────────────────────────────────────────────────

def plot_inhomogeneous_power(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """
    fig_inh1 — Received power vs link range across inhomogeneous media.

    Two panels (one per beam type).  Lines = media profiles.
    Mirrors fig1 with beam↔water role swap.
    """
    ranges = list(cfg.link_ranges_m)
    n_beams = len(ALL_BEAMS)
    fig, axes = plt.subplots(1, n_beams, figsize=(7 * n_beams, 5))
    if n_beams == 1:
        axes = [axes]
    fig.suptitle(
        "Normalised Received Power vs Link Range  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for ax, beam in zip(axes, ALL_BEAMS):
        for i, medium in enumerate(media):
            colour = _INH_COLOURS[i % len(_INH_COLOURS)]
            marker = _INH_MARKERS[i % len(_INH_MARKERS)]
            pwr    = [metrics.get(RunKey(medium.name, beam.name, float(Z)), _EMPTY_METRIC)["power_dB"]
                      for Z in ranges]
            ax.plot(ranges, pwr, marker=marker, color=colour,
                    label=medium.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "Normalised Power (dB)", beam.name)
        ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh1_received_power.png"))


def plot_inhomogeneous_cir(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """
    fig_inh2 — CIR at near and far range across inhomogeneous media.

    Grid layout (rows = medium, cols = beam × {near, far}):
      Medium 0 | Coll@near | Coll@far | Diff@near | Diff@far
      Medium 1 | …
      …

    Mirrors fig2 with medium replacing water type in the row axis.
    """
    z_near, z_far = _z_show(cfg)
    n_rows  = len(media)
    n_cols  = len(ALL_BEAMS) * 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Channel Impulse Response  |  Inhomogeneous Media\n"
        f"Cols: {ALL_BEAMS[0].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[0].name} @{z_far:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_far:.0f} m",
        fontsize=11, fontweight="bold",
    )
    entity_names = [m.name for m in media]
    colours      = [_INH_COLOURS[i % len(_INH_COLOURS)] for i in range(len(media))]
    _render_cir_grid(axes, metrics, entity_names, colours, cfg)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh2_cir.png"))


def plot_inhomogeneous_frequency_response(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """
    fig_inh3 — Channel frequency response |H(f)| at near and far range.

    Grid layout mirrors fig_inh2 (rows = medium, cols = beam × {near, far}).
    Each subplot shows |H(f)|, the −3 dB threshold, and the 3 dB bandwidth.

    Mirrors fig3 exactly — only the entity axis (water → medium) changes.
    """
    z_near, z_far = _z_show(cfg)
    n_rows  = len(media)
    n_cols  = len(ALL_BEAMS) * 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Channel Frequency Response |H(f)|  |  Inhomogeneous Media\n"
        f"Cols: {ALL_BEAMS[0].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[0].name} @{z_far:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_near:.0f} m | "
        f"{ALL_BEAMS[1].name} @{z_far:.0f} m",
        fontsize=11, fontweight="bold",
    )
    entity_names = [m.name for m in media]
    colours      = [_INH_COLOURS[i % len(_INH_COLOURS)] for i in range(len(media))]
    _render_fr_grid(axes, metrics, entity_names, colours, cfg)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh3_frequency_response.png"))


def plot_inhomogeneous_delay_spread(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """
    fig_inh4 — RMS delay spread vs link range across inhomogeneous media.

    Two panels (one per beam type).  Semi-log Y axis.  Lines = media profiles.
    NaN values from zero-capture runs are silently skipped.

    Mirrors fig4 with beam↔water role swap.
    """
    ranges = list(cfg.link_ranges_m)
    n_beams = len(ALL_BEAMS)
    fig, axes = plt.subplots(1, n_beams, figsize=(7 * n_beams, 5))
    if n_beams == 1:
        axes = [axes]
    fig.suptitle(
        "RMS Delay Spread vs Link Range  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for ax, beam in zip(axes, ALL_BEAMS):
        for i, medium in enumerate(media):
            colour = _INH_COLOURS[i % len(_INH_COLOURS)]
            marker = _INH_MARKERS[i % len(_INH_MARKERS)]
            ds     = [metrics.get(RunKey(medium.name, beam.name, float(Z)), _EMPTY_METRIC)["delay_spread_s"]
                      for Z in ranges]
            _semilogy_finite(ax, ranges, ds, marker=marker, color=colour,
                             label=medium.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "RMS Delay Spread (s)", beam.name)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.2e}")
        )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh4_delay_spread.png"))


def plot_inhomogeneous_bandwidth(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """
    fig_inh5 — 3 dB channel bandwidth vs link range across inhomogeneous media.

    Two panels (one per beam type).  Semi-log Y axis.  Lines = media profiles.

    Mirrors fig5 with beam↔water role swap.
    """
    ranges = list(cfg.link_ranges_m)
    n_beams = len(ALL_BEAMS)
    fig, axes = plt.subplots(1, n_beams, figsize=(7 * n_beams, 5))
    if n_beams == 1:
        axes = [axes]
    fig.suptitle(
        "3 dB Channel Bandwidth vs Link Range  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for ax, beam in zip(axes, ALL_BEAMS):
        for i, medium in enumerate(media):
            colour = _INH_COLOURS[i % len(_INH_COLOURS)]
            marker = _INH_MARKERS[i % len(_INH_MARKERS)]
            bw     = [metrics.get(RunKey(medium.name, beam.name, float(Z)), _EMPTY_METRIC)["bandwidth_hz"] / 1e6
                      for Z in ranges]
            _semilogy_finite(ax, ranges, bw, marker=marker, color=colour,
                             label=medium.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "3 dB Bandwidth (MHz)", beam.name)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.1f}")
        )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh5_bandwidth.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """Render and save all five homogeneous figures (fig1 – fig5)."""
    os.makedirs(save_dir, exist_ok=True)
    plot_received_power(metrics, cfg, save_dir)
    plot_cir(metrics, cfg, save_dir)
    plot_frequency_response(metrics, cfg, save_dir)
    plot_delay_spread(metrics, cfg, save_dir)
    plot_bandwidth(metrics, cfg, save_dir)


def plot_all_inhomogeneous(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Sequence,
    save_dir: str,
) -> None:
    """Render and save all five inhomogeneous figures (fig_inh1 – fig_inh5)."""
    os.makedirs(save_dir, exist_ok=True)
    plot_inhomogeneous_power(metrics, cfg, media, save_dir)
    plot_inhomogeneous_cir(metrics, cfg, media, save_dir)
    plot_inhomogeneous_frequency_response(metrics, cfg, media, save_dir)
    plot_inhomogeneous_delay_spread(metrics, cfg, media, save_dir)
    plot_inhomogeneous_bandwidth(metrics, cfg, media, save_dir)