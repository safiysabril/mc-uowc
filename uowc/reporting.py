"""
uowc.reporting
==============
Human-readable console output — summary tables and run headers.

Separation-of-Concern role
---------------------------
  This module owns all formatted text output.  It consumes the same
  Dict[RunKey, dict] of computed metrics as the plotting module, but
  renders to stdout rather than to files.  Separating it from `plotting`
  means either can be swapped, suppressed, or redirected independently.

  Nothing here computes a metric or touches matplotlib.
"""

from __future__ import annotations
from typing import Dict, Sequence

from uowc.config import (
    WaterParams, BeamParams,
    ALL_WATERS, ALL_BEAMS, SimConfig,
)
from uowc.simulation import RunKey


# ─────────────────────────────────────────────────────────────────────────────
# Run headers
# ─────────────────────────────────────────────────────────────────────────────

def print_run_header(cfg: SimConfig) -> None:
    """Print a startup banner for a homogeneous medium sweep."""
    sep = "=" * 70
    print(sep)
    print("  UOWC Monte Carlo Simulation  —  Homogeneous Medium")
    print(sep)
    print(f"  Photons per run : {cfg.n_photons:,}")
    print(f"  CPU workers     : {cfg.n_workers}")
    print(f"  Link ranges     : {list(cfg.link_ranges_m)} m")
    print(f"  Waters          : "
          + "  |  ".join(f"{w.name} (c={w.c} m⁻¹)" for w in ALL_WATERS))
    print(f"  Beams           : "
          + "  |  ".join(f"{b.name}" for b in ALL_BEAMS))
    print(f"  Wavelength      : 530 nm")
    print(f"  Master RNG seed : {cfg.master_seed}")
    print(sep + "\n")


def print_inhomogeneous_header(cfg: SimConfig, media: Sequence) -> None:
    """Print a startup banner for an inhomogeneous medium sweep."""
    sep = "=" * 70
    print(sep)
    print("  UOWC Monte Carlo Simulation  —  Inhomogeneous Medium")
    print(sep)
    print(f"  Photons per run : {cfg.n_photons:,}")
    print(f"  CPU workers     : {cfg.n_workers}")
    print(f"  Link ranges     : {list(cfg.link_ranges_m)} m")
    print(f"  Beams           : "
          + "  |  ".join(f"{b.name}" for b in ALL_BEAMS))
    print(f"  Wavelength      : 530 nm")
    print(f"  Master RNG seed : {cfg.master_seed + 1}")
    print(f"\n  Media profiles:")
    for medium in media:
        for line in medium.summary().splitlines():
            print(f"    {line}")
        print()
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables  (homogeneous)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_tables(metrics: Dict[RunKey, dict], cfg: SimConfig) -> None:
    """Print all three metric summary tables to stdout."""
    _print_delay_spread_table(metrics, cfg)
    _print_power_table(metrics, cfg)
    _print_bandwidth_table(metrics, cfg)


def _table_header(label: str) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)


def _water_section_header(water: WaterParams, extra: str = "") -> None:
    print(f"\n  {water.name}  "
          f"(c={water.c} m⁻¹  ω={water.omega:.3f}  g={water.g})"
          + (f"  {extra}" if extra else ""))
    col_head = f"  {'Range (m)':<12}" + "".join(
        f"  {b.name:<28}" for b in ALL_BEAMS
    )
    print(col_head)
    print("  " + "-" * (len(col_head) - 2))


def _print_delay_spread_table(
    metrics: Dict[RunKey, dict], cfg: SimConfig
) -> None:
    _table_header("DELAY SPREAD  (seconds)")
    for water in ALL_WATERS:
        _water_section_header(water)
        for Z in cfg.link_ranges_m:
            row = f"  {Z:<12}"
            for beam in ALL_BEAMS:
                key = RunKey(water.name, beam.name, float(Z))
                ds  = metrics[key]["delay_spread_s"]
                row += f"  {ds:<28.4e}"
            print(row)


def _print_power_table(
    metrics: Dict[RunKey, dict], cfg: SimConfig
) -> None:
    _table_header("RECEIVED POWER  (dB, normalised to launched power)")
    for water in ALL_WATERS:
        _water_section_header(water)
        for Z in cfg.link_ranges_m:
            row = f"  {Z:<12}"
            for beam in ALL_BEAMS:
                key = RunKey(water.name, beam.name, float(Z))
                p   = metrics[key]["power_dB"]
                bl  = metrics[key]["beer_lambert_dB"]
                row += f"  MC={p:+7.2f}  BL={bl:+7.2f}{'':8}"
            print(row)


def _print_bandwidth_table(
    metrics: Dict[RunKey, dict], cfg: SimConfig
) -> None:
    _table_header("3 dB CHANNEL BANDWIDTH  (MHz)")
    for water in ALL_WATERS:
        _water_section_header(water)
        for Z in cfg.link_ranges_m:
            row = f"  {Z:<12}"
            for beam in ALL_BEAMS:
                key = RunKey(water.name, beam.name, float(Z))
                bw  = metrics[key]["bandwidth_hz"] / 1e6
                row += f"  {bw:<28.2f}"
            print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables  (inhomogeneous)
# ─────────────────────────────────────────────────────────────────────────────

def print_inhomogeneous_summary(
    metrics: Dict[RunKey, dict],
    cfg:     SimConfig,
    media:   Sequence,
) -> None:
    """Print power and delay spread tables for all inhomogeneous media."""
    _table_header("INHOMOGENEOUS — RECEIVED POWER  (dB)")
    for medium in media:
        print(f"\n  Medium: {medium.name}  (c_max={medium.c_max:.3f} m⁻¹)")
        col_head = f"  {'Range (m)':<12}" + "".join(
            f"  {b.name:<28}" for b in ALL_BEAMS
        )
        print(col_head)
        print("  " + "-" * (len(col_head) - 2))
        for Z in cfg.link_ranges_m:
            row = f"  {Z:<12}"
            for beam in ALL_BEAMS:
                key = RunKey(medium.name, beam.name, float(Z))
                p   = metrics[key]["power_dB"]
                row += f"  MC={p:+7.2f}{'':18}"
            print(row)

    _table_header("INHOMOGENEOUS — DELAY SPREAD  (seconds)")
    for medium in media:
        print(f"\n  Medium: {medium.name}")
        col_head = f"  {'Range (m)':<12}" + "".join(
            f"  {b.name:<28}" for b in ALL_BEAMS
        )
        print(col_head)
        print("  " + "-" * (len(col_head) - 2))
        for Z in cfg.link_ranges_m:
            row = f"  {Z:<12}"
            for beam in ALL_BEAMS:
                key = RunKey(medium.name, beam.name, float(Z))
                ds  = metrics[key]["delay_spread_s"]
                row += f"  {ds:<28.4e}"
            print(row)
    print()