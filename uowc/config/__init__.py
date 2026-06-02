"""
uowc.config
===========
Single source of truth for every configurable value in the simulation.
Nothing in this module performs computation — it only declares data.

Separation-of-Concern role
---------------------------
  All magic numbers, physical constants, and simulation knobs live here.
  Every other module imports from here; nothing hard-codes values locally.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT:  float = 3.0e8
N_WATER:  float = 1.33
C_MEDIUM: float = C_LIGHT / N_WATER


# ─────────────────────────────────────────────────────────────────────────────
# Parameter dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WaterParams:
    """Inherent optical properties for one water type at the simulation wavelength."""
    name: str
    c:    float   # beam-attenuation coefficient  (m⁻¹)  c = a + b
    a:    float   # absorption coefficient         (m⁻¹)
    b:    float   # scattering coefficient         (m⁻¹)
    g:    float   # Henyey-Greenstein asymmetry parameter

    @property
    def omega(self) -> float:
        """Single-scattering albedo  ω = b / c."""
        return self.b / self.c


@dataclass(frozen=True)
class BeamParams:
    """Transmitter beam geometry."""
    name:           str
    divergence_rad: float
    waist_m:        float


@dataclass(frozen=True)
class ReceiverParams:
    """Optical receiver geometry."""
    aperture_radius_m: float
    fov_rad:           float


@dataclass(frozen=True)
class SimConfig:
    """
    All tunable knobs for one simulation run.

    Fixed-launch fields
    -------------------
    n_photons : int
        Number of photons launched per batch.  In fixed mode (min_captured=0)
        this is the total per run.  In adaptive mode this is the batch size —
        the simulation keeps launching batches until min_captured_photons is
        reached or max_launched_photons is exceeded.

    Adaptive capture fields
    -----------------------
    min_captured_photons : int
        Target minimum number of captured (received) photons per run.
        Set to 0 to disable adaptive mode (use exactly n_photons per run).
        Recommended: ≥ 10 000 for smooth CIR histograms and reliable
        delay-spread / bandwidth estimates.

    max_launched_photons : int
        Hard safety cap on total photons launched per run.  Prevents
        infinite loops when the capture rate is near zero (very long range
        in turbid water).  A warning is printed if the cap is hit before
        min_captured_photons is reached.

    Statistical guidance
    --------------------
    For n_time_bins = 3000 with ~100 occupied bins, 10 000 captured photons
    gives ~100 photons/peak-bin — adequate for delay spread and bandwidth.
    For publication-quality CIR shapes, 50 000+ is recommended.
    """
    # ── Required fields (no defaults) ────────────────────────────────────
    n_photons:        int
    link_ranges_m:    Tuple[float, ...]
    dt_bin_s:         float
    n_time_bins:      int
    weight_threshold: float
    roulette_m:       int
    n_workers:        int
    master_seed:      int
    chunk_size:       int
    # ── Optional adaptive fields (defaults allow old call sites to work) ─
    min_captured_photons: int = 10_000
    max_launched_photons: int = 100_000_000
    conv_batch_photons: int = 1_000_000
    rel_error_tol: float = 0.05
    min_conv_batches: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Preset water types  (530 nm, Gabriel et al. 2013 / Petzold 1972)
# ─────────────────────────────────────────────────────────────────────────────
CLEAR_WATER = WaterParams(
    name="Clear Water",
    c=0.241, a=0.151, b=0.090, g=0.924,
)
COASTAL_WATER = WaterParams(
    name="Coastal Water",
    c=0.775, a=0.220, b=0.555, g=0.924,
)
TURBID_WATER = WaterParams(
    name="Turbid Water",
    c=2.19, a=0.366, b=1.824, g=0.945,
)

# ─────────────────────────────────────────────────────────────────────────────
# Preset beam and receiver configurations
# ─────────────────────────────────────────────────────────────────────────────
COLLIMATED = BeamParams(
    name="Collimated (Laser)",
    divergence_rad=1.5e-3,
    waist_m=1e-3,
)
DIFFUSED = BeamParams(
    name="Diffused (LED)",
    divergence_rad=np.deg2rad(15.0),
    waist_m=1e-3,
)
RECEIVER = ReceiverParams(
    aperture_radius_m=0.1016 / 2.0,
    fov_rad=np.pi,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default simulation configuration
# ─────────────────────────────────────────────────────────────────────────────
SIM = SimConfig(
    n_photons            = 1_000_000,   # batch size per adaptive round
    link_ranges_m        = (5, 10, 15, 20, 25),
    dt_bin_s             = 1e-11,       # 10 ps bins
    n_time_bins          = 3000,
    weight_threshold     = 1e-4,
    roulette_m           = 10,
    n_workers            = max(1, os.cpu_count() or 2),
    master_seed          = 20260519,
    chunk_size           = 10_000,
    min_captured_photons = 10_000,      # stop when ≥ 10 k photons captured
    max_launched_photons = 100_000_000, # hard cap: 100 M photons
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience collections
# ─────────────────────────────────────────────────────────────────────────────
ALL_WATERS: Tuple[WaterParams, ...] = (CLEAR_WATER, COASTAL_WATER)
ALL_BEAMS:  Tuple[BeamParams,  ...] = (COLLIMATED, DIFFUSED)