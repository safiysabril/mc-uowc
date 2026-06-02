"""
uowc.medium
===========
Spatial optical medium descriptions for homogeneous and inhomogeneous
underwater channels.

Separation-of-Concern role
---------------------------
  This module owns one responsibility: answering "what are the IOPs at
  depth z?"  It has no knowledge of photon transport mechanics, channel
  metrics, parallelism, or plotting.

  Every concrete class satisfies the `MediumProfile` interface so the
  transport kernel can treat them identically.  Adding a new medium type
  (e.g. bio-optical model, empirical CTD profile) only requires adding a
  new class here — the rest of the pipeline is untouched.

Design
------
  The protocol uses a vectorised API: all `attenuation(z)`, `scattering(z)`,
  etc. methods accept a NumPy array of depths and return arrays of the same
  shape.  This eliminates Python loops in the hot path.

  All classes are `frozen=True` dataclasses so they are:
    - hashable  (usable as dict keys, e.g. in simulation result caches)
    - immutable (safe to share between worker processes without copying)
    - picklable (survive the ProcessPoolExecutor pickle/unpickle round-trip)

Woodcock delta-tracking context
---------------------------------
  For inhomogeneous media the transport engine uses the Woodcock delta-
  tracking algorithm (Woodcock et al. 1965, Lux & Koblinger 1991).
  The algorithm requires a global majorant `c_max ≥ c(z)` for all z.
  Every `MediumProfile` exposes `c_max` for this purpose.

  The HomogeneousMedium special-cases as is_homogeneous() == True, allowing
  the transport kernel to take the original fast path (no acceptance test).

References
----------
  Woodcock et al. (1965) — "Techniques used in the GEM code" (delta tracking)
  Lux & Koblinger (1991) — Monte Carlo Particle Transport Methods, CRC Press
  Mobley (1994) — Light and Water: Radiative Transfer in Natural Waters
  Petzold (1972) — Vol. Scat. Functions for Selected Ocean Waters, SIO ref 72-78
  Haltrin (1999) — Applied Optics 38(33):6826  (chlorophyll-based IOP model)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray

from uowc.config import WaterParams, CLEAR_WATER, COASTAL_WATER, TURBID_WATER


# ─────────────────────────────────────────────────────────────────────────────
# MediumProfile Protocol  (structural subtyping — no forced inheritance)
# ─────────────────────────────────────────────────────────────────────────────

class MediumProfile:
    """
    Abstract interface for a spatially varying optical medium.

    All concrete medium classes implement this interface.  The transport
    kernel depends only on this interface, never on a specific subclass.

    Vectorised contract
    -------------------
    All IOP methods accept a depth array `z` of shape (N,) and return
    a NumPy array of the same shape.  This supports batch photon operations
    without Python-level loops.

    Units
    -----
    z       : metres (positive = downward from transmitter)
    c, b    : m⁻¹
    g       : dimensionless (0 = isotropic, 1 = fully forward)
    omega   : dimensionless (0 ≤ ω ≤ 1)
    c_max   : m⁻¹  (global supremum, used as Woodcock majorant)
    """

    @property
    def c_max(self) -> float:
        """
        Global upper bound on the beam-attenuation coefficient (m⁻¹).

        For the Woodcock delta-tracking algorithm this acts as the majorant:
            c_max ≥ c(z)  for all z in the simulation domain.

        Raising c_max beyond the true maximum remains statistically correct
        (it only adds null collisions and increases computation), but setting
        it too low biases the result — so implementations must be conservative.
        """
        raise NotImplementedError

    def attenuation(self, z: ndarray) -> ndarray:
        """Beam-attenuation coefficient c(z) [m⁻¹], shape (N,)."""
        raise NotImplementedError

    def scattering(self, z: ndarray) -> ndarray:
        """Scattering coefficient b(z) [m⁻¹], shape (N,)."""
        raise NotImplementedError

    def asymmetry(self, z: ndarray) -> ndarray:
        """HG asymmetry parameter g(z) [dimensionless], shape (N,)."""
        raise NotImplementedError

    def albedo(self, z: ndarray) -> ndarray:
        """
        Single-scattering albedo ω(z) = b(z) / c(z), shape (N,).

        Used as the weight-reduction factor at each real scattering event
        in the implicit-absorption MCML scheme.
        """
        raise NotImplementedError

    def is_homogeneous(self) -> bool:
        """
        True iff IOPs are depth-independent throughout the simulation domain.

        When True, the transport kernel skips the Woodcock acceptance test
        and uses the original faster homogeneous code path.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Short descriptive label used as the RunKey identifier and in figure titles.

        Every concrete subclass must implement this.
        HomogeneousMedium exposes it via a @property delegating to params.name.
        LayeredMedium and GradientMedium declare it as a dataclass field.
        """
        raise NotImplementedError

    def summary(self) -> str:
        """One-line human-readable description for console output."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# HomogeneousMedium  — backward-compatible constant-IOP wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HomogeneousMedium(MediumProfile):
    """
    Depth-uniform medium — wraps a single WaterParams instance.

    This is the original model exactly reproduced.  When the transport
    kernel detects `is_homogeneous() == True` it skips the Woodcock
    acceptance test and uses scalar-c step sampling, giving the same
    performance as the pre-inhomogeneous codebase.

    Parameters
    ----------
    params : WaterParams for the uniform channel
    """
    params: WaterParams

    @property
    def name(self) -> str:
        """Expose water type name for use as RunKey identifier."""
        return self.params.name

    @property
    def c_max(self) -> float:
        return self.params.c

    def attenuation(self, z: ndarray) -> ndarray:
        return np.full(z.shape, self.params.c, dtype=np.float64)

    def scattering(self, z: ndarray) -> ndarray:
        return np.full(z.shape, self.params.b, dtype=np.float64)

    def asymmetry(self, z: ndarray) -> ndarray:
        return np.full(z.shape, self.params.g, dtype=np.float64)

    def albedo(self, z: ndarray) -> ndarray:
        return np.full(z.shape, self.params.omega, dtype=np.float64)

    def is_homogeneous(self) -> bool:
        return True

    def summary(self) -> str:
        p = self.params
        return (f"HomogeneousMedium({p.name}  "
                f"c={p.c} m⁻¹  ω={p.omega:.3f}  g={p.g})")


# ─────────────────────────────────────────────────────────────────────────────
# LayeredMedium  — piecewise-constant depth stratification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LayeredMedium(MediumProfile):
    """
    Depth-stratified medium with piecewise-constant IOPs.

    The water column is partitioned into N horizontal slabs.  Within each
    slab the IOPs are uniform (the medium is homogeneous in that layer but
    discontinuous at the boundaries).

    Parameters
    ----------
    layers : sequence of (z_bottom_m, WaterParams) pairs sorted ascending.
             The last layer implicitly extends to +∞.
             Example — two-layer ocean column (25 m link):
               layers = [
                   (10.0, CLEAR_WATER),    # surface slab:   0–10 m
                   (np.inf, COASTAL_WATER),# deep slab:      10+ m
               ]
    name   : descriptive label for console output / figure titles.

    Physical motivation
    -------------------
    Real ocean columns are optically stratified (Mobley 1994):

      0–10 m   Surface mixed layer — high solar irradiance, lower
               phytoplankton density, weaker scattering → CLEAR_WATER.

      10–50 m  Deep Chlorophyll Maximum (DCM) — phytoplankton bloom at the
               nutricline, much higher chlorophyll and scattering → COASTAL.

      50+ m    Aphotic zone — high CDOM, particulate detritus, reducing
               visibility dramatically → TURBID_WATER.

    IOP lookup
    ----------
    Uses `np.searchsorted` on the boundary array: O(N log K) where K is the
    number of layers.  For typical K ≤ 10 this is negligible.

    Woodcock majorant
    -----------------
    `c_max` is the maximum `c` over all layers, which is always a valid
    upper bound.  Null-collision rate ≈ 1 - c_mean/c_max; for a 3-layer
    clear→coastal→turbid profile this is typically 50–70 %, which is
    acceptable given that null steps are cheap (no RNG for scattering).
    """
    layers: Tuple[Tuple[float, WaterParams], ...]
    name:   str = "Layered Medium"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("LayeredMedium requires at least one layer.")
        # Validate that boundaries are strictly increasing
        bounds = [zb for zb, _ in self.layers]
        for i in range(len(bounds) - 1):
            if bounds[i] >= bounds[i + 1]:
                raise ValueError(
                    f"Layer boundaries must be strictly increasing; "
                    f"got {bounds[i]} ≥ {bounds[i + 1]}."
                )

    @property
    def c_max(self) -> float:
        """Maximum c over all layers — used as Woodcock majorant."""
        return float(max(wp.c for _, wp in self.layers))

    # ------------------------------------------------------------------
    # Internal: fast vectorised lookup using searchsorted
    # ------------------------------------------------------------------
    def _layer_index(self, z: ndarray) -> ndarray:
        """
        Return the 0-based layer index for each depth in `z`.

        Layer i spans  [boundaries[i-1], boundaries[i])  with boundary[−1] = 0.
        np.searchsorted(boundaries, z, side='right') returns the index of the
        first boundary strictly greater than z, which equals the layer index.
        """
        boundaries = np.asarray([zb for zb, _ in self.layers], dtype=np.float64)
        idx = np.searchsorted(boundaries, z, side='right')
        return np.clip(idx, 0, len(self.layers) - 1)

    def _lookup(self, z: ndarray, attr: str) -> ndarray:
        """Vectorised lookup of scalar WaterParams attribute `attr` at depths z."""
        values = np.asarray(
            [getattr(wp, attr) for _, wp in self.layers], dtype=np.float64
        )
        return values[self._layer_index(z)]

    # ------------------------------------------------------------------
    # MediumProfile interface
    # ------------------------------------------------------------------
    def attenuation(self, z: ndarray) -> ndarray:
        return self._lookup(z, 'c')

    def scattering(self, z: ndarray) -> ndarray:
        return self._lookup(z, 'b')

    def asymmetry(self, z: ndarray) -> ndarray:
        return self._lookup(z, 'g')

    def albedo(self, z: ndarray) -> ndarray:
        b = self._lookup(z, 'b')
        c = self._lookup(z, 'c')
        return b / np.where(c > 0, c, 1.0)

    def is_homogeneous(self) -> bool:
        return len(self.layers) == 1

    def summary(self) -> str:
        parts = []
        prev = 0.0
        for zb, wp in self.layers:
            depth_str = f"∞" if np.isinf(zb) else f"{zb:.1f}"
            parts.append(f"  [{prev:.1f}–{depth_str} m] {wp.name} "
                         f"(c={wp.c} m⁻¹ ω={wp.omega:.3f} g={wp.g})")
            prev = zb
        return f"LayeredMedium '{self.name}'\n" + "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# GradientMedium  — continuously depth-varying IOPs via linear interpolation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GradientMedium(MediumProfile):
    """
    Continuously varying IOP profile sampled at discrete depths.

    IOPs are linearly interpolated between sample depths; outside the sampled
    range the nearest boundary value is clamped (np.interp default).

    Parameters
    ----------
    z_samples : 1-D tuple of depths (m), strictly increasing, shape (K,).
    c_samples : beam-attenuation at each depth (m⁻¹), shape (K,).
    b_samples : scattering coefficient at each depth (m⁻¹), shape (K,).
    g_samples : HG asymmetry parameter at each depth, shape (K,).
    name      : descriptive label.

    Use-case
    --------
    Ideal for profiles obtained from in-situ instruments (CTD, AC-9,
    HydroScat) where IOPs are measured at discrete depths and the physical
    expectation is a smooth depth gradient rather than sharp interfaces.

    Tuples are used instead of arrays to keep the object hashable and
    compatible with the `frozen=True` dataclass requirement.

    Woodcock majorant
    -----------------
    `c_max = max(c_samples)`.  If the true profile peak lies between two
    sample points and the interpolation undershoots, the majorant can be
    set conservatively by the caller via the `c_max_override` parameter.
    """
    z_samples:      Tuple[float, ...]
    c_samples:      Tuple[float, ...]
    b_samples:      Tuple[float, ...]
    g_samples:      Tuple[float, ...]
    name:           str   = "Gradient Medium"
    c_max_override: float = 0.0   # 0 → auto (max of c_samples)

    def __post_init__(self) -> None:
        K = len(self.z_samples)
        for arr_name, arr in [('c_samples', self.c_samples),
                               ('b_samples', self.b_samples),
                               ('g_samples', self.g_samples)]:
            if len(arr) != K:
                raise ValueError(
                    f"GradientMedium: {arr_name} must have the same length as "
                    f"z_samples ({K}), got {len(arr)}."
                )
        zs = np.asarray(self.z_samples)
        if not np.all(np.diff(zs) > 0):
            raise ValueError("GradientMedium: z_samples must be strictly increasing.")

    @property
    def c_max(self) -> float:
        if self.c_max_override > 0:
            return float(self.c_max_override)
        return float(max(self.c_samples))

    def _interp(self, z: ndarray, samples: Tuple[float, ...]) -> ndarray:
        zs = np.asarray(self.z_samples, dtype=np.float64)
        vs = np.asarray(samples,        dtype=np.float64)
        return np.interp(z, zs, vs)

    def attenuation(self, z: ndarray) -> ndarray:
        return self._interp(z, self.c_samples)

    def scattering(self, z: ndarray) -> ndarray:
        return self._interp(z, self.b_samples)

    def asymmetry(self, z: ndarray) -> ndarray:
        return self._interp(z, self.g_samples)

    def albedo(self, z: ndarray) -> ndarray:
        b = self._interp(z, self.b_samples)
        c = self._interp(z, self.c_samples)
        return b / np.where(c > 0, c, 1.0)

    def is_homogeneous(self) -> bool:
        return False

    def summary(self) -> str:
        return (f"GradientMedium '{self.name}' "
                f"(K={len(self.z_samples)} samples  "
                f"c_max={self.c_max:.3f} m⁻¹  "
                f"c_range=[{min(self.c_samples):.3f}, {max(self.c_samples):.3f}])")


# ─────────────────────────────────────────────────────────────────────────────
# Preset inhomogeneous medium instances
# ─────────────────────────────────────────────────────────────────────────────

# ── Two-layer: surface clear + deep coastal ───────────────────────────────────
# Represents a 25-m link through a stratified tropical/subtropical ocean where
# the upper 10 m is the well-lit mixed layer and 10–25 m crosses the nutricline
# into the deep chlorophyll maximum.
STRATIFIED_OCEAN = LayeredMedium(
    layers=(
        (10.0,      CLEAR_WATER),    # 0–10 m  : surface mixed layer
        (np.inf,    COASTAL_WATER),  # 10+ m   : deep chlorophyll maximum
    ),
    name="Stratified Ocean (Clear → Coastal)",
)

# ── Three-layer: clear → coastal → turbid ────────────────────────────────────
# Models an estuarine or near-shore channel with a fresh-water plume at depth.
# Suitable for links longer than 15 m in mixed-water environments.
DEEP_OCEAN_COLUMN = LayeredMedium(
    layers=(
        (8.0,    CLEAR_WATER),     # 0–8 m   : surface layer
        (18.0,   COASTAL_WATER),   # 8–18 m  : DCM / nutricline
        (np.inf, TURBID_WATER),    # 18+ m   : turbid bottom layer
    ),
    name="Deep Ocean Column (Clear → Coastal → Turbid)",
)

# ── Smooth exponential-like gradient (modelled as 7 sample points) ─────────
# Models the Beer-law-like increase in CDOM with depth seen in many coastal
# water bodies.  IOPs transition smoothly from clear at the surface to
# turbid at 25 m.
_Z_GRAD    = (0.0,   4.0,   8.0,   12.0,  16.0,  20.0,  25.0)
_C_GRAD    = (0.241, 0.350, 0.520, 0.720, 1.100, 1.600, 2.190)
_B_GRAD    = (0.090, 0.140, 0.230, 0.380, 0.620, 1.100, 1.824)
_G_GRAD    = (0.924, 0.928, 0.932, 0.936, 0.940, 0.942, 0.945)

COASTAL_GRADIENT = GradientMedium(
    z_samples = _Z_GRAD,
    c_samples = _C_GRAD,
    b_samples = _B_GRAD,
    g_samples = _G_GRAD,
    name      = "Coastal Gradient (clear surface → turbid bottom)",
)

# ── Convenience collection of all inhomogeneous presets ──────────────────────
ALL_INHOMOGENEOUS_MEDIA = (STRATIFIED_OCEAN, DEEP_OCEAN_COLUMN, COASTAL_GRADIENT)