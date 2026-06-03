"""
uowc.turbulence
===============
Underwater optical turbulence — Nikishov parameterisation, phase-screen
angular kicks, scintillation fading, and coupled ocean-layer model.

Separation-of-Concern role
---------------------------
  This module owns all turbulence physics: how C_n² is derived from physical
  ocean parameters, how phase screens induce angular deflections, and how
  accumulated wavefront distortion translates into scintillation fading.
  It knows nothing about photon transport mechanics, simulation sweeps,
  Pandas, or plotting.

Two distinct physical effects (both modelled)
---------------------------------------------
1. Phase-screen angular kicks  (stochastic beam wander + small-scale spreading)
   At every transport step of length s, each photon receives a transverse
   angular perturbation drawn from the Markov-limit phase-screen variance:

       σ²_α(z, s) = 3.04 · l₀^(-1/3) · C_n²(z) · s     [rad²]

   This is the "continuous random-walk" limit of many thin screens — it
   naturally models both beam wander (large eddies deflecting the whole beam)
   and small-scale spreading.  The kicks accumulate coherently along the path
   and affect aperture/FOV capture probability directly.

2. Scintillation fading at receiver  (intensity fluctuations)
   After aperture+FOV acceptance, each photon weight is multiplied by h_t,
   a random fading coefficient with E[h_t]=1.  The fading distribution is
   regime-dependent, selected by the path-integrated Rytov variance:
       σ²_R = 1.23 · C_n² · k^(7/6) · L^(11/6)

   Weak  (σ²_R < 0.3) → log-normal
   Moderate (0.3–5)   → Gamma-Gamma (Andrews & Phillips 2001)
   Strong (≥5)         → negative-exponential

Nikishov parameterisation
--------------------------
Turbulence strength is specified by ocean physical parameters (not C_n²
directly), following Nikishov & Nikishov (2000):

    C_n²(z) = 2.05×10⁻⁸ · ε(z)^(−1/3) · χ_T(z) · (dn/dT)²

  ε     [m²/s³]  — kinetic energy dissipation rate
  χ_T   [K²/s]   — temperature variance dissipation rate
  dn/dT [K⁻¹]   — refractive index temperature gradient ≈ −1.80×10⁻⁴ at 530 nm

Typical ocean values:
  Open ocean (calm):      ε~10⁻⁹,  χ_T~10⁻¹⁰  → C_n²~10⁻¹⁵
  Coastal surface:        ε~10⁻⁷,  χ_T~10⁻⁸   → C_n²~10⁻¹³
  Active thermocline:     ε~10⁻⁶,  χ_T~10⁻⁶   → C_n²~10⁻¹¹

Layer alignment — CoupledOceanMedium
--------------------------------------
  The key class in this module is CoupledOceanMedium.  It bundles WaterParams
  (IOPs) and turbulence params (ε, χ_T) into one OceanLayer per depth slab.
  One set of boundaries controls BOTH optical and turbulence properties,
  eliminating the physical inconsistency that arises when separate LayeredMedium
  and DepthVaryingTurbulence objects have different depth grids.

  CoupledOceanMedium implements both the MediumProfile and TurbulenceProfile
  interfaces so the transport kernel calls one object for everything.

References
----------
  Nikishov & Nikishov (2000)  Int. J. Fluid Mech. Res. 27(1):82–98
  Korotkova et al. (2012)     Waves Random Complex Media 22(2):260–266
  Andrews & Phillips (2001)   Laser Beam Scintillation. SPIE Press.
  Yi et al. (2015)            Opt. Express 23(4):4886–4895
  Quan & Fry (1995)           Appl. Opt. 34(18):3477–3480
  Thorpe (2005)               The Turbulent Ocean. Cambridge UP.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray

from uowc.config import WaterParams

# ── optical constants at 530 nm ───────────────────────────────────────────────
_LAMBDA_M: float = 530e-9
_K:        float = 2.0 * np.pi / _LAMBDA_M      # wavenumber (m⁻¹)
_DN_DT:    float = -1.80e-4                       # dn/dT seawater 530nm (K⁻¹)
_L0_M:     float = 10.0                           # default outer scale (m)
_l0_M:     float = 1e-3                           # default inner scale (m)

# ── Kolmogorov-Corrsin structural constant for the Nikishov formula ────────────
# C_n² = _B0 · (dn/dT)² · ε^(−1/3) · χ_T
# _B0 ≈ 3.63 is the scalar-field Obukhov-Corrsin constant.  This value is
# calibrated so that at the default dn/dT = −1.80×10⁻⁴ K⁻¹ the combined
# prefactor _B0 · (dn/dT)² = 1.176×10⁻⁷ matches published UOWC literature
# (Yi et al. 2015 Opt. Express; Korotkova et al. 2012).
# Exposing _B0 separately lets the formula correctly respond to a custom dn/dT.
_B0: float = 1.176e-7 / (_DN_DT ** 2)            # ≈ 3.63  (dimensionless)


# ─────────────────────────────────────────────────────────────────────────────
# Pure physics functions
# ─────────────────────────────────────────────────────────────────────────────

def nikishov_cn2(
    epsilon: float,
    chi_T:   float,
    dn_dT:   float = _DN_DT,
) -> float:
    """
    Refractive-index structure parameter from Nikishov & Nikishov (2000).

        C_n² = _B0 · (dn/dT)² · ε^(−1/3) · χ_T   [m⁻²/³]

    where _B0 ≈ 3.63 is the Kolmogorov-Corrsin scalar-turbulence constant.

    At the default dn/dT = −1.80×10⁻⁴ K⁻¹ the combined prefactor
    _B0 · (dn/dT)² = 1.176×10⁻⁷ matches the UOWC literature value.

    Parameters
    ----------
    epsilon : kinetic energy dissipation rate (m²/s³)
    chi_T   : temperature variance dissipation rate (K²/s)
    dn_dT   : refractive-index temperature gradient (K⁻¹).
              Default −1.80×10⁻⁴ K⁻¹ for seawater at 530 nm.
              Pass a different value when modelling other wavelengths
              or water temperatures; Quan & Fry (1995) provide the
              dispersion formula.
    """
    if epsilon <= 0.0 or chi_T <= 0.0:
        return 0.0
    return _B0 * (dn_dT ** 2) * (epsilon ** (-1.0 / 3.0)) * chi_T


def nikishov_cn2_array(
    epsilon: ndarray,
    chi_T:   ndarray,
    dn_dT:   float = _DN_DT,
) -> ndarray:
    """
    Vectorised version of nikishov_cn2 over depth arrays.

    C_n²[i] = _B0 · (dn/dT)² · ε[i]^(−1/3) · χ_T[i]
    Elements with ε ≤ 0 or χ_T ≤ 0 are set to zero.
    """
    out   = np.zeros(epsilon.shape, dtype=np.float64)
    valid = (epsilon > 0) & (chi_T > 0)
    out[valid] = (_B0 * (dn_dT ** 2)
                  * epsilon[valid] ** (-1.0 / 3.0)
                  * chi_T[valid])
    return out


def screen_angular_variance(
    cn2:  float | ndarray,
    dz:   float | ndarray,
    l0:   float = _l0_M,
) -> float | ndarray:
    """
    Angular kick variance per phase screen of thickness dz.

    Derived from the Kolmogorov point-receiver tilt formula in the
    Markov (continuous medium) limit:

        σ²_α = 3.04 · l₀^(−1/3) · C_n² · dz      [rad²]

    Parameters
    ----------
    cn2 : local C_n² (m⁻²/³), scalar or array
    dz  : screen thickness / step length (m), scalar or array
    l0  : Kolmogorov inner scale (m), default 1 mm
    """
    return 3.04 * (l0 ** (-1.0 / 3.0)) * cn2 * dz


def fried_parameter(cn2_path_integral: float, k: float = _K) -> float:
    """
    Fried coherence radius r₀  [m].

        r₀ = (0.423 · k² · ∫C_n² dz)^(−3/5)

    Parameters
    ----------
    cn2_path_integral : ∫C_n²(z) dz over the full link (m⁻²/³ · m = m^(1/3))
    k                 : optical wavenumber (m⁻¹)
    """
    if cn2_path_integral <= 0.0:
        return float("inf")
    return (0.423 * k ** 2 * cn2_path_integral) ** (-3.0 / 5.0)


def rytov_variance_from_cn2(cn2: float, L: float, k: float = _K) -> float:
    """
    Rytov variance σ²_R = 1.23 · C_n² · k^(7/6) · L^(11/6).
    """
    if cn2 <= 0.0 or L <= 0.0:
        return 0.0
    return 1.23 * cn2 * (k ** (7.0 / 6.0)) * (L ** (11.0 / 6.0))


def turbulence_regime(sigma_R2: float) -> str:
    if sigma_R2 < 0.3:  return "weak"
    if sigma_R2 < 5.0:  return "moderate"
    return "strong"


def coherence_time(r0: float, v_current: float) -> float:
    """
    Turbulence coherence time τ_c = r₀ / v_⊥  (Taylor frozen turbulence).

    Parameters
    ----------
    r0        : Fried parameter (m)
    v_current : transverse current speed (m/s)
    """
    if v_current <= 0.0 or r0 == float("inf"):
        return float("inf")
    return r0 / v_current


def gamma_gamma_params(sigma_R2: float) -> tuple[float, float]:
    """Gamma-Gamma shape parameters (α, β) from Andrews & Phillips (2001)."""
    sX = 0.49 * sigma_R2 / ((1.0 + 0.56 * sigma_R2 ** (6.0/5.0)) ** (7.0/6.0))
    sY = 0.51 * sigma_R2 / ((1.0 + 0.69 * sigma_R2 ** (6.0/5.0)) ** (5.0/6.0))
    alpha = 1.0 / max(np.exp(sX) - 1.0, 1e-10)
    beta  = 1.0 / max(np.exp(sY) - 1.0, 1e-10)
    return float(alpha), float(beta)


def sample_scintillation_fading(
    cn2_eff:    ndarray,
    link_range: float,
    rng:        np.random.Generator,
) -> ndarray:
    """
    Sample per-photon scintillation fading h_t at the receiver.

    E[h_t] = 1 for all regimes.  The distribution is selected per photon
    from the Rytov variance:

        σ²_R = 1.23 · C_n²_eff · k^(7/6) · L^(11/6)

    Parameters
    ----------
    cn2_eff    : effective path-average C_n² per photon, shape (N,)
    link_range : link length L (m)
    rng        : numpy Generator

    Returns
    -------
    h_t : fading weights, shape (N,), all ≥ 0
    """
    N        = len(cn2_eff)
    h_t      = np.ones(N, dtype=np.float64)
    sigma_R2 = 1.23 * cn2_eff * (_K ** (7.0/6.0)) * (link_range ** (11.0/6.0))

    # Weak → log-normal
    wk = sigma_R2 < 0.3
    if wk.any():
        sR = sigma_R2[wk]
        sX = 0.49 * sR / ((1.0 + 0.56 * sR**(6/5))**(7/6))
        X  = rng.normal(loc=-sX, scale=np.sqrt(np.maximum(sX, 1e-30)))
        h_t[wk] = np.exp(2.0 * X)

    # Moderate → Gamma-Gamma (per-photon loop is unavoidable here)
    md = (sigma_R2 >= 0.3) & (sigma_R2 < 5.0)
    if md.any():
        for i in np.where(md)[0]:
            a, b   = gamma_gamma_params(float(sigma_R2[i]))
            h_t[i] = rng.gamma(a, 1.0/a) * rng.gamma(b, 1.0/b)

    # Strong → exponential
    st = sigma_R2 >= 5.0
    if st.any():
        h_t[st] = rng.exponential(scale=1.0, size=st.sum())

    return np.maximum(h_t, 0.0)


def apply_angular_kick(
    ux:          ndarray,
    uy:          ndarray,
    uz:          ndarray,
    sigma2_alpha: ndarray,
    rng:          np.random.Generator,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Apply random transverse angular perturbation from turbulence phase screen.

    Each photon receives an independent Gaussian angular kick in the two
    transverse directions perpendicular to its propagation axis:

        Δα_x, Δα_y ~ N(0, σ²_α / 2)

    The factor 1/2 distributes the total variance equally between the two
    transverse axes.  The direction is renormalised after the kick.

    Parameters
    ----------
    ux, uy, uz   : direction cosines, shape (N,)
    sigma2_alpha : per-photon total angular variance σ²_α(z,s), shape (N,)
    rng          : numpy Generator

    Returns
    -------
    (ux_new, uy_new, uz_new) : renormalised direction cosines
    """
    N     = len(ux)
    sigma = np.sqrt(np.maximum(sigma2_alpha / 2.0, 0.0))

    dux = rng.normal(loc=0.0, scale=sigma)
    duy = rng.normal(loc=0.0, scale=sigma)

    ux_new = ux + dux
    uy_new = uy + duy
    # uz regenerated from unit-vector constraint, preserving sign
    uz_new = np.sign(uz) * np.sqrt(np.maximum(1.0 - ux_new**2 - uy_new**2, 0.0))

    # Renormalise (guard against extreme kicks)
    norm = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    norm = np.where(norm > 1e-15, norm, 1.0)
    return ux_new / norm, uy_new / norm, uz_new / norm


# ─────────────────────────────────────────────────────────────────────────────
# OceanLayer — one depth slab with aligned IOP + turbulence parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OceanLayer:
    """
    One depth slab carrying both optical and turbulence properties.

    Having both in a single object guarantees layer boundaries are
    identical for IOP lookups and turbulence calculations — a photon
    in "coastal water" is also in the correct turbulence regime.

    Parameters
    ----------
    z_bottom_m : lower boundary of this layer (m).  Last layer → np.inf.
    water      : WaterParams (c, a, b, g) for this slab.
    epsilon    : kinetic energy dissipation rate ε (m²/s³).
                 Controls turbulence intensity.  Typical range 10⁻⁹–10⁻⁵.
    chi_T      : temperature variance dissipation rate χ_T (K²/s).
                 Controls refractive-index fluctuation amplitude.
                 Typical range 10⁻¹⁰–10⁻⁶.
    v_current  : mean horizontal current speed (m/s).
                 Used to compute channel coherence time τ_c = r₀ / v.
                 Default 0.1 m/s (typical sub-surface current).
    """
    z_bottom_m: float
    water:      WaterParams
    epsilon:    float          # m²/s³
    chi_T:      float          # K²/s
    v_current:  float = 0.1    # m/s

    @property
    def cn2(self) -> float:
        """C_n² derived from Nikishov formula (m⁻²/³)."""
        return nikishov_cn2(self.epsilon, self.chi_T)


# ─────────────────────────────────────────────────────────────────────────────
# TurbulenceProfile base class
# ─────────────────────────────────────────────────────────────────────────────

class TurbulenceProfile:
    """
    Interface for a depth-dependent turbulence description.

    Vectorised: all methods accept ndarray z of shape (N,) and return
    ndarray of the same shape.  Mirrors the MediumProfile convention.
    """

    @property
    def name(self) -> str:
        raise NotImplementedError

    def C_n2(self, z: ndarray) -> ndarray:
        """C_n²(z) [m⁻²/³], shape (N,)."""
        raise NotImplementedError

    def dn_dz_mean(self, z: ndarray) -> ndarray:
        """Mean index gradient dn/dz (m⁻¹) for deterministic bending."""
        raise NotImplementedError

    def v_current(self, z: ndarray) -> ndarray:
        """Horizontal current speed (m/s) for coherence time, shape (N,)."""
        raise NotImplementedError

    def is_turbulence_free(self) -> bool:
        raise NotImplementedError

    def path_cn2_integral(self, link_range: float) -> float:
        """
        Path integral  ∫₀^L C_n²(z) dz  [m^(1/3)].

        Used for computing the path-averaged Rytov variance:

            σ²_R = 1.23 · k^(7/6) · L^(5/6) · ∫₀^L C_n²(z) dz
                                                   ─────────────
                                                      (plane wave)

        The default implementation uses the midpoint rule on a uniform
        depth grid with 0.5 m spacing.  Subclasses with analytically
        known profiles (e.g. CoupledOceanMedium) should override this
        with an exact or more efficient computation.

        Parameters
        ----------
        link_range : total link length L (m)

        Returns
        -------
        integral : ∫₀^L C_n²(z) dz  (m^(1/3))
        """
        n_pts = max(int(link_range / 0.5), 10)
        z     = np.linspace(0.0, link_range, n_pts)
        dz    = z[1] - z[0]
        return float(np.sum(self.C_n2(z)) * dz)

    def summary(self) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# NoTurbulence  — null object, zero overhead
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoTurbulence(TurbulenceProfile):
    """Zero turbulence everywhere. Skips all turbulence operations."""

    @property
    def name(self) -> str:
        return "No Turbulence"

    def C_n2(self, z: ndarray) -> ndarray:
        return np.zeros(z.shape, dtype=np.float64)

    def dn_dz_mean(self, z: ndarray) -> ndarray:
        return np.zeros(z.shape, dtype=np.float64)

    def v_current(self, z: ndarray) -> ndarray:
        return np.full(z.shape, 0.1, dtype=np.float64)

    def is_turbulence_free(self) -> bool:
        return True

    def path_cn2_integral(self, link_range: float) -> float:
        """Zero turbulence everywhere — integral is identically zero."""
        return 0.0

    def summary(self) -> str:
        return "NoTurbulence  (C_n² = 0 everywhere)"


# ─────────────────────────────────────────────────────────────────────────────
# CoupledOceanMedium  — aligned IOP + turbulence in one object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoupledOceanMedium:
    """
    Depth-stratified channel model coupling optical IOPs and turbulence.

    Single layer stack — one set of boundaries for both IOP and turbulence
    lookups.  This is the primary class for realistic simulations: a photon
    in a given depth slab always uses the same WaterParams and (ε, χ_T)
    pair, eliminating boundary-misalignment errors.

    Implements both the MediumProfile and TurbulenceProfile protocols so
    the transport kernel calls one object for all channel properties.

    Parameters
    ----------
    layers  : tuple of OceanLayer, sorted ascending by z_bottom_m.
              The last layer must have z_bottom_m = np.inf.
    name    : descriptive label for figures and console output.
    l0_m    : Kolmogorov inner scale (m), default 1 mm.
    L0_m    : outer turbulence scale (m), default 10 m.

    Example — 25 m stratified link crossing the thermocline:
    ::

        CoupledOceanMedium(
            layers=(
                OceanLayer( 8.0, CLEAR_WATER,   epsilon=1e-8,  chi_T=1e-9),
                OceanLayer(18.0, COASTAL_WATER,  epsilon=5e-7,  chi_T=5e-7),
                OceanLayer(np.inf, TURBID_WATER, epsilon=1e-8,  chi_T=1e-9),
            ),
            name="Stratified thermocline link",
        )
    """
    layers: Tuple[OceanLayer, ...]
    name:   str   = "Coupled Ocean Medium"
    l0_m:   float = 1e-3
    L0_m:   float = 10.0

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("CoupledOceanMedium requires at least one layer.")
        bounds = [ly.z_bottom_m for ly in self.layers]
        for i in range(len(bounds) - 1):
            if bounds[i] >= bounds[i + 1]:
                raise ValueError("Layer z_bottom_m must be strictly increasing.")
        if not np.isinf(self.layers[-1].z_bottom_m):
            raise ValueError("Last layer z_bottom_m must be np.inf.")

    # ── Internal ────────────────────────────────────────────────────────────
    def _idx(self, z: ndarray) -> ndarray:
        bounds = np.asarray([ly.z_bottom_m for ly in self.layers], dtype=np.float64)
        return np.clip(np.searchsorted(bounds, z, side="right"),
                       0, len(self.layers) - 1)

    def _get(self, z: ndarray, attr: str) -> ndarray:
        """Vectorised attribute lookup from WaterParams."""
        vals = np.asarray([getattr(ly.water, attr) for ly in self.layers],
                          dtype=np.float64)
        return vals[self._idx(z)]

    # ── MediumProfile interface ───────────────────────────────────────────────
    @property
    def c_max(self) -> float:
        return float(max(ly.water.c for ly in self.layers))

    def attenuation(self, z: ndarray) -> ndarray:
        return self._get(z, "c")

    def scattering(self, z: ndarray) -> ndarray:
        return self._get(z, "b")

    def asymmetry(self, z: ndarray) -> ndarray:
        return self._get(z, "g")

    def albedo(self, z: ndarray) -> ndarray:
        b = self._get(z, "b")
        c = self._get(z, "c")
        return b / np.where(c > 0, c, 1.0)

    def is_homogeneous(self) -> bool:
        return len(self.layers) == 1

    # ── TurbulenceProfile interface ──────────────────────────────────────────
    def C_n2(self, z: ndarray) -> ndarray:
        cn2_vals = np.asarray([ly.cn2 for ly in self.layers], dtype=np.float64)
        return cn2_vals[self._idx(z)]

    def dn_dz_mean(self, z: ndarray) -> ndarray:
        """
        Approximate mean vertical refractive-index gradient dn/dz (m⁻¹).

        Uses the Osborn-Cox relation to estimate the temperature gradient:
            |∂T/∂z| = √(χ_T / (2 K_T))

        where K_T = 1.4×10⁻⁷ m²/s is the molecular thermal diffusivity.
        Then:  dn/dz = (dn/dT) × ∂T/∂z

        Typical magnitudes (Thorpe 2005):
          Calm ocean:    dn/dz ~ 3×10⁻⁷ m⁻¹  (negligible bending)
          Coastal water: dn/dz ~ 1×10⁻⁵ m⁻¹
          Thermocline:   dn/dz ~ 2×10⁻⁴ m⁻¹  (significant beam bending)
        """
        KT  = 1.4e-7
        idx = self._idx(z)
        out = np.zeros(z.shape, dtype=np.float64)
        for i, ly in enumerate(self.layers):
            mask = idx == i
            if not mask.any() or ly.chi_T <= 0:
                continue
            dT_dz     = np.sqrt(ly.chi_T / (2.0 * KT))
            out[mask] = _DN_DT * dT_dz
        return out

    def v_current(self, z: ndarray) -> ndarray:
        v_vals = np.asarray([ly.v_current for ly in self.layers], dtype=np.float64)
        return v_vals[self._idx(z)]

    def is_turbulence_free(self) -> bool:
        return all(ly.cn2 == 0.0 for ly in self.layers)

    # ── Shared summary ────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [f"CoupledOceanMedium  '{self.name}'  (c_max={self.c_max:.3f} m⁻¹)"]
        prev = 0.0
        for ly in self.layers:
            z_str = "∞" if np.isinf(ly.z_bottom_m) else f"{ly.z_bottom_m:.1f}"
            sr2   = rytov_variance_from_cn2(ly.cn2, max(ly.z_bottom_m - prev, 1.0))
            lines.append(
                f"  [{prev:5.1f}–{z_str:6s} m]  "
                f"{ly.water.name:<14s}  "
                f"c={ly.water.c:.3f}  "
                f"ε={ly.epsilon:.2e}  "
                f"χ_T={ly.chi_T:.2e}  "
                f"C_n²={ly.cn2:.2e}  "
                f"[{turbulence_regime(sr2)}]"
            )
            prev = ly.z_bottom_m
        return "\n".join(lines)

    # ── Coherence metrics ─────────────────────────────────────────────────────
    def path_cn2_integral(self, link_range: float) -> float:
        """
        Numerically integrate C_n²(z) from 0 to link_range.
        Uses mid-point rule on the piecewise-constant profile.
        """
        integral = 0.0
        prev = 0.0
        for ly in self.layers:
            z_bot = min(ly.z_bottom_m, link_range)
            dz    = z_bot - prev
            if dz > 0:
                integral += ly.cn2 * dz
            prev = z_bot
            if z_bot >= link_range:
                break
        return integral

    def fried_radius(self, link_range: float) -> float:
        """Fried coherence radius r₀ for this link (m)."""
        return fried_parameter(self.path_cn2_integral(link_range))

    def channel_coherence_time(self, link_range: float) -> float:
        """
        Estimated channel coherence time τ_c = r₀ / v̄_⊥  (s).

        Uses the depth-average current speed weighted by C_n².
        """
        integral_cn2 = self.path_cn2_integral(link_range)
        if integral_cn2 <= 0.0:
            return float("inf")
        # C_n²-weighted mean current
        weighted_v = 0.0
        prev = 0.0
        for ly in self.layers:
            z_bot = min(ly.z_bottom_m, link_range)
            dz    = z_bot - prev
            if dz > 0 and ly.cn2 > 0:
                weighted_v += ly.cn2 * dz * ly.v_current
            prev = z_bot
            if z_bot >= link_range:
                break
        v_eff = weighted_v / integral_cn2 if integral_cn2 > 0 else 0.1
        r0    = fried_parameter(integral_cn2)
        return coherence_time(r0, v_eff)


# ─────────────────────────────────────────────────────────────────────────────
# Preset coupled ocean profiles
# ─────────────────────────────────────────────────────────────────────────────

from uowc.config import CLEAR_WATER, COASTAL_WATER, TURBID_WATER

# ── Calm open-ocean link  ────────────────────────────────────────────────────
# ε=1e-9, χ_T=8.5e-12  →  C_n²≈1e-15 m⁻²/³  (weak, σ²_R≪0.3)
CALM_OPEN_OCEAN = CoupledOceanMedium(
    layers=(
        OceanLayer(np.inf, CLEAR_WATER,
                   epsilon=1e-9, chi_T=8.5e-12, v_current=0.05),
    ),
    name="Calm Open Ocean (homogeneous)",
)

# ── Coastal surface layer  ───────────────────────────────────────────────────
# ε=1e-7, χ_T=3.95e-9  →  C_n²≈1e-13 m⁻²/³  (weak-to-moderate)
COASTAL_SURFACE = CoupledOceanMedium(
    layers=(
        OceanLayer(np.inf, COASTAL_WATER,
                   epsilon=1e-7, chi_T=3.95e-9, v_current=0.15),
    ),
    name="Coastal Surface Layer",
)

# ── Stratified thermocline crossing  (25 m link) ─────────────────────────────
# Boundaries match STRATIFIED_OCEAN in uowc.medium — IOPs and turbulence
# change at the SAME depth (10 m), ensuring physical consistency.
#   0–10 m  : CLEAR,   ε=1e-9,  χ_T=8.5e-12 → C_n²≈1e-15  [weak]
#   10+ m   : COASTAL, ε=1e-5,  χ_T=9.16e-7 → C_n²≈5e-12  [moderate]
STRATIFIED_THERMOCLINE = CoupledOceanMedium(
    layers=(
        OceanLayer(10.0,   CLEAR_WATER,
                   epsilon=1e-9,  chi_T=8.5e-12, v_current=0.08),
        OceanLayer(np.inf, COASTAL_WATER,
                   epsilon=1e-5,  chi_T=9.16e-7, v_current=0.12),
    ),
    name="Stratified Thermocline (Clear → Coastal)",
)

# ── Deep ocean column  ───────────────────────────────────────────────────────
# Boundaries match DEEP_OCEAN_COLUMN in uowc.medium: 0–8, 8–18, 18+ m
#   0–8 m   : CLEAR,   C_n²≈1e-15  [weak]
#   8–18 m  : COASTAL, C_n²≈5e-12  [moderate, thermocline crossing]
#   18+ m   : TURBID,  C_n²≈1e-14  [weak again, below active mixing]
DEEP_OCEAN_STRATIFIED = CoupledOceanMedium(
    layers=(
        OceanLayer(8.0,    CLEAR_WATER,
                   epsilon=1e-9,  chi_T=8.5e-12, v_current=0.05),
        OceanLayer(18.0,   COASTAL_WATER,
                   epsilon=1e-5,  chi_T=9.16e-7, v_current=0.12),
        OceanLayer(np.inf, TURBID_WATER,
                   epsilon=1e-8,  chi_T=8.5e-11, v_current=0.08),
    ),
    name="Deep Ocean Column (Clear → Coastal → Turbid)",
)

# ── Convenience null turbulence object ────────────────────────────────────────
NO_TURBULENCE = NoTurbulence()

ALL_COUPLED_MEDIA = (
    CALM_OPEN_OCEAN,
    COASTAL_SURFACE,
    STRATIFIED_THERMOCLINE,
    DEEP_OCEAN_STRATIFIED,
)


def compute_coherence_time(medium, link_range: float) -> float:
    """Channel coherence time τ_c (s) via Taylor frozen-turbulence hypothesis.

    τ_c = r₀ / v̄_⊥

    Parameters
    ----------
    medium     : CoupledOceanMedium or any object with .C_n2(z) and .v_current(z),
                 or one that exposes .channel_coherence_time(link_range) directly.
    link_range : link length (m)

    Returns
    -------
    τ_c in seconds, or np.inf if the medium carries no turbulence.
    """
    if hasattr(medium, "channel_coherence_time"):
        return float(medium.channel_coherence_time(link_range))

    z_arr = np.linspace(0, link_range, max(int(link_range / 0.5), 10))
    dz    = z_arr[1] - z_arr[0]
    cn2   = medium.C_n2(z_arr)
    r0    = fried_parameter(float((cn2 * dz).sum()))
    v_mid = float(medium.v_current(np.array([link_range / 2.0]))[0])
    return coherence_time(r0, v_mid)
