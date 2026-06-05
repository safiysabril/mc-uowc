"""
uowc.physics
============
Pure functions encoding the underwater optical physics.

Separation-of-Concern role
---------------------------
  This module knows about optics and photon-matter interactions only.
  It has no awareness of receivers, parallelism, file I/O, or plotting.
  Every function is stateless (except the two in-place absorption helpers
  which mutate a caller-owned array) — safe to test in isolation.

Public API
----------
  Beer-Lambert
    beer_lambert_power_dB

  Free-flight step sampling
    sample_step_length          — homogeneous medium (Beer-Lambert inverse-CDF)
    sample_step_woodcock        — inhomogeneous medium (Woodcock majorant)

  Woodcock acceptance
    accept_real_collision       — thinning-theorem acceptance test

  Implicit absorption  (in-place, correct NumPy semantics)
    apply_absorption            — scalar albedo, homogeneous path
    apply_absorption_array      — per-photon albedo, Woodcock path

  Variance reduction
    russian_roulette            — full-array roulette (homogeneous)
    russian_roulette_targeted   — index-targeted roulette (Woodcock)

  Henyey-Greenstein phase function
    sample_hg_cos_theta         — scalar g (homogeneous path)
    sample_hg_cos_theta_array   — per-photon g array (Woodcock path)
    rotate_direction            — MCML frame rotation with near-z singularity fix

  Launch geometry
    sample_launch_positions     — TEM₀₀ Gaussian beam positions
    sample_launch_directions    — solid-angle-uniform cone sampling

  Receiver optics
    fresnel_transmittance       — Fresnel power T at planar water→glass interface

  Time-of-flight
    path_length_to_tof          — path length → propagation delay

References
----------
  Gabriel et al. (2013)  JOCN 5(1):1-12      — HG phase function, IOPs
  Mobley & Preisendorfer (1994)               — seawater radiative transfer
  Haltrin (1999) Applied Optics 38(33):6826   — chlorophyll-based bio-optics
  Wang, Jacques & Zheng (1995) CPC 47:131-146 — MCML algorithm
  Woodcock et al. (1965) — delta-tracking majorant method
  Lux & Koblinger (1991) — Monte Carlo Particle Transport, CRC Press
  Born & Wolf (1999) — Principles of Optics, §1.5 — Fresnel equations
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray

from uowc.config import C_MEDIUM, N_WATER


# ─────────────────────────────────────────────────────────────────────────────
# Beer-Lambert reference
# ─────────────────────────────────────────────────────────────────────────────

def beer_lambert_power_dB(c: float, distance_m: float) -> float:
    """
    Normalised received power from the Beer-Lambert law (dB).

        P = exp(-c · d)   →   P_dB = -c · d · 10 / ln(10)

    No scattering, no geometry — a deterministic lower bound on path loss.
    """
    return 10.0 * np.log10(np.exp(-c * distance_m) + 1e-300)


# ─────────────────────────────────────────────────────────────────────────────
# Free-flight sampling  (homogeneous medium)
# ─────────────────────────────────────────────────────────────────────────────

def sample_step_length(c: float, xi: ndarray) -> ndarray:
    """
    Sample exponentially distributed free-flight step lengths.

        s = -ln(ξ) / c         [Beer-Lambert inverse-CDF]

    Parameters
    ----------
    c  : beam-attenuation coefficient (m⁻¹)  — scalar for homogeneous medium
    xi : uniform random variates in (0, 1], shape (N,)

    Returns
    -------
    s : step lengths (m), shape (N,)
    """
    return -np.log(np.clip(xi, 1e-15, 1.0)) / c


# ─────────────────────────────────────────────────────────────────────────────
# Free-flight sampling  (inhomogeneous medium — Woodcock delta tracking)
# ─────────────────────────────────────────────────────────────────────────────

def sample_step_woodcock(c_max: float, xi: ndarray) -> ndarray:
    """
    Sample free-flight step lengths using the Woodcock majorant.

    In a spatially varying medium where c(z) ≤ c_max everywhere, we can
    sample the step length with the constant-coefficient c_max and then
    decide whether the resulting collision is "real" or "null" (virtual).
    The net effect is statistically equivalent to sampling from the correct
    inhomogeneous path-length distribution.

    This is identical to `sample_step_length(c_max, xi)` — the distinction
    is conceptual: calling this function signals to the reader that the step
    will be subject to a Woodcock acceptance test downstream.

        s = -ln(ξ) / c_max

    Parameters
    ----------
    c_max : global upper bound on c(z) — the Woodcock majorant (m⁻¹)
    xi    : uniform random variates in (0, 1], shape (N,)

    Returns
    -------
    s : tentative step lengths (m), shape (N,)
    """
    return -np.log(np.clip(xi, 1e-15, 1.0)) / c_max


def accept_real_collision(
    c_local: ndarray,
    c_max:   float,
    xi:      ndarray,
) -> ndarray:
    """
    Woodcock acceptance test: decide whether each collision is real or null.

    After advancing a photon by a Woodcock step (sampled with c_max), the
    collision at the new position is accepted as a REAL interaction (i.e.
    actual scattering + absorption) with probability:

        p_real(z) = c(z) / c_max

    A rejected collision is a NULL (virtual) collision — the photon
    continues its trajectory without scattering or weight update.

    Mathematical guarantee
    ----------------------
    By the thinning theorem of Poisson processes (Kingman 1993), the
    sequence of real collisions follows an inhomogeneous Poisson process
    with rate c(z), exactly matching the physical interaction rate.

    Parameters
    ----------
    c_local : local beam-attenuation coefficient c(z) at photon positions,
              shape (N,)
    c_max   : global majorant coefficient (scalar)
    xi      : uniform random variates in [0, 1), shape (N,)

    Returns
    -------
    real : boolean mask, True = real collision, False = null collision, shape (N,)
    """
    return xi < (c_local / c_max)


# ─────────────────────────────────────────────────────────────────────────────
# Implicit-absorption weight update
# ─────────────────────────────────────────────────────────────────────────────

def apply_absorption(w: ndarray, active: ndarray, omega: float) -> None:
    """
    Apply implicit-absorption (MCML) weight reduction in-place for a photon
    subset identified by integer index array `active`.

        w[active] *= ω        ω = b / c  (single-scattering albedo)

    Design note
    -----------
    NumPy fancy indexing with an integer array (e.g. ``np.where(alive)[0]``)
    always returns a **copy**, not a view.  Passing ``w[active]`` to a
    function and doing ``w_arg *= omega`` inside modifies the copy only.
    This function receives the *full* array ``w`` and the index set ``active``
    so the in-place update ``w[active] *= omega`` operates on the original
    data correctly.

    Parameters
    ----------
    w      : full photon weight array, shape (N_chunk,) — modified in-place
    active : 1-D integer index array of photons to update, shape (M,)
    omega  : single-scattering albedo ω = b/c (scalar, homogeneous path)
    """
    w[active] *= omega


def apply_absorption_array(w: ndarray, active: ndarray, omega: ndarray) -> None:
    """
    Per-photon implicit-absorption weight reduction in-place.

        w[active[i]] *= omega[i]    where omega[i] = b(z_i) / c(z_i)

    Used in the Woodcock (inhomogeneous) path where each photon may sit in a
    different layer and therefore has a different local albedo.

    Parameters
    ----------
    w      : full photon weight array, shape (N_chunk,) — modified in-place
    active : 1-D integer index array of real-collision photons, shape (M,)
    omega  : per-photon single-scattering albedo at current layer, shape (M,)
    """
    w[active] *= omega


# ─────────────────────────────────────────────────────────────────────────────
# Russian roulette
# ─────────────────────────────────────────────────────────────────────────────

def russian_roulette(
    w:         ndarray,
    alive:     ndarray,
    xi:        ndarray,
    threshold: float,
    m:         int,
) -> ndarray:
    """
    Kill or boost photons whose weight has fallen below `threshold`.

    Photons survive with probability 1/m and have their weight scaled by m,
    preserving the unbiased estimator  E[W_new] = W_old.

    Parameters
    ----------
    w         : photon weights,               shape (N,)
    alive     : boolean liveness mask,        shape (N,)
    xi        : uniform random variates,      shape (N,) — pre-drawn by caller
    threshold : weight kill threshold (e.g. 1e-4)
    m         : roulette multiplier (e.g. 10)

    Returns
    -------
    alive : updated liveness mask (in-place)
    """
    low = alive & (w < threshold)
    if not low.any():
        return alive
    survive = low & (xi < (1.0 / m))
    kill    = low & ~survive
    w[survive] *= m
    alive[kill] = False
    return alive


def russian_roulette_targeted(
    w:         ndarray,
    alive:     ndarray,
    target:    ndarray,
    xi:        ndarray,
    threshold: float,
    m:         int,
) -> None:
    """
    Russian roulette applied only to a subset of photon indices `target`.

    Used in the inhomogeneous Woodcock path where roulette should only
    be applied to photons that had a real collision in this step (not
    to null-collision photons whose weights did not change).

    Parameters
    ----------
    w, alive  : global photon arrays (size = chunk size)
    target    : 1-D index array of photons to consider
    xi        : uniform variates, shape (len(target),)
    threshold, m : roulette parameters

    Modifies `w` and `alive` in-place for the `target` subset.
    """
    if target.size == 0:
        return
    low_mask = w[target] < threshold
    low_idx  = target[low_mask]
    if low_idx.size == 0:
        return
    low_xi  = xi[low_mask]
    survive = low_xi < (1.0 / m)
    w[low_idx[survive]] *= m
    alive[low_idx[~survive]] = False


# ─────────────────────────────────────────────────────────────────────────────
# Henyey-Greenstein phase function
# ─────────────────────────────────────────────────────────────────────────────

def sample_hg_cos_theta(g: float, xi: ndarray) -> ndarray:
    """
    Sample the polar scattering angle cosine from the Henyey-Greenstein CDF.

        cos θ = 1/(2g) · [1 + g² − ((1−g²)/(1−g+2gξ))²]    g ≠ 0
        cos θ = 1 − 2ξ                                         g = 0

    Parameters
    ----------
    g  : asymmetry parameter — scalar (homogeneous medium path)
    xi : uniform random variates in [0, 1), shape (N,)

    Returns
    -------
    cos_theta : shape (N,), clipped to [−1, 1]
    """
    if abs(g) < 1e-6:
        return np.clip(1.0 - 2.0 * xi, -1.0, 1.0)
    tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    cos_theta = (1.0 + g * g - tmp * tmp) / (2.0 * g)
    return np.clip(cos_theta, -1.0, 1.0)


def sample_hg_cos_theta_array(g: ndarray, xi: ndarray) -> ndarray:
    """
    Sample HG scattering cosines with per-photon asymmetry parameter.

    This is the vectorised generalisation of `sample_hg_cos_theta` for
    inhomogeneous media where each photon occupies a different layer and
    therefore has a different local g(z).

    The analytic inverse-CDF of the Henyey-Greenstein phase function is:

        cos θ = 1/(2g) · [1 + g² − ((1−g²)/(1−g+2gξ))²]   when |g| ≥ ε
        cos θ = 1 − 2ξ                                       when |g| <  ε

    Parameters
    ----------
    g  : per-photon asymmetry parameter, shape (N,)
    xi : uniform random variates in [0, 1), shape (N,)

    Returns
    -------
    cos_theta : shape (N,), clipped to [−1, 1]
    """
    out      = np.empty(g.shape, dtype=np.float64)
    iso_mask = np.abs(g) < 1e-6
    hg_mask  = ~iso_mask

    if iso_mask.any():
        out[iso_mask] = 1.0 - 2.0 * xi[iso_mask]

    if hg_mask.any():
        g_hg  = g[hg_mask]
        xi_hg = xi[hg_mask]
        tmp   = (1.0 - g_hg ** 2) / (1.0 - g_hg + 2.0 * g_hg * xi_hg)
        out[hg_mask] = (1.0 + g_hg ** 2 - tmp ** 2) / (2.0 * g_hg)

    return np.clip(out, -1.0, 1.0)


def rotate_direction(
    ux: ndarray, uy: ndarray, uz: ndarray,
    cos_sc: ndarray, sin_sc: ndarray,
    phi_sc: ndarray,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Rotate a batch of unit direction vectors (ux, uy, uz) by a local
    scattering deflection (cos_sc, sin_sc) around azimuth phi_sc.

    Uses the standard MCML frame-rotation formula with a near-z-axis
    singularity fix to avoid division by zero when |uz| → 1.

    Parameters
    ----------
    ux, uy, uz : direction cosines of current propagation direction, shape (N,)
    cos_sc     : cosine of polar scattering angle,  shape (N,)
    sin_sc     : sine  of polar scattering angle,  shape (N,)
    phi_sc     : azimuthal scattering angle (rad), shape (N,)

    Returns
    -------
    (ux_new, uy_new, uz_new) : rotated unit direction cosines
    """
    cp     = np.cos(phi_sc)
    sp     = np.sin(phi_sc)
    denom  = np.sqrt(np.maximum(1e-12, 1.0 - uz ** 2))
    near_z = np.abs(uz) > (1.0 - 1e-5)
    sgn    = np.where(uz >= 0.0, 1.0, -1.0)

    ux_new = sin_sc * (ux * uz * cp - uy * sp) / denom + ux * cos_sc
    uy_new = sin_sc * (uy * uz * cp + ux * sp) / denom + uy * cos_sc
    uz_new = -sin_sc * cp * denom + uz * cos_sc

    ux_new[near_z] = sin_sc[near_z] * cp[near_z]
    uy_new[near_z] = sin_sc[near_z] * sp[near_z] * sgn[near_z]
    uz_new[near_z] = cos_sc[near_z] * sgn[near_z]

    norm = np.sqrt(ux_new ** 2 + uy_new ** 2 + uz_new ** 2)
    norm = np.where(norm > 1e-15, norm, 1.0)
    return ux_new / norm, uy_new / norm, uz_new / norm


# ─────────────────────────────────────────────────────────────────────────────
# Photon launch geometry
# ─────────────────────────────────────────────────────────────────────────────

def sample_launch_positions(
    waist_m: float, rng: np.random.Generator, N: int
) -> tuple[ndarray, ndarray]:
    """
    Sample (x, y) launch positions from a TEM₀₀ Gaussian beam profile.

    The 1/e² intensity radius is `waist_m`.  The radial CDF of the 2-D
    Gaussian intensity profile I(r) ∝ exp(−2r²/w²) is:

        F(r) = 1 − exp(−2r²/w²)

    Inverting gives  r = w · √(−0.5 · ln ξ)  with ξ ∈ (0, 1].

    Returns
    -------
    x, y : Cartesian launch coordinates (m), shape (N,)
    """
    r   = waist_m * np.sqrt(-0.5 * np.log(np.clip(rng.uniform(size=N), 1e-15, 1.0)))
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    return r * np.cos(phi), r * np.sin(phi)


def sample_launch_directions(
    divergence_rad: float, rng: np.random.Generator, N: int
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Sample initial direction cosines uniformly within a cone of half-angle
    `divergence_rad` (solid-angle uniform sampling).

    The solid-angle element dΩ = sin θ dθ dφ integrates to a CDF that
    inverts as:

        cos θ₀ = 1 − ξ · (1 − cos θ_max)

    Returns
    -------
    ux, uy, uz : direction cosines, shape (N,), with uz > 0 (forward/downward)
    """
    cos_max = np.cos(divergence_rad)
    cos_th  = 1.0 - rng.uniform(size=N) * (1.0 - cos_max)
    sin_th  = np.sqrt(np.maximum(0.0, 1.0 - cos_th ** 2))
    phi     = rng.uniform(0.0, 2.0 * np.pi, N)
    return sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th


# ─────────────────────────────────────────────────────────────────────────────
# Receiver optics — Fresnel transmittance
# ─────────────────────────────────────────────────────────────────────────────

def fresnel_transmittance(
    cos_inc: ndarray,
    n1: float = N_WATER,   # seawater (default)
    n2: float = 1.50,      # borosilicate glass receiver window (default)
) -> ndarray:
    """
    Fresnel power transmittance T for unpolarized light at a planar n1→n2
    interface (e.g. seawater → receiver glass window).

    Uses the exact Fresnel equations for both polarisations:

        R_s = ((n1 cos θᵢ − n2 cos θₜ) / (n1 cos θᵢ + n2 cos θₜ))²
        R_p = ((n2 cos θᵢ − n1 cos θₜ) / (n2 cos θᵢ + n1 cos θₜ))²
        T   = 1 − 0.5 · (R_s + R_p)

    For angles beyond the critical angle (total internal reflection) T = 0.

    Parameters
    ----------
    cos_inc : cosine of the angle of incidence θᵢ (≥ 0), shape (N,)
    n1      : refractive index of the incident medium (default: N_WATER = 1.33)
    n2      : refractive index of the transmitted medium (default: 1.50, glass)

    Returns
    -------
    T : power transmittance ∈ [0, 1], shape (N,)

    Notes
    -----
    At normal incidence (cos_inc = 1): T = 1 − ((n1−n2)/(n1+n2))² ≈ 0.9964
    for the seawater→glass default pair — a ~0.4% correction.
    At the critical angle θ_c = arcsin(n2/n1) (only relevant when n1 > n2):
    T drops to zero.
    """
    sin_inc = np.sqrt(np.maximum(0.0, 1.0 - cos_inc ** 2))
    sin_tr  = (n1 / n2) * sin_inc          # Snell's law
    tir     = sin_tr >= 1.0                 # total internal reflection mask
    sin_tr  = np.minimum(sin_tr, 1.0)       # clamp for sqrt safety
    cos_tr  = np.sqrt(np.maximum(0.0, 1.0 - sin_tr ** 2))

    n1_ci = n1 * cos_inc
    n2_ct = n2 * cos_tr
    n2_ci = n2 * cos_inc
    n1_ct = n1 * cos_tr

    denom_s = n1_ci + n2_ct
    denom_p = n2_ci + n1_ct
    denom_s = np.where(denom_s > 1e-15, denom_s, 1.0)
    denom_p = np.where(denom_p > 1e-15, denom_p, 1.0)

    Rs = ((n1_ci - n2_ct) / denom_s) ** 2
    Rp = ((n2_ci - n1_ct) / denom_p) ** 2
    T  = 1.0 - 0.5 * (Rs + Rp)
    return np.where(tir, 0.0, T)


# ─────────────────────────────────────────────────────────────────────────────
# Time-of-flight
# ─────────────────────────────────────────────────────────────────────────────

def path_length_to_tof(path_length_m: ndarray) -> ndarray:
    """
    Convert cumulative optical path length to time-of-flight.

        t = L / v_medium = L · n_water / c₀

    Parameters
    ----------
    path_length_m : cumulative geometric path length (m), shape (N,)

    Returns
    -------
    tof_s : time of flight (s), shape (N,)

    Notes
    -----
    Uses the global C_MEDIUM = c₀/N_WATER (seawater at 530 nm, n = 1.33).
    For stratified media with slightly different layer refractive indices
    this is a first-order approximation; Δn across typical ocean layers
    is ≲ 0.01, giving a ToF error below 0.8 %.
    """
    return path_length_m / C_MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# Bio-optical model — chlorophyll → IOPs  (Case-1 waters, 530 nm)
# ─────────────────────────────────────────────────────────────────────────────
#
# Converts a chlorophyll-a concentration (mg m⁻³) into absorption a and
# scattering b at the simulation wavelength, using the standard Case-1
# relations.  c = a + b.  This is the "Step 2" that turns a chlorophyll *value*
# into optical *coefficients*; the chlorophyll-vs-depth profile (Step 1) lives in
# uowc.medium.
#
#     a(C) = a_w + A_φ · C^E_φ                       (water + phytoplankton)
#     b(C) = b_w + b_p550 · (550/λ) · C^0.62         (water + particles)
#
# References: Morel (1991); Gordon & Morel (1983); Bricaud et al. (1998);
#             Pope & Fry (1997) for pure-water absorption; Haltrin (1999).
#
# ⚠ The constants below are wavelength-specific (530 nm) reference values.
#   Replace them with your own calibrated numbers if your paper uses a different
#   source — they are isolated here precisely so the calibration is one edit.
_A_W_530   = 0.0430    # pure-seawater absorption at 530 nm   (m⁻¹)
_B_W_530   = 0.0019    # pure-seawater scattering  at 530 nm   (m⁻¹)
_A_PHI_530 = 0.0180    # chl-specific phytoplankton absorption A(530)  (m²/mg)
_E_PHI_530 = 0.650     # phytoplankton absorption exponent E(530)      (–)
_BP_550    = 0.300     # particle scattering coefficient b_p(550) per C^0.62
_LAMBDA_NM = 530.0     # simulation wavelength (nm)


def iops_from_chlorophyll(chl: ndarray) -> tuple[ndarray, ndarray]:
    """
    Case-1 inherent optical properties from chlorophyll concentration.

    Parameters
    ----------
    chl : chlorophyll-a concentration (mg m⁻³), scalar or array.  Negative
          inputs are clipped to 0.

    Returns
    -------
    (a, b) : absorption and scattering coefficients (m⁻¹), same shape as `chl`.
             The beam-attenuation coefficient is c = a + b.

    Notes
    -----
    More chlorophyll ⇒ more phytoplankton absorption *and* more particulate
    scattering ⇒ larger c (murkier water).  Constants are 530 nm specific;
    see the module-level notes to recalibrate.
    """
    chl   = np.maximum(np.asarray(chl, dtype=np.float64), 0.0)
    a_phi = _A_PHI_530 * np.power(chl, _E_PHI_530)
    a     = _A_W_530 + a_phi
    b_p   = _BP_550 * (550.0 / _LAMBDA_NM) * np.power(chl, 0.62)
    b     = _B_W_530 + b_p
    return a, b
