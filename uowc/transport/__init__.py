"""
uowc.transport
==============
Monte Carlo photon propagation engine.

Separation-of-Concern role
---------------------------
  This module owns exactly one responsibility: advancing photon packets
  through the medium until they are captured by the receiver, escape the
  domain, or are terminated by Russian roulette.  It delegates:

    physics    — all IOP / phase-function / Fresnel / weight arithmetic
    turbulence — C_n², phase-screen kicks, scintillation sampling
    medium     — depth-dependent IOP look-ups (MediumProfile interface)

  It has no awareness of Pandas, plotting, file I/O, or convergence logic.

Architecture
------------
  One shared inner chunk loop (_propagate_chunk) handles both paths:
    • Homogeneous  — medium=None, scalar c/omega/g, no Woodcock test
    • Woodcock     — medium is a MediumProfile, per-position IOPs

  The four public worker functions are thin wrappers that unpack their
  argument tuples and forward to _propagate_chunk.  ProcessPoolExecutor
  calls these workers directly (they must be module-level picklable
  functions).

Bug fixes (relative to original)
---------------------------------
  1. Implicit absorption (critical):
       apply_absorption(w, active, omega) and apply_absorption_array(w, real_idx, omega)
       now correctly mutate the original weight array via integer-array
       assignment (w[idx] *= ω).  The previous pattern update_weight(w[idx], ω)
       passed a NumPy fancy-index copy to the function; the in-place *= inside
       modified the copy only, leaving w unchanged.  Weights now decay properly
       and Russian roulette fires when weights fall below threshold.

  2. Scintillation (Woodcock path):
       σ²_R now uses the path-integrated C_n² via turbulence.path_cn2_integral()
       instead of the local C_n² at the receiver plane.  For stratified links
       this prevents assigning the wrong turbulence regime (e.g. Gamma-Gamma
       instead of log-normal) to the captured photons.

  3. Fresnel transmittance:
       fresnel_transmittance(cos_inc) is applied at the receiver interface
       (water → glass window) for all captured photons.

References
----------
  Wang, Jacques & Zheng (1995) CPC 47:131-146 — MCML algorithm
  Woodcock et al. (1965) — delta-tracking majorant method
  Andrews & Phillips (2001) — scintillation fading models
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Dict

from uowc.physics import (
    sample_step_length, sample_step_woodcock,
    accept_real_collision,
    apply_absorption, apply_absorption_array,
    russian_roulette, russian_roulette_targeted,
    sample_hg_cos_theta, sample_hg_cos_theta_array,
    rotate_direction,
    sample_launch_positions, sample_launch_directions,
    path_length_to_tof,
    fresnel_transmittance,
)
from uowc.turbulence import (
    NoTurbulence, TurbulenceProfile,
    screen_angular_variance,
    apply_angular_kick,
    sample_scintillation_fading,
)

_MAX_ITER = 8_000
_NO_TURB  = NoTurbulence()

PhotonRecord = Dict[str, ndarray]


def _empty_record() -> PhotonRecord:
    return {
        "weight":        np.array([], dtype=np.float64),
        "tof_s":         np.array([], dtype=np.float64),
        "x_m":           np.array([], dtype=np.float32),
        "y_m":           np.array([], dtype=np.float32),
        "path_length_m": np.array([], dtype=np.float32),
        "n_scatters":    np.array([], dtype=np.int32),
        "n_nulls":       np.array([], dtype=np.int32),
    }


def _concat_records(records: list[PhotonRecord]) -> PhotonRecord:
    if not records:
        return _empty_record()
    return {k: np.concatenate([r[k] for r in records])
            for k in _empty_record()}


# ─────────────────────────────────────────────────────────────────────────────
# Receiver-crossing helper  (shared by all paths)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_receiver_crossing(
    cross, idx, x, y, z, ux, uy, uz, L, s,
    n_sc, n_nl,
    link_range, rx_radius, rx_fov,
    w, alive, records,
    turbulence: TurbulenceProfile,
    rng: np.random.Generator,
) -> None:
    """
    Process photons that crossed the receiver plane during the current step.

    Responsibilities (in order):
      1. Interpolate exact position at the receiver plane.
      2. Apply aperture (radial) + FOV (angular) acceptance filter.
      3. Apply Fresnel transmittance at the water→glass interface.
      4. Apply scintillation fading (turbulence path only), using the
         path-integrated C_n² for the correct Rytov variance.
      5. Append a PhotonRecord entry for captured photons.
      6. Kill all crossing photons (hit or miss) in the alive mask.

    All crossing photons are killed regardless of hit/miss — a photon that
    passes the receiver plane without entering the aperture is considered
    lost (it has propagated beyond the link).
    """
    ci    = idx[cross]
    uz_ci = uz[ci]
    s_ci  = s[cross]

    # ── 1. Interpolate to receiver plane ──────────────────────────────────────
    safe_uz = np.where(np.abs(uz_ci) > 1e-12, uz_ci, 1e-12)
    frac    = np.clip((link_range - z[ci]) / (safe_uz * s_ci), 0.0, 1.0)

    x_rx = x[ci] + frac * s_ci * ux[ci]
    y_rx = y[ci] + frac * s_ci * uy[ci]
    L_rx = L[ci] + frac * s_ci

    # ── 2. Aperture + FOV filter ──────────────────────────────────────────────
    r_rx    = np.hypot(x_rx, y_rx)
    cos_inc = np.clip(np.abs(uz_ci), 0.0, 1.0)
    ang_rx  = np.arccos(cos_inc)
    hit     = (r_rx <= rx_radius) & (ang_rx <= rx_fov)

    if hit.any():
        hi    = ci[hit]
        w_cap = w[hi].copy()

        # ── 3. Fresnel transmittance (water → receiver glass window) ──────────
        w_cap *= fresnel_transmittance(cos_inc[hit])

        # ── 4. Scintillation fading (path-integrated Rytov variance) ─────────
        if not turbulence.is_turbulence_free():
            # path_cn2_integral gives ∫C_n²dz; divide by L for path-avg C_n²
            cn2_integral = turbulence.path_cn2_integral(link_range)
            cn2_eff      = cn2_integral / max(link_range, 1e-9)
            h_t = sample_scintillation_fading(
                np.full(hi.shape, cn2_eff), link_range, rng)
            w_cap *= h_t

        records.append({
            "weight":        w_cap.astype(np.float64),
            "tof_s":         path_length_to_tof(L_rx[hit]).astype(np.float64),
            "x_m":           x_rx[hit].astype(np.float32),
            "y_m":           y_rx[hit].astype(np.float32),
            "path_length_m": L_rx[hit].astype(np.float32),
            "n_scatters":    n_sc[hi].copy().astype(np.int32),
            "n_nulls":       n_nl[hi].copy().astype(np.int32),
        })

    # ── 5. Kill all crossing photons (hit or miss) ────────────────────────────
    alive[ci] = False


# ─────────────────────────────────────────────────────────────────────────────
# Turbulence angular-kick helper  (shared by all turbulent paths)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_turbulence_kick(
    ux:         ndarray,
    uy:         ndarray,
    uz:         ndarray,
    active:     ndarray,
    z_active:   ndarray,
    s_active:   ndarray,
    turbulence: TurbulenceProfile,
    rng:        np.random.Generator,
) -> None:
    """
    In-place per-step angular kick from turbulence phase screen.

    Computes σ²_α = screen_angular_variance(C_n²(z), s) per photon,
    draws Gaussian transverse kicks, and renormalises the direction vector.
    Applied to the `active` subset of the photon arrays.
    """
    cn2  = turbulence.C_n2(z_active)
    sig2 = np.asarray(screen_angular_variance(cn2, s_active), dtype=np.float64)
    ux[active], uy[active], uz[active] = apply_angular_kick(
        ux[active], uy[active], uz[active], sig2, rng)


# ─────────────────────────────────────────────────────────────────────────────
# Shared chunk loop  (both homogeneous and Woodcock paths)
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_chunk(
    N:   int,
    rng: np.random.Generator,
    *,
    # ── IOP source ────────────────────────────────────────────────────────────
    # Homogeneous path: medium=None; pass scalar c, omega, g.
    # Woodcock path:    medium is a MediumProfile; c_max = medium.c_max.
    #                   The scalar c / omega / g arguments are unused.
    medium,           # MediumProfile | None
    c:     float,     # beam-attenuation coefficient (homogeneous path)
    omega: float,     # single-scattering albedo     (homogeneous path)
    g:     float,     # HG asymmetry parameter       (homogeneous path)
    c_max: float,     # step-sampling majorant (= c for hom.; = medium.c_max for Woodcock)
    # ── Geometry ──────────────────────────────────────────────────────────────
    beam_div:   float,
    beam_waist: float,
    rx_radius:  float,
    rx_fov:     float,
    link_range: float,
    # ── Variance reduction ────────────────────────────────────────────────────
    weight_threshold: float,
    roulette_m:       int,
    # ── Turbulence ────────────────────────────────────────────────────────────
    turbulence: TurbulenceProfile,
    # ── Output accumulator (mutated in-place) ─────────────────────────────────
    records: list,
) -> None:
    """
    Propagate N photons and append captured PhotonRecord dicts to `records`.

    Homogeneous path  (medium is None)
    ------------------------------------
      Steps are sampled with c directly.  Every step is a real interaction:
      all active photons undergo implicit absorption (w *= omega) and scatter
      (Henyey-Greenstein with scalar g).

    Woodcock path  (medium is a MediumProfile)
    -------------------------------------------
      Steps are sampled with c_max (the Woodcock majorant).  After each step
      the collision is accepted as real with probability c(z)/c_max; null
      collisions advance the photon without weight change or scatter.  Weight
      update and scattering use per-photon IOPs from the medium object.

    Turbulence
    ----------
      When turbulence.is_turbulence_free() is False, a Gaussian angular kick
      is applied to every active photon at every step (real and null).
      Scintillation fading is applied in _handle_receiver_crossing.

    Step-by-step loop structure (shared by both paths)
    ---------------------------------------------------
      1. Sample free-flight step (sample_step_length or sample_step_woodcock)
      2. Advance positions tentatively
      3. Detect and process receiver-plane crossings
      4. Commit positions for surviving photons; apply domain-boundary cull
      5. Apply turbulence angular kick (if turbulence active)
      6a. [Woodcock] Acceptance test → real_idx; weight update; roulette
      6b. [Hom.]     Weight update for all active; roulette
      7. Scatter surviving photons via HG phase function
    """
    use_woodcock = medium is not None
    do_turb      = not turbulence.is_turbulence_free()

    x, y   = sample_launch_positions(beam_waist, rng, N)
    z      = np.zeros(N)
    ux, uy, uz = sample_launch_directions(beam_div, rng, N)
    w      = np.ones(N);  L = np.zeros(N)
    n_sc   = np.zeros(N, dtype=np.int32)
    n_nl   = np.zeros(N, dtype=np.int32)
    alive  = np.ones(N, dtype=bool)

    for _ in range(_MAX_ITER):
        if not alive.any():
            break

        idx = np.where(alive)[0]
        n   = idx.size

        # ── 1. Free-flight step ───────────────────────────────────────────────
        s = (sample_step_woodcock(c_max, rng.uniform(size=n))
             if use_woodcock
             else sample_step_length(c, rng.uniform(size=n)))

        # ── 2. Tentative position advance ─────────────────────────────────────
        xn = x[idx] + s * ux[idx]
        yn = y[idx] + s * uy[idx]
        zn = z[idx] + s * uz[idx]

        # ── 3. Receiver-plane crossing ────────────────────────────────────────
        cross = (z[idx] < link_range) & (zn >= link_range)
        if cross.any():
            _handle_receiver_crossing(
                cross, idx, x, y, z, ux, uy, uz, L, s, n_sc, n_nl,
                link_range, rx_radius, rx_fov, w, alive, records,
                turbulence, rng)

        # ── 4. Commit positions; domain-boundary cull ─────────────────────────
        alive_mask = alive.copy()
        alive_mask[idx[cross]] = False
        si = np.where(alive_mask)[0]
        if si.size == 0:
            continue
        sm = np.isin(idx, si)
        x[si] = xn[sm]; y[si] = yn[sm]; z[si] = zn[sm]; L[si] += s[sm]

        out   = (z[si] < 0) | (z[si] > link_range * 3) | (np.hypot(x[si], y[si]) > 10)
        alive[si[out]] = False
        active = si[~out]
        if active.size == 0:
            continue
        s_active = s[sm][~out]

        # ── 5. Turbulence angular kick (real and null steps) ──────────────────
        if do_turb:
            _apply_turbulence_kick(
                ux, uy, uz, active, z[active], s_active, turbulence, rng)

        # ── 6. Interaction: weight update, roulette, scatter ──────────────────
        if use_woodcock:
            # 6a. Woodcock: accept real or null collision
            c_local   = medium.attenuation(z[active])
            real_mask = accept_real_collision(c_local, c_max, rng.uniform(size=active.size))
            real_idx  = active[real_mask]
            n_nl[active[~real_mask]] += 1
            if real_idx.size == 0:
                continue

            apply_absorption_array(w, real_idx, medium.albedo(z[real_idx]))
            russian_roulette_targeted(w, alive, real_idx,
                                      rng.uniform(size=real_idx.size),
                                      weight_threshold, roulette_m)
            scatter_idx = real_idx[alive[real_idx]]
            if scatter_idx.size == 0:
                continue
            g_local = medium.asymmetry(z[scatter_idx])
            cos_sc  = sample_hg_cos_theta_array(g_local, rng.uniform(size=scatter_idx.size))

        else:
            # 6b. Homogeneous: every step is a real absorption + scatter event
            apply_absorption(w, active, omega)
            russian_roulette(w, alive, rng.uniform(size=N), weight_threshold, roulette_m)
            scatter_idx = np.where(alive)[0]
            if scatter_idx.size == 0:
                continue
            cos_sc = sample_hg_cos_theta(g, rng.uniform(size=scatter_idx.size))

        # ── 7. Scatter: rotate direction by HG phase-function angle ───────────
        sin_sc = np.sqrt(np.maximum(0.0, 1.0 - cos_sc ** 2))
        phi_sc = rng.uniform(0.0, 2.0 * np.pi, scatter_idx.size)
        ux[scatter_idx], uy[scatter_idx], uz[scatter_idx] = rotate_direction(
            ux[scatter_idx], uy[scatter_idx], uz[scatter_idx], cos_sc, sin_sc, phi_sc)
        n_sc[scatter_idx] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Public worker functions  (one per (medium-type × turbulence) combination)
# Each is a thin wrapper: unpack args → create RNG → dispatch to _propagate_chunk
# These must be module-level functions for ProcessPoolExecutor pickling.
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch(args: tuple) -> PhotonRecord:
    """Worker: homogeneous medium, no turbulence."""
    (n_photons, seed_state,
     c, b, g, omega,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))
    records: list[PhotonRecord] = []
    remaining = n_photons
    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        _propagate_chunk(
            N, rng,
            medium=None, c=c, omega=omega, g=g, c_max=c,
            beam_div=beam_div, beam_waist=beam_waist,
            rx_radius=rx_radius, rx_fov=rx_fov, link_range=link_range,
            weight_threshold=weight_threshold, roulette_m=roulette_m,
            turbulence=_NO_TURB, records=records,
        )
    return _concat_records(records)


def propagate_batch_inhomogeneous(args: tuple) -> PhotonRecord:
    """Worker: Woodcock delta-tracking, no turbulence."""
    (n_photons, seed_state, medium,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))
    records: list[PhotonRecord] = []
    remaining = n_photons
    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        _propagate_chunk(
            N, rng,
            medium=medium, c=0.0, omega=0.0, g=0.0, c_max=medium.c_max,
            beam_div=beam_div, beam_waist=beam_waist,
            rx_radius=rx_radius, rx_fov=rx_fov, link_range=link_range,
            weight_threshold=weight_threshold, roulette_m=roulette_m,
            turbulence=_NO_TURB, records=records,
        )
    return _concat_records(records)


def propagate_batch_turbulent(args: tuple) -> PhotonRecord:
    """Worker: homogeneous medium + turbulence (phase-screen kicks + scintillation)."""
    (n_photons, seed_state,
     c, b, g, omega,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size,
     turbulence) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))
    records: list[PhotonRecord] = []
    remaining = n_photons
    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        _propagate_chunk(
            N, rng,
            medium=None, c=c, omega=omega, g=g, c_max=c,
            beam_div=beam_div, beam_waist=beam_waist,
            rx_radius=rx_radius, rx_fov=rx_fov, link_range=link_range,
            weight_threshold=weight_threshold, roulette_m=roulette_m,
            turbulence=turbulence, records=records,
        )
    return _concat_records(records)


def propagate_batch_inhomogeneous_turbulent(args: tuple) -> PhotonRecord:
    """Worker: Woodcock delta-tracking + turbulence."""
    (n_photons, seed_state, medium,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size,
     turbulence) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))
    records: list[PhotonRecord] = []
    remaining = n_photons
    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        _propagate_chunk(
            N, rng,
            medium=medium, c=0.0, omega=0.0, g=0.0, c_max=medium.c_max,
            beam_div=beam_div, beam_waist=beam_waist,
            rx_radius=rx_radius, rx_fov=rx_fov, link_range=link_range,
            weight_threshold=weight_threshold, roulette_m=roulette_m,
            turbulence=turbulence, records=records,
        )
    return _concat_records(records)
