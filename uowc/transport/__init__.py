"""
uowc.transport
==============
Monte Carlo photon propagation engine.

Four workers
------------
  propagate_batch                         — homogeneous, no turbulence
  propagate_batch_inhomogeneous           — Woodcock, no turbulence
  propagate_batch_turbulent               — homogeneous + turbulence
  propagate_batch_inhomogeneous_turbulent — Woodcock + turbulence

Turbulence is applied via CoupledOceanMedium or any TurbulenceProfile:
  1. Per-step phase-screen angular kicks  (beam wander + spreading)
  2. Scintillation fading at receiver crossing  (intensity fluctuation)
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Dict

from uowc.physics import (
    sample_step_length, sample_step_woodcock,
    accept_real_collision,
    update_weight, update_weight_array,
    russian_roulette, russian_roulette_targeted,
    sample_hg_cos_theta, sample_hg_cos_theta_array,
    rotate_direction,
    sample_launch_positions, sample_launch_directions,
    path_length_to_tof,
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
# Shared receiver-crossing helper
# ─────────────────────────────────────────────────────────────────────────────

def _handle_receiver_crossing(
    cross, idx, x, y, z, ux, uy, uz, L, s,
    n_sc, n_nl,
    link_range, rx_radius, rx_fov,
    w, alive, records,
    turbulence: TurbulenceProfile,
    rng: np.random.Generator,
) -> None:
    """Aperture+FOV filter, scintillation fading, record append."""
    ci    = idx[cross]
    uz_ci = uz[ci]
    s_ci  = s[cross]

    safe_uz = np.where(np.abs(uz_ci) > 1e-12, uz_ci, 1e-12)
    frac    = np.clip((link_range - z[ci]) / (safe_uz * s_ci), 0.0, 1.0)

    x_rx = x[ci] + frac * s_ci * ux[ci]
    y_rx = y[ci] + frac * s_ci * uy[ci]
    L_rx = L[ci] + frac * s_ci

    r_rx    = np.hypot(x_rx, y_rx)
    cos_inc = np.clip(np.abs(uz_ci), 0.0, 1.0)
    ang_rx  = np.arccos(cos_inc)

    hit = (r_rx <= rx_radius) & (ang_rx <= rx_fov)
    if hit.any():
        hi    = ci[hit]
        w_cap = w[hi].copy()

        # ── scintillation fading ───────────────────────────────────────────
        if not turbulence.is_turbulence_free():
            cn2_rx = turbulence.C_n2(np.full(hi.shape, link_range))
            h_t    = sample_scintillation_fading(cn2_rx, link_range, rng)
            w_cap  = w_cap * h_t

        records.append({
            "weight":        w_cap.astype(np.float64),
            "tof_s":         path_length_to_tof(L_rx[hit]).astype(np.float64),
            "x_m":           x_rx[hit].astype(np.float32),
            "y_m":           y_rx[hit].astype(np.float32),
            "path_length_m": L_rx[hit].astype(np.float32),
            "n_scatters":    n_sc[hi].copy().astype(np.int32),
            "n_nulls":       n_nl[hi].copy().astype(np.int32),
        })

    alive[ci] = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared turbulence-kick helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_turbulence_kick(
    ux:          ndarray,
    uy:          ndarray,
    uz:          ndarray,
    active:      ndarray,
    z_active:    ndarray,
    s_active:    ndarray,
    turbulence:  TurbulenceProfile,
    rng:         np.random.Generator,
) -> None:
    """
    In-place per-step angular kick from turbulence phase screen.

    Computes σ²_α = screen_angular_variance(C_n²(z), s, l0) per photon,
    then draws Gaussian transverse kicks and renormalises.  Applied to the
    `active` subset of photon arrays.
    """
    cn2    = turbulence.C_n2(z_active)
    sig2   = np.asarray(
        screen_angular_variance(cn2, s_active), dtype=np.float64
    )
    ux[active], uy[active], uz[active] = apply_angular_kick(
        ux[active], uy[active], uz[active], sig2, rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker 1: Homogeneous, no turbulence
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch(args: tuple) -> PhotonRecord:
    """Homogeneous medium — original algorithm, no turbulence overhead."""
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
        x, y   = sample_launch_positions(beam_waist, rng, N)
        z      = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w      = np.ones(N); L = np.zeros(N)
        n_sc   = np.zeros(N, dtype=np.int32)
        n_nl   = np.zeros(N, dtype=np.int32)
        alive  = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any(): break
            idx = np.where(alive)[0]; n = idx.size
            s   = sample_step_length(c, rng.uniform(size=n))

            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s, n_sc, n_nl,
                    link_range, rx_radius, rx_fov, w, alive, records,
                    _NO_TURB, rng)

            alive_mask = alive.copy(); alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0: continue
            sm = np.isin(idx, si)
            x[si] = xn[sm]; y[si] = yn[sm]; z[si] = zn[sm]; L[si] += s[sm]
            out = (z[si] < 0) | (z[si] > link_range * 3) | (np.hypot(x[si], y[si]) > 10)
            alive[si[out]] = False; active = si[~out]
            if active.size == 0: continue

            update_weight(w[active], omega)
            russian_roulette(w, alive, rng.uniform(size=N), weight_threshold, roulette_m)
            active = np.where(alive)[0]
            if active.size == 0: continue

            cos_sc = sample_hg_cos_theta(g, rng.uniform(size=active.size))
            sin_sc = np.sqrt(np.maximum(0.0, 1 - cos_sc**2))
            phi_sc = rng.uniform(0.0, 2*np.pi, active.size)
            ux[active], uy[active], uz[active] = rotate_direction(
                ux[active], uy[active], uz[active], cos_sc, sin_sc, phi_sc)
            n_sc[active] += 1

    return _concat_records(records)


# ─────────────────────────────────────────────────────────────────────────────
# Worker 2: Woodcock (inhomogeneous), no turbulence
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch_inhomogeneous(args: tuple) -> PhotonRecord:
    """Woodcock delta-tracking, no turbulence."""
    (n_photons, seed_state, medium,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size) = args

    rng   = np.random.default_rng(np.random.PCG64(seed_state))
    c_max = medium.c_max
    records: list[PhotonRecord] = []
    remaining = n_photons

    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        x, y   = sample_launch_positions(beam_waist, rng, N)
        z      = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w      = np.ones(N); L = np.zeros(N)
        n_sc   = np.zeros(N, dtype=np.int32)
        n_nl   = np.zeros(N, dtype=np.int32)
        alive  = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any(): break
            idx = np.where(alive)[0]; n = idx.size
            s   = sample_step_woodcock(c_max, rng.uniform(size=n))

            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s, n_sc, n_nl,
                    link_range, rx_radius, rx_fov, w, alive, records,
                    _NO_TURB, rng)

            alive_mask = alive.copy(); alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0: continue
            sm = np.isin(idx, si)
            x[si] = xn[sm]; y[si] = yn[sm]; z[si] = zn[sm]; L[si] += s[sm]
            out = (z[si] < 0) | (z[si] > link_range * 3) | (np.hypot(x[si], y[si]) > 10)
            alive[si[out]] = False; active = si[~out]
            if active.size == 0: continue

            c_local   = medium.attenuation(z[active])
            real_mask = accept_real_collision(c_local, c_max, rng.uniform(size=active.size))
            real_idx  = active[real_mask]; n_nl[active[~real_mask]] += 1
            if real_idx.size == 0: continue

            update_weight_array(w[real_idx], medium.albedo(z[real_idx]))
            russian_roulette_targeted(w, alive, real_idx,
                                      rng.uniform(size=real_idx.size),
                                      weight_threshold, roulette_m)
            scatter_idx = real_idx[alive[real_idx]]
            if scatter_idx.size == 0: continue
            g_local = medium.asymmetry(z[scatter_idx])
            cos_sc  = sample_hg_cos_theta_array(g_local, rng.uniform(size=scatter_idx.size))
            sin_sc  = np.sqrt(np.maximum(0.0, 1 - cos_sc**2))
            phi_sc  = rng.uniform(0.0, 2*np.pi, scatter_idx.size)
            ux[scatter_idx], uy[scatter_idx], uz[scatter_idx] = rotate_direction(
                ux[scatter_idx], uy[scatter_idx], uz[scatter_idx], cos_sc, sin_sc, phi_sc)
            n_sc[scatter_idx] += 1

    return _concat_records(records)


# ─────────────────────────────────────────────────────────────────────────────
# Worker 3: Homogeneous + turbulence (phase-screen kicks + scintillation)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch_turbulent(args: tuple) -> PhotonRecord:
    """
    Homogeneous medium with turbulence.

    At every step: σ²_α(z,s) computed from C_n²(z) and step length s;
    random angular kick applied to active photons.  At receiver crossing:
    scintillation fading h_t multiplied into weight.
    """
    (n_photons, seed_state,
     c, b, g, omega,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size,
     turbulence) = args

    rng       = np.random.default_rng(np.random.PCG64(seed_state))
    do_turb   = not turbulence.is_turbulence_free()
    records: list[PhotonRecord] = []
    remaining = n_photons

    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        x, y   = sample_launch_positions(beam_waist, rng, N)
        z      = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w      = np.ones(N); L = np.zeros(N)
        n_sc   = np.zeros(N, dtype=np.int32)
        n_nl   = np.zeros(N, dtype=np.int32)
        alive  = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any(): break
            idx = np.where(alive)[0]; n = idx.size
            s   = sample_step_length(c, rng.uniform(size=n))

            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s, n_sc, n_nl,
                    link_range, rx_radius, rx_fov, w, alive, records,
                    turbulence, rng)

            alive_mask = alive.copy(); alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0: continue
            sm = np.isin(idx, si)
            x[si] = xn[sm]; y[si] = yn[sm]; z[si] = zn[sm]; L[si] += s[sm]
            out = (z[si] < 0) | (z[si] > link_range * 3) | (np.hypot(x[si], y[si]) > 10)
            alive[si[out]] = False; active = si[~out]
            if active.size == 0: continue

            s_active = s[sm][~out]

            # ── phase-screen angular kick ──────────────────────────────────
            if do_turb:
                _apply_turbulence_kick(
                    ux, uy, uz, active, z[active], s_active, turbulence, rng)

            update_weight(w[active], omega)
            russian_roulette(w, alive, rng.uniform(size=N), weight_threshold, roulette_m)
            active = np.where(alive)[0]
            if active.size == 0: continue

            cos_sc = sample_hg_cos_theta(g, rng.uniform(size=active.size))
            sin_sc = np.sqrt(np.maximum(0.0, 1 - cos_sc**2))
            phi_sc = rng.uniform(0.0, 2*np.pi, active.size)
            ux[active], uy[active], uz[active] = rotate_direction(
                ux[active], uy[active], uz[active], cos_sc, sin_sc, phi_sc)
            n_sc[active] += 1

    return _concat_records(records)


# ─────────────────────────────────────────────────────────────────────────────
# Worker 4: Woodcock (inhomogeneous) + turbulence
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch_inhomogeneous_turbulent(args: tuple) -> PhotonRecord:
    """
    Woodcock delta-tracking with phase-screen angular kicks and
    scintillation fading.  Designed for CoupledOceanMedium which provides
    both IOP and turbulence lookups from the same layer grid.
    """
    (n_photons, seed_state, medium,
     beam_div, beam_waist,
     rx_radius, rx_fov, link_range,
     weight_threshold, roulette_m, chunk_size,
     turbulence) = args

    rng     = np.random.default_rng(np.random.PCG64(seed_state))
    c_max   = medium.c_max
    do_turb = not turbulence.is_turbulence_free()
    records: list[PhotonRecord] = []
    remaining = n_photons

    while remaining > 0:
        N = min(chunk_size, remaining); remaining -= N
        x, y   = sample_launch_positions(beam_waist, rng, N)
        z      = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w      = np.ones(N); L = np.zeros(N)
        n_sc   = np.zeros(N, dtype=np.int32)
        n_nl   = np.zeros(N, dtype=np.int32)
        alive  = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any(): break
            idx = np.where(alive)[0]; n = idx.size
            s   = sample_step_woodcock(c_max, rng.uniform(size=n))

            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s, n_sc, n_nl,
                    link_range, rx_radius, rx_fov, w, alive, records,
                    turbulence, rng)

            alive_mask = alive.copy(); alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0: continue
            sm = np.isin(idx, si)
            x[si] = xn[sm]; y[si] = yn[sm]; z[si] = zn[sm]; L[si] += s[sm]
            out = (z[si] < 0) | (z[si] > link_range * 3) | (np.hypot(x[si], y[si]) > 10)
            alive[si[out]] = False; active = si[~out]
            if active.size == 0: continue

            s_active = s[sm][~out]

            # ── phase-screen kick (every step, real and null) ──────────────
            if do_turb:
                _apply_turbulence_kick(
                    ux, uy, uz, active, z[active], s_active, turbulence, rng)

            c_local   = medium.attenuation(z[active])
            real_mask = accept_real_collision(c_local, c_max, rng.uniform(size=active.size))
            real_idx  = active[real_mask]; n_nl[active[~real_mask]] += 1
            if real_idx.size == 0: continue

            update_weight_array(w[real_idx], medium.albedo(z[real_idx]))
            russian_roulette_targeted(w, alive, real_idx,
                                      rng.uniform(size=real_idx.size),
                                      weight_threshold, roulette_m)
            scatter_idx = real_idx[alive[real_idx]]
            if scatter_idx.size == 0: continue
            g_local = medium.asymmetry(z[scatter_idx])
            cos_sc  = sample_hg_cos_theta_array(g_local, rng.uniform(size=scatter_idx.size))
            sin_sc  = np.sqrt(np.maximum(0.0, 1 - cos_sc**2))
            phi_sc  = rng.uniform(0.0, 2*np.pi, scatter_idx.size)
            ux[scatter_idx], uy[scatter_idx], uz[scatter_idx] = rotate_direction(
                ux[scatter_idx], uy[scatter_idx], uz[scatter_idx], cos_sc, sin_sc, phi_sc)
            n_sc[scatter_idx] += 1

    return _concat_records(records)