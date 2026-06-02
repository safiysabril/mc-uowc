"""
uowc.metrics
============
Channel characterisation metrics derived from RunResult.

Accepts RunResult directly — no need to manually unpack weights/times.
n_launched is read from RunResult.n_launched for correct normalisation.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from numpy import ndarray
from uowc.config import SimConfig
from uowc.physics import beer_lambert_power_dB


def received_power_dB(weights: ndarray, n_launched: int) -> float:
    if weights.size == 0 or n_launched == 0:
        return -np.inf
    return float(10.0 * np.log10(np.maximum(weights.sum() / n_launched, 1e-300)))


def compute_cir(weights, times, dt, n_bins):
    edges  = np.linspace(0.0, dt * n_bins, n_bins + 1)
    h, _   = np.histogram(times, bins=edges, weights=weights)
    t_axis = 0.5 * (edges[:-1] + edges[1:])
    return t_axis, h / (h.sum() + 1e-30)


def rms_delay_spread(weights, times) -> float:
    if weights.size < 2 or weights.sum() == 0:
        return float("nan")
    w_sum = weights.sum()
    tau_m = (weights * times).sum() / w_sum
    return float(np.sqrt((weights * (times - tau_m) ** 2).sum() / w_sum))


def frequency_response(h_norm, dt, pad_factor=8):
    N_pad = len(h_norm) * pad_factor
    H     = np.fft.rfft(h_norm, n=N_pad)
    H_mag = np.abs(H)
    dc    = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    return np.fft.rfftfreq(N_pad, d=dt), H_mag / dc


def bandwidth_3dB(freqs, H_norm) -> float:
    idx = np.where(H_norm < (1.0 / np.sqrt(2.0)))[0]
    return float(freqs[idx[0]]) if idx.size > 0 else float(freqs[-1])


def compute_all_metrics(result, cfg: SimConfig, c: float, link_range: float) -> Dict:
    """
    Compute all channel metrics from a RunResult.

    Parameters
    ----------
    result     : RunResult  (has .weights, .times, .n_launched, .record)
    cfg        : SimConfig
    c          : reference attenuation for Beer-Lambert (m⁻¹)
    link_range : link length (m)
    """
    weights    = result.weights
    times      = result.times
    n_launched = result.n_launched

    p_dB  = received_power_dB(weights, n_launched)
    p_bl  = beer_lambert_power_dB(c, link_range)
    ds    = rms_delay_spread(weights, times) if weights.size > 1 else float("nan")
    t_ax, h   = compute_cir(weights, times, cfg.dt_bin_s, cfg.n_time_bins)
    freqs, Hn = frequency_response(h, cfg.dt_bin_s)
    bw        = bandwidth_3dB(freqs, Hn)

    return {
        "power_dB":        p_dB,
        "beer_lambert_dB": p_bl,
        "delay_spread_s":  ds,
        "bandwidth_hz":    bw,
        "t_axis":          t_ax,
        "cir":             h,
        "freqs":           freqs,
        "fr":              Hn,
        "n_launched":      n_launched,
        "n_captured":      weights.size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BER and coherence time
# ─────────────────────────────────────────────────────────────────────────────

from scipy.special import erfc as _erfc
import warnings as _warnings


def compute_ber_ook(
    snr_electrical: float,
    sigma_R2:       float = 0.0,
) -> float:
    """
    Bit error rate for OOK (On-Off Keying) with optional turbulence.

    Without turbulence (σ²_R = 0):
        BER = 0.5 · erfc(√(SNR_e / 2))

    With log-normal fading (weak turbulence, σ²_R < 0.3):
        BER ≈ 0.5 · erfc(√(SNR_e · exp(−2σ²_X) / 2))
        where σ²_X = 0.49·σ²_R / (1 + 0.56·σ²_R^(6/5))^(7/6)

    With Gamma-Gamma (moderate, σ²_R ≥ 0.3):
        Integral approximation via moment-matched log-normal equivalent.

    Parameters
    ----------
    snr_electrical : mean electrical SNR = (P_r / σ_noise)²
    sigma_R2       : Rytov variance σ²_R  (0 = no turbulence)

    Returns
    -------
    BER in [0, 0.5]
    """
    if snr_electrical <= 0:
        return 0.5

    if sigma_R2 <= 0.0:
        # AWGN only
        return 0.5 * float(_erfc(np.sqrt(snr_electrical / 2.0)))

    if sigma_R2 < 0.3:
        # Weak turbulence — log-normal fading penalty
        sX2  = 0.49 * sigma_R2 / ((1.0 + 0.56 * sigma_R2**(6/5))**(7/6))
        snr_eff = snr_electrical * np.exp(-2.0 * sX2)
        return 0.5 * float(_erfc(np.sqrt(max(snr_eff, 0.0) / 2.0)))

    # Moderate/strong — moment-matched Gamma-Gamma via equivalent log-normal
    # Using the approximation from Nistazakis et al. (2009):
    #   BER ≈ 0.5 · erfc(√(SNR_e / (2·(1 + σ²_I))))
    # where σ²_I = 1/α + 1/β + 1/(αβ)  (scintillation index)
    sX = 0.49 * sigma_R2 / ((1.0 + 0.56 * sigma_R2**(6/5))**(7/6))
    sY = 0.51 * sigma_R2 / ((1.0 + 0.69 * sigma_R2**(6/5))**(5/6))
    alpha = 1.0 / max(np.exp(sX) - 1.0, 1e-10)
    beta  = 1.0 / max(np.exp(sY) - 1.0, 1e-10)
    sigma_I2 = 1.0/alpha + 1.0/beta + 1.0/(alpha * beta)
    snr_eff  = snr_electrical / (1.0 + sigma_I2)
    return 0.5 * float(_erfc(np.sqrt(max(snr_eff, 0.0) / 2.0)))


def compute_snr(
    weights:      "ndarray",
    n_launched:   int,
    noise_power:  float = 1e-7,
    responsivity: float = 0.5,
) -> float:
    """
    Estimate electrical SNR from Monte Carlo output.

        SNR_e = (R · P_r)² / σ²_noise

    where R is photodetector responsivity [A/W] and P_r = Σwᵢ/N_launched.

    Parameters
    ----------
    weights      : captured photon weights
    n_launched   : total launched photons (adaptive denominator)
    noise_power  : receiver noise variance σ²_noise (A²) — depends on
                   bandwidth, temperature, dark current.  Default 1e-7 A².
    responsivity : photodetector responsivity R (A/W). Default 0.5 A/W.
    """
    if weights.size == 0 or n_launched == 0:
        return 0.0
    P_r = float(weights.sum()) / n_launched
    return (responsivity * P_r) ** 2 / noise_power


def compute_coherence_time(
    result,
    medium,
    link_range: float,
) -> float:
    """
    Channel coherence time τ_c (s) via Taylor frozen-turbulence hypothesis.

        τ_c = r₀ / v̄_⊥

    Parameters
    ----------
    result     : RunResult (not directly used, provided for API consistency)
    medium     : CoupledOceanMedium or TurbulenceProfile with
                 .channel_coherence_time(link_range) or
                 .fried_radius(link_range) + .v_current(z)
    link_range : link length (m)

    Returns
    -------
    τ_c in seconds, or np.inf if medium has no turbulence.
    """
    # CoupledOceanMedium has a direct method
    if hasattr(medium, "channel_coherence_time"):
        return float(medium.channel_coherence_time(link_range))

    # Fallback for bare TurbulenceProfile
    from uowc.turbulence import fried_parameter, coherence_time
    z_arr = np.linspace(0, link_range, max(int(link_range / 0.5), 10))
    dz    = z_arr[1] - z_arr[0]
    cn2   = medium.C_n2(z_arr)
    r0    = fried_parameter(float((cn2 * dz).sum()))
    v_mid = float(medium.v_current(np.array([link_range / 2.0]))[0])
    return coherence_time(r0, v_mid)