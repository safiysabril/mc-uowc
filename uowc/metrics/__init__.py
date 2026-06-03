"""
uowc.metrics
============
Channel characterisation metrics derived from RunResult.

Accepts RunResult directly — no need to manually unpack weights/times.
n_launched is read from RunResult.n_launched for correct normalisation.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d
from uowc.config import SimConfig
from uowc.physics import beer_lambert_power_dB


def received_power_dB(weights: ndarray, n_launched: int) -> float:
    if weights.size == 0 or n_launched == 0:
        return -np.inf
    return float(10.0 * np.log10(np.maximum(weights.sum() / n_launched, 1e-300)))


def compute_cir(weights, times, dt, n_bins, *, min_photons_per_bin: int = 5):
    """
    Build a normalised CIR histogram of excess delay.

    tof_s stored in PhotonRecord is the *absolute* time of flight
    (path_length / C_medium).  For a 25 m link the ballistic arrival is
    ~111 ns — well outside a fixed 30 ns window.  Subtracting the first
    arrival time converts to excess delay so the CIR always starts at 0
    regardless of link range.

    Bin count also adapts when captured photons are too few for the
    requested resolution: n_bins is reduced so each bin holds ≥
    min_photons_per_bin photons on average.

    delay_spread and received_power are computed from raw (weights, times)
    pairs in their own functions and are NOT affected by this binning.
    """
    if times.size == 0:
        edges  = np.linspace(0.0, dt * n_bins, n_bins + 1)
        h, _   = np.histogram(times, bins=edges, weights=weights)
        t_axis = 0.5 * (edges[:-1] + edges[1:])
        return t_axis, h / (h.sum() + 1e-30)

    # Convert absolute TOF → excess delay (ballistic photon defines t = 0)
    excess = times - times.min()

    t_span = float(excess.max()) * 1.05   # 5% tail padding
    t_span = max(t_span, dt * 10)         # at least 10 bins wide

    # Statistics limit: enough photons per bin
    n_stat = max(times.size // max(min_photons_per_bin, 1), 10)
    # Resolution limit: keep dt if we have photons to fill it
    n_phys = max(int(np.ceil(t_span / dt)), 10)
    # Use whichever is tighter, capped at the requested maximum
    n_use  = min(n_stat, n_phys, n_bins)

    edges  = np.linspace(0.0, t_span, n_use + 1)
    h, _   = np.histogram(excess, bins=edges, weights=weights)
    t_axis = 0.5 * (edges[:-1] + edges[1:])
    return t_axis, h / (h.sum() + 1e-30)


def _weighted_quantile(values: ndarray, weights: ndarray, q: float) -> float:
    """Weighted quantile: smallest value whose cumulative weight fraction ≥ q."""
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    v     = values[order]
    cw    = np.cumsum(weights[order])
    if cw[-1] <= 0.0:
        return float(v[-1])
    return float(np.interp(q, cw / cw[-1], v))


def compute_cir_kde(
    weights:       ndarray,
    times:         ndarray,
    *,
    n_grid:        int   = 512,
    bw_scale:      float = 1.0,
    tail_quantile: float = 0.995,
    min_bw_s:      float = 1e-12,
) -> Tuple[ndarray, ndarray]:
    """
    Smooth channel impulse response h(τ) via a weighted Gaussian kernel
    density estimate (KDE) of the excess-delay distribution.

    Why KDE instead of a histogram?
    -------------------------------
    A histogram of N captured photons into K bins carries ~Poisson(N/K)
    counts per bin.  At the modest photon counts typical of long-range or
    turbid UOWC links this produces a jagged, spiky CIR that is unfit for
    publication — and FFT-ing that spiky CIR yields an equally noisy
    frequency response.  A weighted Gaussian KDE turns the same photons into
    a smooth, continuous h(τ) — the natural estimator of a continuous impulse
    response — and degrades gracefully as the photon count falls.

    Outlier-robust bandwidth and time window
    -----------------------------------------
    A handful of heavily-scattered photons can arrive far later than the bulk
    of the energy.  If the kernel bandwidth and the plotted time window were
    driven by the *maximum* delay, those rare outliers would (a) inflate the
    bandwidth and over-smooth the sharp ballistic peak, and (b) stretch the
    x-axis so the peak collapses into the left edge — exactly the failure seen
    for clear water at short range.  Both quantities are therefore made robust:

      • Bandwidth — Silverman's robust rule of thumb with the Kish effective
        sample size N_eff = (Σwᵢ)²/Σwᵢ² for weighted data::

            b = bw_scale · 0.9 · min(σ_τ, IQR/1.349) · N_eff^(−1/5)

        Using ``min(σ, IQR/1.349)`` makes the smoothing track the *core* spread
        of the arrivals, immune to a few far-tail photons.

      • Time window — the grid runs from 0 to a weighted high-quantile of the
        delay energy (``tail_quantile``, default 99.5 %) plus a few bandwidths,
        instead of to the maximum delay.  The window then auto-adapts: tight
        for a near-ballistic channel (peak clearly visible) and wide for a
        genuinely dispersive one (full multipath tail shown).  Photons beyond
        the window — the excluded <0.5 % outliers — are dropped from the
        estimate so they create no edge artefact.

    ``bw_scale`` lets the caller trade smoothness against fidelity.

    Implementation (binned KDE)
    ---------------------------
    The estimate is a fine weighted histogram convolved with a Gaussian —
    O(N + n_grid) rather than the O(N·n_grid) of a direct kernel sum — so it
    stays cheap even for millions of captured photons.  The convolution uses
    ``mode="reflect"`` at τ = 0, the reflection boundary correction for a
    one-sided density: it stops the kernel leaking energy to τ < 0, keeps the
    leading (ballistic) edge sharp, and conserves energy.

    delay_spread and received_power are computed from the raw (weights, times)
    in their own functions and are NOT affected by this smoothing or windowing.

    Returns
    -------
    t_axis : excess-delay grid (s), shape (n_grid,), starting at 0
    h      : peak-normalised CIR, shape (n_grid,)
    """
    if times.size == 0 or float(weights.sum()) <= 0.0:
        return np.linspace(0.0, 1e-9, n_grid), np.zeros(n_grid)

    excess = np.asarray(times, dtype=np.float64)
    excess = excess - excess.min()
    w      = np.asarray(weights, dtype=np.float64)
    w_sum  = float(w.sum())

    # Kish effective sample size for weighted data
    n_eff = w_sum * w_sum / (float(np.sum(w * w)) + 1e-300)

    # Robust core spread: min(weighted std, weighted IQR / 1.349)
    mean  = float(np.sum(w * excess) / w_sum)
    var   = float(np.sum(w * (excess - mean) ** 2) / w_sum)
    sigma = np.sqrt(max(var, 0.0))
    iqr   = (_weighted_quantile(excess, w, 0.75)
             - _weighted_quantile(excess, w, 0.25))
    spread = min(sigma, iqr / 1.349) if iqr > 0.0 else sigma

    bw = 0.9 * spread * n_eff ** (-0.2) if spread > 0.0 else 0.0
    bw = max(bw * bw_scale, min_bw_s)

    # Outlier-robust time window: weighted high-quantile, not the max.
    t_hi   = _weighted_quantile(excess, w, tail_quantile)
    t_max  = max(t_hi + 4.0 * bw, 10.0 * bw)
    t_axis = np.linspace(0.0, t_max, n_grid)
    dt_grid = t_axis[1] - t_axis[0]

    # Fine weighted histogram on the grid (drop the rare beyond-window
    # outliers so they do not pile up at the right edge), then Gaussian-smooth.
    keep = excess <= t_max
    idx  = np.clip(np.round(excess[keep] / dt_grid).astype(np.int64), 0, n_grid - 1)
    hist = np.zeros(n_grid, dtype=np.float64)
    np.add.at(hist, idx, w[keep])

    sigma_bins = bw / dt_grid
    h = gaussian_filter1d(hist, sigma=max(sigma_bins, 1e-6), mode="reflect")

    peak = float(h.max())
    if peak > 0.0:
        h = h / peak
    return t_axis, h


def rms_delay_spread(weights, times) -> float:
    if weights.size < 2 or weights.sum() == 0:
        return float("nan")
    # Use excess delay (subtract first arrival) so the result is independent
    # of link range and represents only the multipath spread.
    excess = times - times.min()
    w_sum  = weights.sum()
    tau_m  = (weights * excess).sum() / w_sum
    return float(np.sqrt((weights * (excess - tau_m) ** 2).sum() / w_sum))


def frequency_response(h_norm, dt, pad_factor=8):
    N_pad = len(h_norm) * pad_factor
    H     = np.fft.rfft(h_norm, n=N_pad)
    H_mag = np.abs(H)
    dc    = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    return np.fft.rfftfreq(N_pad, d=dt), H_mag / dc


def frequency_response_direct(
    weights:        ndarray,
    times:          ndarray,
    delay_spread_s: float,
    *,
    n_freq:         int   = 800,
    f_span:         float = 15.0,
    max_photons:    int   = 8000,
    rng_seed:       int   = 0,
) -> Tuple[ndarray, ndarray]:
    """
    Channel frequency response |H(f)| via the exact discrete-time Fourier
    transform of the weighted photon arrival train:

        H(f) = Σ wᵢ · e^{−j2πf τᵢ} / Σ wᵢ ,     τᵢ = excess delay

    Why not FFT-of-CIR?
    -------------------
    The displayed CIR is a Gaussian KDE (see ``compute_cir_kde``); its kernel
    is a low-pass filter whose cut-off is set by the bandwidth *floor*
    ``min_bw_s``.  Reading the 3 dB bandwidth off the FFT of that smoothed CIR
    therefore **saturates** for near-ballistic channels: every short-range /
    clear-water case clips to ~1/(2π·min_bw_s) instead of tracking its true
    (few-picosecond) delay spread, producing a flat, range-independent
    bandwidth curve.

    The direct DTFT uses no histogram bin and no kernel, so the bandwidth it
    yields is independent of any display smoothing and correctly decreases as
    the delay spread grows with range.

    Frequency grid
    --------------
    Scaled to the RMS delay spread, ``f_max = f_span / (2π·τ_rms)``, so the
    −3 dB crossing is always captured and the grid itself shrinks with range
    (no fixed-resolution artefact).  The arrival train is uniformly subsampled
    to at most ``max_photons`` — the bandwidth is a low-order statistic and
    converges well before then, and |H(f)| normalised to DC is unbiased under
    uniform subsampling.

    Returns
    -------
    freqs : frequency grid (Hz), shape (n_freq,)
    H_mag : |H(f)| normalised to |H(0)| = 1, shape (n_freq,)
    """
    if times.size == 0 or float(weights.sum()) <= 0.0:
        return np.array([0.0]), np.array([1.0])

    excess = np.asarray(times, dtype=np.float64)
    excess = excess - excess.min()
    w      = np.asarray(weights, dtype=np.float64)

    # Subsample for cost control (bandwidth converges with a few thousand)
    if excess.size > max_photons:
        rng = np.random.default_rng(rng_seed)
        sel = rng.choice(excess.size, size=max_photons, replace=False)
        excess, w = excess[sel], w[sel]

    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return np.array([0.0]), np.array([1.0])
    w = w / w_sum

    # Frequency span from the RMS delay spread (fall back to the sample's own
    # spread if the caller's value is undefined).
    ds = delay_spread_s
    if not (ds and np.isfinite(ds) and ds > 0.0):
        mean = float(np.sum(w * excess))
        ds   = float(np.sqrt(max(np.sum(w * (excess - mean) ** 2), 0.0)))
    if ds <= 0.0:
        ds = 1e-12
    f_max = f_span / (2.0 * np.pi * ds)
    freqs = np.linspace(0.0, f_max, n_freq)

    # DTFT, chunked over frequency to bound memory.
    H     = np.empty(n_freq, dtype=np.complex128)
    chunk = 256
    for i in range(0, n_freq, chunk):
        fb = freqs[i:i + chunk]
        H[i:i + fb.size] = np.exp(-2.0j * np.pi * np.outer(fb, excess)) @ w
    H_mag = np.abs(H)
    dc    = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    return freqs, H_mag / dc


def bandwidth_3dB(freqs, H_norm) -> float:
    """
    3 dB bandwidth: lowest frequency where |H(f)| drops below 1/√2.

    The crossing is **linearly interpolated** between the two samples that
    straddle the threshold, so the returned value is not quantised to the FFT
    frequency grid — important when the bandwidth is a reported metric.
    """
    thr   = 1.0 / np.sqrt(2.0)
    below = np.where(H_norm < thr)[0]
    if below.size == 0:
        return float(freqs[-1])
    i = int(below[0])
    if i == 0:
        return float(freqs[0])
    f0, f1 = float(freqs[i - 1]), float(freqs[i])
    h0, h1 = float(H_norm[i - 1]), float(H_norm[i])
    if h1 == h0:
        return f1
    return f0 + (thr - h0) * (f1 - f0) / (h1 - h0)


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

    # CIR figure: weighted Gaussian KDE — smooth and outlier-robust even at low
    # photon counts (see compute_cir_kde).
    bw_scale  = float(getattr(cfg, "cir_kde_bw_scale", 1.0))
    n_grid    = int(getattr(cfg, "cir_n_grid", 512))
    tail_q    = float(getattr(cfg, "cir_tail_quantile", 0.995))
    t_ax, h   = compute_cir_kde(weights, times, n_grid=n_grid,
                                bw_scale=bw_scale, tail_quantile=tail_q)

    # Frequency response + 3 dB bandwidth: direct DTFT of the raw arrivals, NOT
    # the FFT of the smoothed CIR.  The KDE kernel is a low-pass whose cut-off
    # is pinned by the bandwidth floor, which would otherwise saturate the
    # bandwidth of near-ballistic channels (a flat, range-independent curve).
    # The direct DTFT is smoothing-independent and tracks the true delay spread.
    if weights.size > 1:
        freqs, Hn = frequency_response_direct(weights, times, ds)
        bw        = bandwidth_3dB(freqs, Hn)
    else:
        freqs, Hn = np.array([0.0]), np.array([1.0])
        bw        = float("nan")

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


