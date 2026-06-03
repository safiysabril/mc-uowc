"""
uowc.simulation.convergence
============================
Generalized, multi-metric convergence framework for the adaptive loop.

Why this exists
---------------
Monte Carlo received *power* is a first moment (Σw / N) and converges quickly.
It does **not** imply that distribution-shaped metrics — the CIR, the RMS delay
spread, the frequency response, the 3 dB bandwidth — have converged.  Stopping
on power alone is single-metric convergence and can leave shape metrics noisy.

This module turns convergence into a *metric-driven* decision.  Each metric is a
``ConvergenceCriterion`` that bundles three things:

    estimator           how to reduce one batch of photons to a sample
    uncertainty measure batch-means relative standard error  (‖SE‖ / ‖mean‖)
    convergence test    n_samples >= min_samples  and  rel_error < rel_tol

A ``ConvergenceMonitor`` holds several criteria and is *converged* only when
**all required** criteria are individually converged.

Statistical method  (batch means)
----------------------------------
Each adaptive round launches an independent batch (independent RNG sub-stream),
so the per-batch estimate ``m_i`` is an i.i.d. draw of the metric.  After ``n``
batches:

    mean = (1/n) Σ m_i
    SE   = std(m_i, ddof=1) / sqrt(n)
    rel_error = ‖SE‖₂ / ‖mean‖₂

For a scalar metric (power, delay spread, bandwidth) the L2 norm is the absolute
value, so this is the familiar SE/|mean|.  For a vector metric (CIR, frequency
response) it is the relative L2 size of the uncertainty of the whole shape — tail
bins with tiny mean contribute tiny absolute SE, so they do not dominate.

Extensibility
-------------
Register new metrics without touching the loop::

    from uowc.simulation.convergence import register_metric, ConvergenceCriterion

    def _make_ber(cfg):
        def est(record, n_launched): ...
        return est
    register_metric("ber", _make_ber)

then add ``"ber"`` to ``cfg.conv_metrics``.

Separation of concern
----------------------
Convergence is *simulation control logic*.  Estimators reuse the physics in
``uowc.metrics`` — they never reimplement it — and nothing here is imported by
transport, plotting, or reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # avoid import cost / any ordering issues
    from uowc.config import SimConfig
    from uowc.transport import PhotonRecord


# An estimator maps one batch (record, photons launched in that batch) to a
# metric sample.  Returning None means "undefined for this batch" (e.g. too few
# captured photons); the sample is skipped rather than poisoning the statistics.
Estimator     = Callable[[Dict[str, np.ndarray], int], "float | np.ndarray | None"]
MetricFactory = Callable[["SimConfig"], Estimator]


# ─────────────────────────────────────────────────────────────────────────────
# Status object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConvergenceStatus:
    """Immutable snapshot of one metric's convergence state."""
    name:      str
    estimate:  float        # scalar value (scalars) or mean magnitude (vectors)
    rel_error: float        # ‖SE‖ / ‖mean‖  (inf until >= 2 samples)
    rel_tol:   float
    n_samples: int
    converged: bool
    required:  bool = True
    unit:      str = ""

    @property
    def n_batches(self) -> int:          # legacy alias (old API name)
        return self.n_samples

    def short(self) -> str:
        flag = "✓" if self.converged else " "
        re   = "  inf" if not math.isfinite(self.rel_error) else f"{self.rel_error:.3f}"
        return f"{self.name}={re}{flag}"


# ─────────────────────────────────────────────────────────────────────────────
# Core statistic — works for scalar AND vector samples
# ─────────────────────────────────────────────────────────────────────────────

def relative_standard_error(
    samples: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Batch-means mean and relative standard error of a metric.

    Parameters
    ----------
    samples : sequence of per-batch metric samples, each a scalar or an ndarray
              of identical shape.

    Returns
    -------
    mean      : elementwise mean of the samples (ndarray; 0-d for scalars)
    rel_error : ‖SE‖₂ / ‖mean‖₂  (float, ``inf`` if < 2 samples or zero mean)
    """
    arr = np.stack([np.asarray(s, dtype=float) for s in samples])  # (n, *shape)
    n   = arr.shape[0]
    mean = arr.mean(axis=0)
    if n < 2:
        return mean, float("inf")
    se    = arr.std(axis=0, ddof=1) / math.sqrt(n)
    denom = float(np.linalg.norm(mean.ravel()))
    rel   = float(np.linalg.norm(se.ravel()) / denom) if denom > 0.0 else float("inf")
    return mean, rel


# ─────────────────────────────────────────────────────────────────────────────
# One metric's convergence tracker
# ─────────────────────────────────────────────────────────────────────────────

class ConvergenceCriterion:
    """Tracks one metric toward convergence using the batch-means method.

    Parameters
    ----------
    name        : metric label (used in logs and as registry key)
    estimator   : callable(record, n_launched_batch) -> sample | None
    rel_tol     : relative-error tolerance for this metric
    min_samples : minimum valid batches before convergence may be declared
    required    : if False the metric is tracked/reported but not gating
    unit        : optional unit string for display
    """

    def __init__(
        self,
        name: str,
        estimator: Estimator,
        rel_tol: float,
        *,
        min_samples: int = 3,
        required: bool = True,
        unit: str = "",
    ) -> None:
        self.name        = name
        self.estimator   = estimator
        self.rel_tol     = float(rel_tol)
        self.min_samples = max(2, int(min_samples))     # need >= 2 for an SE
        self.required    = bool(required)
        self.unit        = unit
        self._samples: List[np.ndarray] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def update(self, record: "PhotonRecord", n_launched_batch: int) -> None:
        """Reduce one batch to a sample and append it (skipping degenerate ones)."""
        sample = self.estimator(record, n_launched_batch)
        if sample is None:
            return
        arr = np.asarray(sample, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return
        self._samples.append(arr)

    def status(self) -> ConvergenceStatus:
        n = self.n_samples
        if n < 2:
            est = float(np.mean(self._samples[-1])) if n else 0.0
            return ConvergenceStatus(self.name, est, float("inf"),
                                     self.rel_tol, n, False, self.required, self.unit)
        mean, rel = relative_standard_error(self._samples)
        converged = (n >= self.min_samples) and (rel < self.rel_tol)
        return ConvergenceStatus(self.name, float(np.mean(mean)), rel,
                                 self.rel_tol, n, converged, self.required, self.unit)


# ─────────────────────────────────────────────────────────────────────────────
# A set of criteria — the control object the adaptive loop talks to
# ─────────────────────────────────────────────────────────────────────────────

class ConvergenceMonitor:
    """Aggregates several criteria; converged iff all *required* ones are."""

    def __init__(self, criteria: Sequence[ConvergenceCriterion]) -> None:
        self.criteria: List[ConvergenceCriterion] = list(criteria)

    def update(self, record: "PhotonRecord", n_launched_batch: int) -> None:
        for c in self.criteria:
            c.update(record, n_launched_batch)

    def statuses(self) -> List[ConvergenceStatus]:
        return [c.status() for c in self.criteria]

    @property
    def converged(self) -> bool:
        required = [s for s in self.statuses() if s.required]
        return bool(required) and all(s.converged for s in required)

    def unconverged_names(self) -> List[str]:
        return [s.name for s in self.statuses() if s.required and not s.converged]

    def report(self) -> str:
        """One-line summary of every metric's current relative error."""
        return "  ".join(s.short() for s in self.statuses())


# ─────────────────────────────────────────────────────────────────────────────
# Standard estimators  (reuse uowc.metrics — never reimplement physics)
# ─────────────────────────────────────────────────────────────────────────────

def _fixed_cir(weights, times, dt: float, n_bins: int) -> np.ndarray:
    """CIR on a **fixed** absolute grid (constant length and constant bin
    centres across every batch).

    Why not ``metrics.compute_cir``?
    --------------------------------
    ``compute_cir`` adapts both the bin width and the bin *count* to each
    batch (it picks ``n_use = min(n_stat, n_phys, n_bins)``).  That is correct
    for a single final figure, but it makes the per-batch CIR vectors have
    **different lengths** — so ``relative_standard_error`` could not
    ``np.stack`` them (a hard ``ValueError``), and bin *i* of one batch would
    not correspond to the same delay as bin *i* of another.

    For a convergence *sample* we need one fixed grid so the batch-means
    statistic is well defined.  Energy concentrates in the first few ns of
    excess delay, so the relative-L2 error is dominated by the populated bins;
    the constant grid is the statistically sound choice here.
    """
    h = np.zeros(n_bins, dtype=float)
    if times.size:
        excess = times - times.min()
        edges  = np.linspace(0.0, dt * n_bins, n_bins + 1)
        h, _   = np.histogram(excess, bins=edges, weights=weights)
        s = h.sum()
        if s > 0:
            h = h / s
    return h


def _power_estimator(record: "PhotonRecord", n_launched: int) -> float:
    """Mean linear received power per launched photon for this batch."""
    w = record["weight"]
    return float(w.sum()) / n_launched if n_launched else 0.0


def _make_delay_spread(cfg: "SimConfig") -> Estimator:
    from uowc.metrics import rms_delay_spread

    def est(record, n_launched):
        if record["weight"].size < 2:
            return None
        ds = rms_delay_spread(record["weight"], record["tof_s"])
        return ds if math.isfinite(ds) else None
    return est


def _make_bandwidth(cfg: "SimConfig") -> Estimator:
    from uowc.metrics import frequency_response, bandwidth_3dB

    def est(record, n_launched):
        if record["weight"].size < 2:
            return None
        # Fixed grid → dt is genuinely cfg.dt_bin_s, so the frequency axis is
        # correct (the previous code binned adaptively but still passed
        # cfg.dt_bin_s, mis-scaling the spectrum).
        h         = _fixed_cir(record["weight"], record["tof_s"],
                               cfg.dt_bin_s, cfg.n_time_bins)
        freqs, Hn = frequency_response(h, cfg.dt_bin_s)
        return bandwidth_3dB(freqs, Hn)
    return est


def _make_cir(cfg: "SimConfig") -> Estimator:
    def est(record, n_launched):
        if record["weight"].size < 2:
            return None
        return _fixed_cir(record["weight"], record["tof_s"],
                          cfg.dt_bin_s, cfg.n_time_bins)   # fixed-length vector
    return est


def _make_frequency_response(cfg: "SimConfig") -> Estimator:
    from uowc.metrics import frequency_response

    def est(record, n_launched):
        if record["weight"].size < 2:
            return None
        h      = _fixed_cir(record["weight"], record["tof_s"],
                            cfg.dt_bin_s, cfg.n_time_bins)
        _, Hn  = frequency_response(h, cfg.dt_bin_s)
        return Hn                             # fixed-length vector sample
    return est


# ─────────────────────────────────────────────────────────────────────────────
# Registry  (extensible)
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, MetricFactory] = {
    "power":              lambda cfg: _power_estimator,
    "delay_spread":       _make_delay_spread,
    "bandwidth":          _make_bandwidth,
    "cir":                _make_cir,
    "frequency_response": _make_frequency_response,
}

_UNITS: Dict[str, str] = {
    "power": "", "delay_spread": "s", "bandwidth": "Hz",
    "cir": "", "frequency_response": "",
}


def register_metric(name: str, factory: MetricFactory, *, unit: str = "") -> None:
    """Register a custom convergence metric so it can be named in ``conv_metrics``."""
    _REGISTRY[name] = factory
    _UNITS[name] = unit


def available_metrics() -> Tuple[str, ...]:
    return tuple(_REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# Builder — assemble a monitor from a SimConfig
# ─────────────────────────────────────────────────────────────────────────────

def build_monitor(cfg: "SimConfig") -> ConvergenceMonitor:
    """Construct a :class:`ConvergenceMonitor` from a :class:`SimConfig`.

    Reads three (optional) config fields:

    * ``conv_metrics`` : tuple of metric names that must converge.
                         Defaults to ``("power",)`` (backward compatible).
    * ``conv_tols``    : tuple of ``(name, tol)`` overrides; otherwise
                         ``rel_error_tol`` is used for every metric.
    * ``min_conv_batches`` : minimum valid batches before stopping.
    """
    names       = tuple(getattr(cfg, "conv_metrics", ("power",)))
    tols        = dict(getattr(cfg, "conv_tols", ()))
    default_tol = getattr(cfg, "rel_error_tol", 0.05)
    min_b       = getattr(cfg, "min_conv_batches", 3)

    criteria: List[ConvergenceCriterion] = []
    for name in names:
        if name not in _REGISTRY:
            raise KeyError(
                f"Unknown convergence metric {name!r}. "
                f"Known metrics: {available_metrics()}. "
                f"Register custom ones with register_metric()."
            )
        criteria.append(ConvergenceCriterion(
            name,
            _REGISTRY[name](cfg),
            tols.get(name, default_tol),
            min_samples=min_b,
            required=True,
            unit=_UNITS.get(name, ""),
        ))
    return ConvergenceMonitor(criteria)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible scalar helper
# ─────────────────────────────────────────────────────────────────────────────

def power_rel_error(samples: Sequence[float]) -> ConvergenceStatus:
    """Legacy helper: relative standard error of a list of scalar power samples.

    Retained for backward compatibility.  New code should use
    :func:`build_monitor` / :class:`ConvergenceMonitor`.
    """
    n = len(samples)
    if n == 0:
        return ConvergenceStatus("power", 0.0, float("inf"), 0.05, 0, False)
    if n < 2:
        return ConvergenceStatus("power", float(samples[-1]), float("inf"),
                                 0.05, n, False)
    mean, rel = relative_standard_error([np.asarray(s, float) for s in samples])
    return ConvergenceStatus("power", float(np.mean(mean)), rel, 0.05, n, False)
