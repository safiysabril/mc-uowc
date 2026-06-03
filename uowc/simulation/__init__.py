"""
uowc.simulation
===============
Parallel orchestration layer.

Separation-of-Concern role
---------------------------
  Distributes photon batches across CPU cores and aggregates PhotonRecord
  dicts returned by each worker into RunResult objects.
  No physics, no metrics, no Pandas.

Return format
-------------
  All public functions return Dict[RunKey, RunResult].
  RunResult.record is a PhotonRecord (dict of parallel NumPy arrays).
  RunResult.n_launched is the total photons launched (adaptive denominator).

  Downstream consumers:
    metrics.compute_all_metrics(result)   — channel metrics from arrays
    analysis.to_dataframe(results)        — full Pandas DataFrame
"""

from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

from uowc.config import (
    WaterParams, BeamParams, SimConfig,
    RECEIVER, ALL_WATERS, ALL_BEAMS,
)
from uowc.medium import MediumProfile
from uowc.transport import (
    propagate_batch,
    propagate_batch_inhomogeneous,
    propagate_batch_inhomogeneous_turbulent,
    PhotonRecord,
)
from uowc.turbulence import NoTurbulence, TurbulenceProfile
from uowc.simulation.convergence import build_monitor

# Null-object turbulence shared by all turbulence-free dispatches.
_NO_TURB = NoTurbulence()


def _turbulence_for(medium) -> TurbulenceProfile:
    """Turbulence profile carried by a medium, or the null object.

    A :class:`~uowc.turbulence.CoupledOceanMedium` is *both* a MediumProfile
    and a TurbulenceProfile — its depth layers bundle aligned IOP and
    turbulence parameters — so it is its own turbulence source.  Plain media
    (HomogeneousMedium, LayeredMedium, GradientMedium) carry no turbulence and
    fall back to :data:`_NO_TURB`.

    Detection is by duck typing, not ``isinstance``: CoupledOceanMedium
    implements the TurbulenceProfile *protocol* (``C_n2`` + ``is_turbulence_free``)
    without inheriting from it, so an isinstance check would wrongly classify it
    as turbulence-free.
    """
    is_free = getattr(medium, "is_turbulence_free", None)
    if callable(getattr(medium, "C_n2", None)) and callable(is_free) and not is_free():
        return medium
    return _NO_TURB


# ─────────────────────────────────────────────────────────────────────────────
# Public named tuples
# ─────────────────────────────────────────────────────────────────────────────

class RunKey(NamedTuple):
    water_name: str
    beam_name:  str
    link_range: float


class RunResult(NamedTuple):
    """
    Output of every single-run and sweep function.

    Attributes
    ----------
    record     : PhotonRecord — dict of parallel NumPy arrays for captured photons
    n_launched : total photons launched (correct denominator for power normalisation)

    Usage
    -----
    Unpack like a 2-tuple:
        record, n_launched = result
    Access fields:
        result.record["weight"]
        result.record["tof_s"]
        result.n_launched
    """
    record:     PhotonRecord
    n_launched: int

    @property
    def weights(self) -> np.ndarray:
        """Convenience alias — backward-compatible with old (weights, times) API."""
        return self.record["weight"]

    @property
    def times(self) -> np.ndarray:
        """Convenience alias."""
        return self.record["tof_s"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal: aggregate PhotonRecord results from worker pool
# ─────────────────────────────────────────────────────────────────────────────

def _concat_records(batch_records: List[PhotonRecord]) -> PhotonRecord:
    """Concatenate PhotonRecord dicts returned by each worker process."""
    if not batch_records:
        from uowc.transport import _empty_record
        return _empty_record()
    keys = batch_records[0].keys()
    return {k: np.concatenate([r[k] for r in batch_records]) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: dispatch one batch to worker pool
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_homogeneous(
    water: WaterParams, beam: BeamParams,
    link_range_m: float, cfg: SimConfig,
    seed: np.random.SeedSequence,
    n_photons: int | None = None,
) -> PhotonRecord:
    nw           = cfg.n_workers
    total        = cfg.n_photons if n_photons is None else n_photons
    base, rem    = divmod(total, nw)
    batches      = [base + (1 if i < rem else 0) for i in range(nw)]
    worker_seeds = seed.spawn(nw)

    args_list = [
        (batches[i], worker_seeds[i].generate_state(4, dtype=np.uint64),
         water.c, water.b, water.g, water.omega,
         beam.divergence_rad, beam.waist_m,
         RECEIVER.aperture_radius_m, RECEIVER.fov_rad,
         link_range_m, cfg.weight_threshold, cfg.roulette_m, cfg.chunk_size)
        for i in range(nw)
    ]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        batch_records = list(ex.map(propagate_batch, args_list))
    return _concat_records(batch_records)


def _dispatch_inhomogeneous(
    medium: MediumProfile, beam: BeamParams,
    link_range_m: float, cfg: SimConfig,
    seed: np.random.SeedSequence,
    n_photons: int | None = None,
) -> PhotonRecord:
    """Dispatch a Woodcock (inhomogeneous) batch across the worker pool.

    If the medium carries turbulence (a CoupledOceanMedium with non-zero
    C_n²), the turbulent worker is used and the medium is passed as its own
    turbulence profile — phase-screen angular kicks along the path plus
    scintillation fading at the receiver.  Otherwise the plain Woodcock worker
    runs (turbulence-free).
    """
    nw           = cfg.n_workers
    total        = cfg.n_photons if n_photons is None else n_photons
    base, rem    = divmod(total, nw)
    batches      = [base + (1 if i < rem else 0) for i in range(nw)]
    worker_seeds = seed.spawn(nw)
    turbulence   = _turbulence_for(medium)

    common = lambda i: (
        batches[i], worker_seeds[i].generate_state(4, dtype=np.uint64),
        medium,
        beam.divergence_rad, beam.waist_m,
        RECEIVER.aperture_radius_m, RECEIVER.fov_rad,
        link_range_m, cfg.weight_threshold, cfg.roulette_m, cfg.chunk_size,
    )

    if turbulence.is_turbulence_free():
        worker    = propagate_batch_inhomogeneous
        args_list = [common(i) for i in range(nw)]
    else:
        worker    = propagate_batch_inhomogeneous_turbulent
        args_list = [common(i) + (turbulence,) for i in range(nw)]

    with ProcessPoolExecutor(max_workers=nw) as ex:
        batch_records = list(ex.map(worker, args_list))
    return _concat_records(batch_records)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: adaptive loop
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_loop(
    dispatch_fn,
    cfg: SimConfig,
    seed: np.random.SeedSequence,
    label: str,
    verbose: bool,
) -> RunResult:
    """Launch batches until all required metrics converge (multi-metric).

    Stopping rule
    -------------
    Each batch of ``cfg.conv_batch_photons`` photons is fed to a
    :class:`~uowc.simulation.convergence.ConvergenceMonitor` built from
    ``cfg.conv_metrics``.  Every metric tracks its own batch-means relative
    standard error; the loop stops only when **all required** metrics satisfy

        n_samples >= cfg.min_conv_batches   and   rel_error < tolerance

    ``cfg.max_launched_photons`` is only an emergency safeguard for runs whose
    capture rate is so low that a metric never reaches its tolerance.
    """
    monitor = build_monitor(cfg)
    all_records: List[PhotonRecord] = []
    n_captured = 0
    n_launched = 0
    round_num  = 0
    cap_hit    = False
    converged  = False

    while True:
        round_num += 1
        batch_n = min(cfg.conv_batch_photons,
                      cfg.max_launched_photons - n_launched)
        if batch_n <= 0:
            cap_hit = True
            break

        if verbose:
            print(f"      launching round {round_num} batch={batch_n:,}",
                  flush=True)

        # Pass the per-round batch size explicitly — no shared-state mutation.
        rec         = dispatch_fn(seed, batch_n)
        n_launched += batch_n
        new_cap     = rec["weight"].size
        if new_cap > 0:
            all_records.append(rec)
            n_captured += new_cap

        monitor.update(rec, batch_n)

        if verbose:
            rate = 100.0 * n_captured / n_launched if n_launched else 0.0
            print(
                f"      round {round_num:>3}: +{new_cap:>6,}  "
                f"caps {n_captured:>7,} ({rate:.3f}%)  "
                f"{monitor.report()}  "
                f"[{n_launched/1e6:.1f}M launched]",
                flush=True,
            )

        if monitor.converged:
            converged = True
            break

        if n_launched >= cfg.max_launched_photons:
            cap_hit = True
            break

    if cap_hit:
        pending = ", ".join(monitor.unconverged_names()) or "none"
        print(f"  ⚠  {label}: max photons reached before convergence "
              f"(unconverged: {pending}; {n_launched:,} launched)",
              flush=True)
    elif verbose and converged:
        print(f"  ✓ {label}: all metrics converged after {round_num} batches "
              f"({n_launched:,} launched) — {monitor.report()}",
              flush=True)

    return RunResult(_concat_records(all_records), n_launched)


# ─────────────────────────────────────────────────────────────────────────────
# Public single-run functions
# ─────────────────────────────────────────────────────────────────────────────

def run_one(water, beam, link_range_m, cfg, seed, verbose=False) -> RunResult:
    return RunResult(_dispatch_homogeneous(water, beam, link_range_m, cfg, seed),
                     cfg.n_photons)


def run_one_inhomogeneous(medium, beam, link_range_m, cfg, seed, verbose=False) -> RunResult:
    return RunResult(_dispatch_inhomogeneous(medium, beam, link_range_m, cfg, seed),
                     cfg.n_photons)


def run_one_adaptive(water, beam, link_range_m, cfg, seed, verbose=False) -> RunResult:
    return _adaptive_loop(
        lambda s, n: _dispatch_homogeneous(water, beam, link_range_m, cfg, s, n),
        cfg, seed, f"{water.name}|{beam.name}|{link_range_m:.0f}m", verbose,
    )


def run_one_inhomogeneous_adaptive(medium, beam, link_range_m, cfg, seed,
                                   verbose=False) -> RunResult:
    return _adaptive_loop(
        lambda s, n: _dispatch_inhomogeneous(medium, beam, link_range_m, cfg, s, n),
        cfg, seed, f"{medium.name}|{beam.name}|{link_range_m:.0f}m", verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public sweeps — fixed and adaptive
# ─────────────────────────────────────────────────────────────────────────────

def _run_sweep(run_fn, entities, beams, cfg, seed_offset, verbose) -> Dict[RunKey, RunResult]:
    ss     = np.random.SeedSequence(cfg.master_seed + seed_offset)
    combos = [(e, b, Z) for e in entities for b in beams for Z in cfg.link_ranges_m]
    seeds  = ss.spawn(len(combos))
    results: Dict[RunKey, RunResult] = {}
    t0 = time.perf_counter()
    for (entity, beam, Z), seed in zip(combos, seeds):
        key = RunKey(entity.name, beam.name, float(Z))
        if verbose:
            print(f"  [{entity.name} | {beam.name} | {Z:.0f} m]", flush=True)
        result = run_fn(entity, beam, Z, cfg, seed, verbose=verbose)
        results[key] = result
        if verbose:
            n  = result.weights.size
            pct = 100.0 * n / result.n_launched
            print(f"    → {n:,} captured | {result.n_launched:,} launched "
                  f"| {pct:.3f}% | {time.perf_counter()-t0:.1f}s\n", flush=True)
    return results


def run_sweep(cfg, *, waters=ALL_WATERS, beams=ALL_BEAMS, verbose=True):
    return _run_sweep(run_one, waters, beams, cfg, 0, verbose)


def run_sweep_inhomogeneous(cfg, *, media: Sequence[MediumProfile],
                             beams=ALL_BEAMS, verbose=True):
    return _run_sweep(run_one_inhomogeneous, media, beams, cfg, 1, verbose)


def run_sweep_adaptive(cfg, *, waters=ALL_WATERS, beams=ALL_BEAMS, verbose=True):
    return _run_sweep(run_one_adaptive, waters, beams, cfg, 0, verbose)


def run_sweep_inhomogeneous_adaptive(cfg, *, media: Sequence[MediumProfile],
                                     beams=ALL_BEAMS, seed_offset: int = 1,
                                     verbose=True):
    """Adaptive Woodcock sweep over inhomogeneous media.

    Turbulence is decided per medium by the dispatch layer: a plain
    LayeredMedium/GradientMedium runs turbulence-free (Model 2), a turbulence-
    carrying CoupledOceanMedium runs with phase-screen kicks + scintillation
    (Model 3).  ``seed_offset`` distinguishes the RNG sub-streams of the two
    scenarios so that, within one ``main.py all`` invocation, Model 2 and
    Model 3 draw independent photons (Model 2 uses 1, Model 3 uses 2).
    """
    return _run_sweep(run_one_inhomogeneous_adaptive, media, beams, cfg,
                      seed_offset, verbose)