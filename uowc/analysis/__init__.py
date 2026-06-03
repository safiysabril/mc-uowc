"""
uowc.analysis
=============
Pandas-based data pipeline for photon event analysis.

Separation-of-Concern role
---------------------------
  This module owns the transformation from raw NumPy simulation output into
  structured Pandas DataFrames, and all statistical analyses that operate on
  those DataFrames.  It knows nothing about photon transport physics,
  parallelism, or plot rendering.

  The boundary is:
    simulation  →  RunResult (NumPy arrays)
    analysis    →  pd.DataFrame  (structured records + statistics)
    plotting    →  figures (Matplotlib / Seaborn)

DataFrame schema
----------------
  One row per captured photon.  Columns:

  photon_id     int64     globally unique across the whole sweep
  run_id        int32     index of the (medium, beam, range) combination
  medium_name   category  water type or medium profile name
  beam_name     category  beam type name
  link_range_m  float32   target receiver depth (m)
  weight        float64   final photon weight
  tof_s         float64   time of flight (s)
  tof_ns        float64   time of flight (ns)  — derived, for readability
  x_m           float32   x position at capture plane (m)
  y_m           float32   y position at capture plane (m)
  r_m           float32   radial distance from beam axis at capture (m)
  path_length_m float32   total optical path length (m)
  n_scatters    int32     number of real scattering events
  n_nulls       int32     null (virtual) Woodcock collisions
  excess_path_m float32   path_length_m - link_range_m  (extra distance due to scatter)

Storage strategy
----------------
  A. In-memory (default) — build the full DataFrame once at the end of the
     sweep.  Best for runs up to ~10M captured photons (~2 GB RAM).
     Use to_dataframe() / to_parquet() / to_hdf().

  B. Incremental Parquet (recommended for large runs) — write each
     (medium, beam, range) result as a separate Parquet partition as it
     completes; read back lazily.  Zero peak-memory penalty.
     Use write_parquet_partition() then read_parquet_dataset().

File format guidance
--------------------
  CSV     — human-readable, no dependencies; 5–10× larger than Parquet.
             Use only for small exports or manual inspection.
  Parquet — columnar, compressed, typed; best for large datasets and
             repeated analysis.  Requires pyarrow or fastparquet.
  HDF5    — good for mixed numeric/metadata; heavier dependency (tables).
             Prefer Parquet unless you need random row access.

References
----------
  McKinney (2017) — Python for Data Analysis, 3rd ed.
  VanderPlas (2016) — Python Data Science Handbook
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

from uowc.simulation import RunKey, RunResult


# ─────────────────────────────────────────────────────────────────────────────
# Schema definition (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA: Dict[str, str] = {
    "photon_id":     "int64",
    "run_id":        "int32",
    "medium_name":   "category",
    "beam_name":     "category",
    "link_range_m":  "float32",
    "weight":        "float64",
    "tof_s":         "float64",
    "tof_ns":        "float64",
    "x_m":           "float32",
    "y_m":           "float32",
    "r_m":           "float32",
    "path_length_m": "float32",
    "n_scatters":    "int32",
    "n_nulls":       "int32",
    "excess_path_m": "float32",
    # run-level metadata stored per photon row for self-contained Parquet
    "n_launched":    "int64",   # total photons launched — power normalisation denominator
    "c_ref":         "float32", # beam-attenuation coefficient used for Beer-Lambert reference
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A: In-memory DataFrame construction
# ─────────────────────────────────────────────────────────────────────────────

def to_dataframe(
    sweep_results: Dict[RunKey, RunResult],
    *,
    run_id_map: Optional[Dict[RunKey, int]] = None,
    c_ref_map:  Optional[Dict[RunKey, float]] = None,
) -> pd.DataFrame:
    """
    Convert an entire sweep result dict into a single Pandas DataFrame.

    This is Strategy A — all data is materialised in memory at once.
    Use when total captured photons fit comfortably in RAM (~10M photons
    ≈ 1–2 GB depending on column count).

    Parameters
    ----------
    sweep_results : Dict[RunKey, RunResult] from any run_sweep* function
    run_id_map    : optional explicit mapping of RunKey → int run_id.
                    If None, run_id is assigned by enumeration order.
    c_ref_map     : optional {RunKey: c} mapping of beam-attenuation coefficients.
                    Stored in the ``c_ref`` column so Parquet is self-contained for
                    figure regeneration.  Pass ``{RunKey(...): water.c}`` for
                    homogeneous runs, or ``medium.c_max`` for inhomogeneous.

    Returns
    -------
    pd.DataFrame with columns matching SCHEMA
    """
    frames: List[pd.DataFrame] = []
    photon_cursor = 0

    for run_idx, (key, result) in enumerate(sweep_results.items()):
        rec = result.record
        n   = rec["weight"].size
        if n == 0:
            continue

        run_id = run_id_map[key] if run_id_map else run_idx
        r_m    = np.hypot(rec["x_m"], rec["y_m"])
        c_ref  = float(c_ref_map[key]) if (c_ref_map and key in c_ref_map) else float("nan")

        chunk = pd.DataFrame({
            "photon_id":     np.arange(photon_cursor, photon_cursor + n, dtype=np.int64),
            "run_id":        np.full(n, run_id,               dtype=np.int32),
            "medium_name":   pd.Categorical([key.water_name] * n),
            "beam_name":     pd.Categorical([key.beam_name]  * n),
            "link_range_m":  np.full(n, key.link_range,       dtype=np.float32),
            "weight":        rec["weight"].astype(np.float64),
            "tof_s":         rec["tof_s"].astype(np.float64),
            "tof_ns":        (rec["tof_s"] * 1e9).astype(np.float64),
            "x_m":           rec["x_m"],
            "y_m":           rec["y_m"],
            "r_m":           r_m.astype(np.float32),
            "path_length_m": rec["path_length_m"],
            "n_scatters":    rec["n_scatters"],
            "n_nulls":       rec["n_nulls"],
            "excess_path_m": (rec["path_length_m"] - key.link_range).astype(np.float32),
            "n_launched":    np.full(n, result.n_launched,    dtype=np.int64),
            "c_ref":         np.full(n, c_ref,                dtype=np.float32),
        })
        frames.append(chunk)
        photon_cursor += n

    if not frames:
        return pd.DataFrame(columns=list(SCHEMA.keys())).astype(SCHEMA)

    df = pd.concat(frames, ignore_index=True)
    return df.astype({k: v for k, v in SCHEMA.items() if k in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B: Incremental Parquet I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_parquet_partition(
    key:    RunKey,
    result: RunResult,
    out_dir: str | Path,
    *,
    run_id: int = 0,
    photon_id_offset: int = 0,
    c_ref: float = float("nan"),
    **kwargs,
) -> Path:
    """
    Write one (medium, beam, range) result as a Parquet partition file.

    Strategy B — call this inside your sweep loop immediately after each
    run completes, before moving to the next combination.  Peak RAM is
    bounded to one run at a time, regardless of the total dataset size.

    File naming: {out_dir}/{medium_name}/{beam_name}/range_{Z:.0f}m.parquet
    This creates a Hive-style directory partition that pd.read_parquet()
    can read lazily with partition pruning.

    Parameters
    ----------
    key             : RunKey identifying this combination
    result          : RunResult for this combination
    out_dir         : root output directory
    run_id          : numeric ID for this simulation run
    photon_id_offset: starting photon_id (for globally unique IDs across calls)

    Returns
    -------
    Path to the written Parquet file
    """
    rec = result.record
    n   = rec["weight"].size

    # Build a single-run DataFrame
    if n == 0:
        df = pd.DataFrame(columns=list(SCHEMA.keys())).astype(SCHEMA)
    else:
        r_m = np.hypot(rec["x_m"], rec["y_m"])
        df  = pd.DataFrame({
            "photon_id":     np.arange(photon_id_offset,
                                       photon_id_offset + n, dtype=np.int64),
            "run_id":        np.full(n, run_id,               dtype=np.int32),
            "medium_name":   pd.Categorical([key.water_name] * n),
            "beam_name":     pd.Categorical([key.beam_name]  * n),
            "link_range_m":  np.full(n, key.link_range,       dtype=np.float32),
            "weight":        rec["weight"].astype(np.float64),
            "tof_s":         rec["tof_s"].astype(np.float64),
            "tof_ns":        (rec["tof_s"] * 1e9).astype(np.float64),
            "x_m":           rec["x_m"],
            "y_m":           rec["y_m"],
            "r_m":           r_m.astype(np.float32),
            "path_length_m": rec["path_length_m"],
            "n_scatters":    rec["n_scatters"],
            "n_nulls":       rec["n_nulls"],
            "excess_path_m": (rec["path_length_m"] - key.link_range).astype(np.float32),
            "n_launched":    np.full(n, result.n_launched,    dtype=np.int64),
            "c_ref":         np.full(n, c_ref,                dtype=np.float32),
        })
        df = df.astype({k: v for k, v in SCHEMA.items() if k in df.columns})

    # Hive-style directory structure
    safe_medium = key.water_name.replace(" ", "_").replace("→", "to")
    safe_beam   = key.beam_name.replace(" ", "_").replace("(", "").replace(")", "")
    part_dir    = Path(out_dir) / safe_medium / safe_beam
    part_dir.mkdir(parents=True, exist_ok=True)
    fpath = part_dir / f"range_{key.link_range:.0f}m.parquet"
    df.to_parquet(fpath, index=False, engine="auto", compression="snappy")
    return fpath


def read_parquet_dataset(out_dir: str | Path) -> pd.DataFrame:
    """
    Read all Parquet partitions back into a single DataFrame.

    Uses pandas read_parquet with a directory path — reads all .parquet
    files found recursively and concatenates them.  Suitable for post-hoc
    analysis when the simulation has already completed.
    """
    parts = list(Path(out_dir).rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files found under {out_dir}")
    dfs = [pd.read_parquet(p, engine="auto") for p in sorted(parts)]
    df  = pd.concat(dfs, ignore_index=True)
    # Re-apply category dtype (lost during Parquet round-trip on some versions)
    for col in ("medium_name", "beam_name"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def reconstruct_sweep_results(df: pd.DataFrame) -> Dict:
    """
    Rebuild a ``{RunKey: RunResult}`` dict from a Parquet-loaded DataFrame.

    Requires the ``n_launched`` column (present in files written after the
    schema update).  Use this to regenerate figures from saved Parquet without
    re-running the simulation.

    Example
    -------
    >>> df   = pd.read_parquet("outputs/photons_homogeneous.parquet")
    >>> raw  = reconstruct_sweep_results(df)
    >>> metrics = {k: compute_all_metrics(v, SIM, ...) for k, v in raw.items()}
    """
    from uowc.simulation import RunKey, RunResult  # local to avoid circular import

    results: Dict = {}
    for (medium, beam, Z), grp in df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        n_launched = int(grp["n_launched"].iloc[0]) if "n_launched" in grp.columns else 0
        rec = {
            "weight":        grp["weight"].to_numpy(dtype=np.float64),
            "tof_s":         grp["tof_s"].to_numpy(dtype=np.float64),
            "x_m":           grp["x_m"].to_numpy(dtype=np.float32),
            "y_m":           grp["y_m"].to_numpy(dtype=np.float32),
            "path_length_m": grp["path_length_m"].to_numpy(dtype=np.float32),
            "n_scatters":    grp["n_scatters"].to_numpy(dtype=np.int32),
            "n_nulls":       grp["n_nulls"].to_numpy(dtype=np.int32),
        }
        key = RunKey(str(medium), str(beam), float(Z))
        results[key] = RunResult(rec, n_launched)
    return results


def launched_map_from_df(df: pd.DataFrame) -> Dict:
    """
    Extract ``{RunKey: n_launched}`` from a Parquet DataFrame.

    Replaces the manual ``{k: v.n_launched for k, v in raw.items()}`` that
    requires the original RunResult dict to be in memory.
    """
    from uowc.simulation import RunKey

    out: Dict = {}
    for (medium, beam, Z), grp in df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        out[RunKey(str(medium), str(beam), float(Z))] = int(grp["n_launched"].iloc[0])
    return out


def c_ref_map_from_df(df: pd.DataFrame) -> Dict:
    """
    Extract ``{RunKey: c_ref}`` from a Parquet DataFrame.

    Use as the ``c`` argument to ``compute_all_metrics`` when regenerating
    metrics from Parquet.
    """
    from uowc.simulation import RunKey

    out: Dict = {}
    for (medium, beam, Z), grp in df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        out[RunKey(str(medium), str(beam), float(Z))] = float(grp["c_ref"].iloc[0])
    return out


def merge_parquet_files(
    paths: Sequence[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Merge multiple simulation Parquet files into one, accumulating photons.

    Running the simulation multiple times and merging the outputs is an
    effective way to build up enough captured photons for sparse runs
    (long range, turbid water) without hitting the per-run memory cap.

    ``n_launched`` is **summed** across input files for each
    (medium, beam, range) group so power normalisation stays correct in
    the merged file.  All photon rows are concatenated as-is.

    Parameters
    ----------
    paths       : list of Parquet file paths to merge
    output_path : where to write the merged file

    Returns
    -------
    The merged DataFrame (also written to output_path).
    """
    from uowc.simulation import RunKey  # local import to avoid circular

    launched_per_file: List[Dict] = []
    frames: List[pd.DataFrame]  = []

    for path in paths:
        df = pd.read_parquet(path)
        launched: Dict = {}
        for (medium, beam, Z), grp in df.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ):
            key = RunKey(str(medium), str(beam), float(Z))
            launched[key] = int(grp["n_launched"].iloc[0])
        launched_per_file.append(launched)
        frames.append(df)

    # Sum n_launched across files per group
    launched_total: Dict = {}
    for launched in launched_per_file:
        for key, n in launched.items():
            launched_total[key] = launched_total.get(key, 0) + n

    # Concatenate all photon rows
    merged = pd.concat(frames, ignore_index=True)

    # Update n_launched in merged DataFrame to the cumulative total
    for (medium, beam, Z), idx in merged.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ).groups.items():
        key = RunKey(str(medium), str(beam), float(Z))
        if key in launched_total:
            merged.loc[idx, "n_launched"] = launched_total[key]

    # Re-apply category dtypes (sometimes lost on concat)
    for col in ("medium_name", "beam_name"):
        if col in merged.columns:
            merged[col] = merged[col].astype("category")

    to_parquet(merged, output_path)
    return merged


def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a full in-memory DataFrame to a single Parquet file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="auto", compression="snappy")
    _log.info("Saved Parquet → %s  (%.1f MB)", path, os.path.getsize(path) / 1e6)


def append_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Accumulate photons into a Parquet file across simulation runs.

    If the file does not yet exist, writes ``df`` directly (same as
    to_parquet).  If it already exists, merges the existing data with
    ``df`` using merge_parquet_files logic — photon rows are
    concatenated and ``n_launched`` is summed per (medium, beam, range)
    group so power normalisation stays correct.

    Use this instead of to_parquet in main.py so that successive runs
    accumulate photons in a single file rather than overwriting it.
    This is especially useful for turbid water / long range where a
    single run captures too few photons for a clean CIR histogram.
    """
    path = Path(path)
    if not path.exists():
        to_parquet(df, path)
        return

    # Merge existing file with new data
    existing = pd.read_parquet(path)

    from uowc.simulation import RunKey  # local import to avoid circular

    # Sum n_launched per group across both DataFrames
    launched_total: Dict = {}
    for frame in (existing, df):
        for (medium, beam, Z), grp in frame.groupby(
            ["medium_name", "beam_name", "link_range_m"], observed=True
        ):
            key = RunKey(str(medium), str(beam), float(Z))
            launched_total[key] = launched_total.get(key, 0) + int(grp["n_launched"].iloc[0])

    merged = pd.concat([existing, df], ignore_index=True)

    for (medium, beam, Z), idx in merged.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ).groups.items():
        key = RunKey(str(medium), str(beam), float(Z))
        if key in launched_total:
            merged.loc[idx, "n_launched"] = launched_total[key]

    for col in ("medium_name", "beam_name"):
        if col in merged.columns:
            merged[col] = merged[col].astype("category")

    to_parquet(merged, path)
    n_prev = len(existing)
    n_new  = len(df)
    _log.info("Appended %d rows to existing %d rows → %d total", n_new, n_prev, len(merged))


def to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame to CSV (for human inspection of small subsets)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _log.info("Saved CSV → %s  (%.1f MB)", path, os.path.getsize(path) / 1e6)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analyses
# ─────────────────────────────────────────────────────────────────────────────

def capture_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(medium, beam, range) capture statistics.

    Uses named aggregations via .agg() instead of .apply() to avoid
    the include_groups parameter (added pandas 2.2) and its older-stub
    overload gap.

    Returns a summary DataFrame with columns:
      n_captured, mean_weight, std_weight,
      mean_tof_ns, std_tof_ns,
      mean_n_scatters, mean_excess_path_m
    """
    return (
        df.groupby(["medium_name", "beam_name", "link_range_m"], observed=True)
          .agg(
              n_captured        = ("weight",        "count"),
              mean_weight       = ("weight",        "mean"),
              std_weight        = ("weight",        "std"),
              mean_tof_ns       = ("tof_ns",        "mean"),
              std_tof_ns        = ("tof_ns",        "std"),
              mean_n_scatters   = ("n_scatters",    "mean"),
              mean_excess_path_m = ("excess_path_m", "mean"),
          )
          .reset_index()
    )


def capture_statistics_with_launched(
    df: pd.DataFrame,
    launched_map: Dict[RunKey, int],
) -> pd.DataFrame:
    """
    Same as capture_statistics but uses the correct n_launched denominator
    from the simulation sweep results to compute capture_rate_pct and power_dB.

    Parameters
    ----------
    df           : photon DataFrame from to_dataframe()
    launched_map : {RunKey: n_launched} from {key: result.n_launched for ...}
    """
    rows = []
    for key, n_launched in launched_map.items():
        mask = (
            (df["medium_name"]  == key.water_name) &
            (df["beam_name"]    == key.beam_name) &
            (df["link_range_m"] == np.float32(key.link_range))
        )
        sub = df[mask]
        n   = len(sub)
        w   = sub["weight"]
        rows.append({
            "medium_name":       key.water_name,
            "beam_name":         key.beam_name,
            "link_range_m":      key.link_range,
            "n_captured":        n,
            "n_launched":        n_launched,
            "capture_rate_pct":  100.0 * n / n_launched if n_launched else 0.0,
            "power_dB":          (10.0 * np.log10(w.sum() / n_launched + 1e-300)
                                  if n_launched else -np.inf),
            "mean_weight":       float(w.mean()) if n else float("nan"),
            "std_weight":        float(w.std())  if n > 1 else float("nan"),
            "mean_tof_ns":       float(sub["tof_ns"].mean()) if n else float("nan"),
            "std_tof_ns":        float(sub["tof_ns"].std())  if n > 1 else float("nan"),
            "mean_n_scatters":   float(sub["n_scatters"].mean()) if n else float("nan"),
            "mean_excess_path_m": float(sub["excess_path_m"].mean()) if n else float("nan"),
            "ci_95_low_dB":      _power_ci(w, n_launched, z=1.96)[0],
            "ci_95_high_dB":     _power_ci(w, n_launched, z=1.96)[1],
        })
    return pd.DataFrame(rows)


def _power_ci(weights: pd.Series, n_launched: int, z: float = 1.96):
    """
    95% confidence interval on received power (dB) via bootstrap on weight sum.

    The variance of Σwᵢ / N is Var(wᵢ) / N for iid photons.
    We propagate via the delta method:
        σ_P ≈ (10/ln10) · σ_w_sum / (N · P_lin)
    where P_lin = Σwᵢ / N.
    """
    n = len(weights)
    if n < 2 or n_launched == 0:
        return float("nan"), float("nan")
    w_sum  = float(weights.sum())
    w_std  = float(weights.std())
    P_lin  = w_sum / n_launched
    if P_lin <= 0:
        return float("nan"), float("nan")
    # std of the sample mean of w, scaled to the sum-over-N denominator
    sigma_P_lin = w_std / np.sqrt(n) / n_launched * n    # simplifies to w_std / sqrt(n) / n_lau * n
    sigma_P_dB  = (10.0 / np.log(10)) * sigma_P_lin / P_lin
    p_dB        = 10.0 * np.log10(P_lin)
    return p_dB - z * sigma_P_dB, p_dB + z * sigma_P_dB


def depth_bin_stats(
    df: pd.DataFrame,
    *,
    n_bins: int = 10,
    medium_name: Optional[str] = None,
    beam_name:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Bin captured photons by link_range_m and compute per-bin statistics.

    Useful for diagnosing where the capture rate drops off and how
    scattering statistics change with depth.

    Parameters
    ----------
    df          : full photon DataFrame
    n_bins      : number of depth bins
    medium_name : filter to this medium (None = all)
    beam_name   : filter to this beam (None = all)

    Returns
    -------
    DataFrame with columns:
      depth_bin (interval), n_captured, mean_weight, mean_n_scatters,
      mean_tof_ns, mean_excess_path_m
    """
    sub = df.copy()
    if medium_name:
        sub = sub[sub["medium_name"] == medium_name]
    if beam_name:
        sub = sub[sub["beam_name"] == beam_name]
    if sub.empty:
        return pd.DataFrame()

    sub["depth_bin"] = pd.cut(sub["link_range_m"], bins=n_bins)
    return (
        sub.groupby("depth_bin", observed=True)
           .agg(
               n_captured        = ("weight",        "count"),
               mean_weight       = ("weight",        "mean"),
               mean_n_scatters   = ("n_scatters",    "mean"),
               mean_tof_ns       = ("tof_ns",        "mean"),
               mean_excess_path_m = ("excess_path_m", "mean"),
           )
           .reset_index()
    )


def tof_histograms(
    df: pd.DataFrame,
    *,
    n_bins: int = 300,
    group_by: str = "link_range_m",
) -> pd.DataFrame:
    """
    Compute normalised ToF histograms for each group value.

    Returns a long-format DataFrame with columns:
      {group_by}, tof_bin_ns (float, bin centre), density (normalised)
    """
    rows = []
    for val, grp in df.groupby(group_by, observed=True):
        counts, edges = np.histogram(
            grp["tof_ns"].to_numpy(dtype=np.float64),
            bins=n_bins,
            weights=grp["weight"].to_numpy(dtype=np.float64),
            density=False,
        )
        total   = counts.sum() + 1e-30
        centres = 0.5 * (edges[:-1] + edges[1:])
        for c, v in zip(centres, counts / total):
            rows.append({group_by: val, "tof_bin_ns": float(c), "density": float(v)})
    return pd.DataFrame(rows)


def scattering_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(medium, range) mean and percentiles of n_scatters and excess_path_m.

    Shows how scattering load increases with depth — the primary cause of
    photon loss and delay spread growth.
    """
    return (
        df.groupby(["medium_name", "link_range_m"], observed=True)
          .agg(
              mean_scatters    = ("n_scatters",     "mean"),
              p50_scatters     = ("n_scatters",     lambda x: x.quantile(0.50)),
              p95_scatters     = ("n_scatters",     lambda x: x.quantile(0.95)),
              mean_excess_path = ("excess_path_m",  "mean"),
              p95_excess_path  = ("excess_path_m",  lambda x: x.quantile(0.95)),
              mean_tof_ns      = ("tof_ns",         "mean"),
              std_tof_ns       = ("tof_ns",         "std"),
          )
          .reset_index()
    )


def multi_run_aggregate(
    sweep_list: List[Dict[RunKey, RunResult]],
    *,
    seed_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combine results from multiple independent simulation runs (different seeds)
    into one DataFrame, with a 'seed_label' column for grouping.

    Use this to compute cross-seed statistics and confidence intervals
    without relying on the single-run delta-method approximation.

    Parameters
    ----------
    sweep_list   : list of sweep result dicts (one per seed)
    seed_labels  : optional list of string labels; defaults to "seed_0", ...
    """
    labels = seed_labels or [f"seed_{i}" for i in range(len(sweep_list))]
    frames = []
    photon_cursor = 0
    for label, sweep in zip(labels, sweep_list):
        df = to_dataframe(sweep)
        df["seed_label"] = label
        n = len(df)
        df["photon_id"] = np.arange(photon_cursor, photon_cursor + n, dtype=np.int64)
        frames.append(df)
        photon_cursor += n
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cross_seed_power_stats(
    multi_df: pd.DataFrame,
    launched_per_seed: int,
) -> pd.DataFrame:
    """
    Compute mean ± std of received power across seeds for each (medium, beam, range).

    Parameters
    ----------
    multi_df          : DataFrame from multi_run_aggregate()
    launched_per_seed : n_launched per seed (assumed equal across seeds)
    """
    rows = []
    for (medium, beam, Z), grp in multi_df.groupby(
        ["medium_name", "beam_name", "link_range_m"], observed=True
    ):
        per_seed_power = []
        for _, sg in grp.groupby("seed_label", observed=True):
            w_sum   = sg["weight"].sum()
            p_lin   = w_sum / launched_per_seed
            per_seed_power.append(10.0 * np.log10(p_lin + 1e-300))

        ps = np.array(per_seed_power)
        rows.append({
            "medium_name":   medium,
            "beam_name":     beam,
            "link_range_m":  Z,
            "n_seeds":       len(ps),
            "mean_power_dB": float(ps.mean()),
            "std_power_dB":  float(ps.std()) if len(ps) > 1 else float("nan"),
            "ci_95_low_dB":  float(ps.mean() - 1.96 * ps.std() / np.sqrt(len(ps)))
                             if len(ps) > 1 else float("nan"),
            "ci_95_high_dB": float(ps.mean() + 1.96 * ps.std() / np.sqrt(len(ps)))
                             if len(ps) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)