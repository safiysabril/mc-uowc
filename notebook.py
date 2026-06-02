import marimo

__generated_with = "0.23.8"
app = marimo.App(width="wide", app_title="UOWC Monte Carlo — Research Notebook")

# ─────────────────────────────────────────────────────────────────────────────
# Section 0 — Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    import os
    import time
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — Marimo renders figs as images
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pd
    return matplotlib, np, os, pd, plt, sys, ticker, time


@app.cell
def _(sys):
    # Ensure the project root (containing uowc/) is importable regardless of
    # where marimo is invoked from.
    if "." not in sys.path:
        sys.path.insert(0, ".")

    from uowc.config import (
        SimConfig,
        CLEAR_WATER, COASTAL_WATER, TURBID_WATER,
        COLLIMATED, DIFFUSED,
        ALL_WATERS, ALL_BEAMS, SIM,
    )
    from uowc.medium import (
        ALL_INHOMOGENEOUS_MEDIA,
        STRATIFIED_OCEAN, DEEP_OCEAN_COLUMN, COASTAL_GRADIENT,
    )
    from uowc.simulation import (
        RunKey,
        run_sweep_adaptive,
        run_sweep_inhomogeneous_adaptive,
    )
    from uowc.metrics import compute_all_metrics
    from uowc.analysis import (
        to_dataframe,
        capture_statistics_with_launched,
        tof_histograms,
    )

    # Convenience lookup dicts used by config widgets
    WATER_OPTIONS = {
        "Clear Water":   CLEAR_WATER,
        "Coastal Water": COASTAL_WATER,
        "Turbid Water":  TURBID_WATER,
    }
    BEAM_OPTIONS  = {b.name: b for b in ALL_BEAMS}
    MEDIA_OPTIONS = {m.name: m for m in ALL_INHOMOGENEOUS_MEDIA}

    return (
        ALL_BEAMS, ALL_INHOMOGENEOUS_MEDIA, ALL_WATERS,
        BEAM_OPTIONS, CLEAR_WATER, COASTAL_GRADIENT, COASTAL_WATER,
        COLLIMATED, DEEP_OCEAN_COLUMN, DIFFUSED, MEDIA_OPTIONS, SIM,
        STRATIFIED_OCEAN, SimConfig, TURBID_WATER, WATER_OPTIONS,
        RunKey, capture_statistics_with_launched, compute_all_metrics,
        run_sweep_adaptive, run_sweep_inhomogeneous_adaptive,
        to_dataframe, tof_histograms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md(r"""
    # 🌊 UOWC Monte Carlo — Research Notebook

    Interactive Monte Carlo photon transport simulation for Underwater Optical
    Wireless Communication. Configure parameters below, click **▶ Run**, then
    explore channel metrics and raw photon statistics.

    **Pipeline:** `SimConfig` → `run_sweep_adaptive` → `compute_all_metrics`
    → figures + `to_dataframe` → explorer
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Configuration
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("## ⚙️  Configuration")


@app.cell
def _(mo):
    preset = mo.ui.radio(
        options=["Quick Test (50 k photons)", "Full Run (1 M photons)"],
        value="Quick Test (50 k photons)",
        label="**Run preset**",
    )
    return (preset,)


@app.cell
def _(mo, preset):
    _is_quick = preset.value.startswith("Quick")
    n_photons_ui = mo.ui.number(
        value=50_000 if _is_quick else 1_000_000,
        start=10_000,
        stop=5_000_000,
        step=10_000,
        label="Photons per adaptive batch",
    )
    min_captured_ui = mo.ui.number(
        value=1_000 if _is_quick else 10_000,
        start=500,
        stop=100_000,
        step=500,
        label="Min captured photons (adaptive target)",
    )
    link_ranges_ui = mo.ui.multiselect(
        options=[5, 10, 15, 20, 25],
        value=[5, 10, 15, 20, 25],
        label="Link ranges (m)",
    )
    return link_ranges_ui, min_captured_ui, n_photons_ui


@app.cell
def _(mo, WATER_OPTIONS, BEAM_OPTIONS, MEDIA_OPTIONS):
    selected_waters_ui = mo.ui.multiselect(
        options=list(WATER_OPTIONS.keys()),
        value=["Clear Water", "Coastal Water"],
        label="**Homogeneous water types**",
    )
    selected_beams_ui = mo.ui.multiselect(
        options=list(BEAM_OPTIONS.keys()),
        value=list(BEAM_OPTIONS.keys()),
        label="**Beam types**",
    )
    include_inh_ui = mo.ui.checkbox(
        value=False,
        label="**Also run inhomogeneous media sweep**",
    )
    selected_media_ui = mo.ui.multiselect(
        options=list(MEDIA_OPTIONS.keys()),
        value=list(MEDIA_OPTIONS.keys()),
        label="**Inhomogeneous media profiles**",
    )
    return include_inh_ui, selected_beams_ui, selected_media_ui, selected_waters_ui


@app.cell
def _(
    mo, preset, n_photons_ui, min_captured_ui, link_ranges_ui,
    selected_waters_ui, selected_beams_ui, include_inh_ui, selected_media_ui,
):
    return mo.hstack(
        [
            mo.vstack([preset, n_photons_ui, min_captured_ui, link_ranges_ui]),
            mo.vstack([
                selected_waters_ui, selected_beams_ui,
                mo.md("---"),
                include_inh_ui, selected_media_ui,
            ]),
        ],
        gap=4,
    )


@app.cell
def _(SimConfig, SIM, os, n_photons_ui, min_captured_ui, link_ranges_ui):
    cfg = SimConfig(
        n_photons            = int(n_photons_ui.value),
        link_ranges_m        = tuple(sorted(int(r) for r in link_ranges_ui.value)),
        dt_bin_s             = SIM.dt_bin_s,
        n_time_bins          = SIM.n_time_bins,
        weight_threshold     = SIM.weight_threshold,
        roulette_m           = SIM.roulette_m,
        n_workers            = max(1, os.cpu_count() or 2),
        master_seed          = SIM.master_seed,
        chunk_size           = SIM.chunk_size,
        min_captured_photons = int(min_captured_ui.value),
        max_launched_photons = SIM.max_launched_photons,
    )
    return (cfg,)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Run Controls
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("## ▶  Run Simulation")


@app.cell
def _(mo):
    # on_click=lambda _: True makes .value become True after the first click.
    # Without on_click, mo.ui.button keeps .value as None and mo.stop never releases.
    run_btn = mo.ui.button(
        label="▶  Run Simulation",
        kind="success",
        on_click=lambda _: True,
    )
    return (run_btn,)


@app.cell
def _(mo, run_btn, cfg):
    return mo.hstack([
        run_btn,
        mo.callout(
            mo.md(
                f"**Config snapshot** — "
                f"{cfg.n_photons:,} photons/batch · "
                f"min {cfg.min_captured_photons:,} captured · "
                f"ranges {list(cfg.link_ranges_m)} m · "
                f"{cfg.n_workers} workers"
            ),
            kind="info",
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Simulation Execution
# run_btn.value is None on load; becomes True on first click.
# mo.stop(value is not True) halts this cell and all downstream cells until
# the button has been clicked, preventing accidental long runs on notebook load.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(
    mo, run_btn, cfg, time,
    selected_waters_ui, selected_beams_ui,
    include_inh_ui, selected_media_ui,
    WATER_OPTIONS, BEAM_OPTIONS, MEDIA_OPTIONS,
    run_sweep_adaptive, run_sweep_inhomogeneous_adaptive,
):
    mo.stop(
        run_btn.value is not True,
        mo.callout(
            mo.md("Configure parameters above, then click **▶ Run Simulation**."),
            kind="neutral",
        ),
    )

    _waters = [WATER_OPTIONS[n] for n in selected_waters_ui.value]
    _beams  = [BEAM_OPTIONS[n]  for n in selected_beams_ui.value]
    _media  = [MEDIA_OPTIONS[n] for n in selected_media_ui.value] \
              if include_inh_ui.value else []

    mo.stop(
        not _waters or not _beams,
        mo.callout(mo.md("⚠ Select at least one water type and one beam type."), kind="warn"),
    )
    mo.stop(
        not cfg.link_ranges_m,
        mo.callout(mo.md("⚠ Select at least one link range."), kind="warn"),
    )

    _t0 = time.perf_counter()

    raw_hom = run_sweep_adaptive(
        cfg, waters=tuple(_waters), beams=tuple(_beams), verbose=False,
    )
    raw_inh = (
        run_sweep_inhomogeneous_adaptive(
            cfg, media=tuple(_media), beams=tuple(_beams), verbose=False,
        )
        if _media else {}
    )

    _elapsed = time.perf_counter() - _t0
    sim_waters = _waters
    sim_beams  = _beams
    sim_media  = _media

    return (
        raw_hom, raw_inh, sim_beams, sim_media, sim_waters,
        mo.callout(
            mo.md(
                f"✅ Simulation complete in **{_elapsed:.1f} s**  — "
                f"{len(raw_hom)} homogeneous run(s)"
                + (f", {len(raw_inh)} inhomogeneous run(s)" if raw_inh else "")
            ),
            kind="success",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(cfg, RunKey, raw_hom, raw_inh, sim_waters, sim_beams, sim_media, compute_all_metrics):
    metrics_hom: dict = {}
    for _water in sim_waters:
        for _beam in sim_beams:
            for _Z in cfg.link_ranges_m:
                _key = RunKey(_water.name, _beam.name, float(_Z))
                metrics_hom[_key] = compute_all_metrics(raw_hom[_key], cfg, _water.c, _Z)

    metrics_inh: dict = {}
    for _medium in sim_media:
        for _beam in sim_beams:
            for _Z in cfg.link_ranges_m:
                _key = RunKey(_medium.name, _beam.name, float(_Z))
                metrics_inh[_key] = compute_all_metrics(raw_inh[_key], cfg, _medium.c_max, _Z)

    return metrics_hom, metrics_inh


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Channel Figures
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("## 📊  Channel Figures")


# ── Shared plot styling helpers — no underscore prefix so Marimo exports them ─

@app.cell
def _(plt, ticker):
    GRID_KW  = dict(alpha=0.25, linestyle="--")
    LINE_KW  = dict(linewidth=1.8, markersize=6)
    COLOURS  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    MARKERS  = ["o", "s", "D", "^", "v", "P"]

    def style_ax(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title,   fontsize=9, fontweight="bold")
        ax.grid(True, **GRID_KW)
        ax.tick_params(labelsize=8)

    def semilogy_finite(ax, x, y, **kw):
        """Plot only finite (non-NaN, non-inf, >0) points on a semilogy axis."""
        import numpy as _np
        pairs = [(xi, yi) for xi, yi in zip(x, y) if _np.isfinite(yi) and yi > 0]
        if pairs:
            xf, yf = zip(*pairs)
            ax.semilogy(list(xf), list(yf), **kw)

    # ticker imported only to satisfy Marimo dependency tracking
    _ = ticker
    return COLOURS, GRID_KW, LINE_KW, MARKERS, semilogy_finite, style_ax


# ── Fig 1 — Received Power ───────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Fig 1 — Received Power vs Link Range")


@app.cell
def _(
    plt, cfg, RunKey, metrics_hom, metrics_inh,
    sim_waters, sim_beams, sim_media,
    COLOURS, MARKERS, LINE_KW, style_ax,
):
    _ranges   = list(cfg.link_ranges_m)
    _n_panels = max(len(sim_waters), 1) + (1 if sim_media else 0)
    _fig1, _axes = plt.subplots(1, _n_panels, figsize=(6.5 * _n_panels, 5), squeeze=False)
    _fig1.suptitle("Normalised Received Power vs Link Range", fontsize=12, fontweight="bold")

    for _i, _water in enumerate(sim_waters):
        _ax = _axes[0, _i]
        for _j, _beam in enumerate(sim_beams):
            _pwr = [metrics_hom[RunKey(_water.name, _beam.name, float(Z))]["power_dB"]
                    for Z in _ranges]
            _bl  = [metrics_hom[RunKey(_water.name, _beam.name, float(Z))]["beer_lambert_dB"]
                    for Z in _ranges]
            _ax.plot(_ranges, _pwr, marker=MARKERS[_j], color=COLOURS[_j],
                     label=f"{_beam.name} (MC)", **LINE_KW)
            _ax.plot(_ranges, _bl, "--", color=COLOURS[_j], alpha=0.5,
                     label=f"{_beam.name} (Beer-Lambert)")
        style_ax(_ax, "Link Range (m)", "Normalised Power (dB)", _water.name)
        _ax.legend(fontsize=8, loc="lower left")

    if sim_media:
        _ax = _axes[0, len(sim_waters)]
        for _j, _medium in enumerate(sim_media):
            for _k, _beam in enumerate(sim_beams):
                _pwr = [metrics_inh[RunKey(_medium.name, _beam.name, float(Z))]["power_dB"]
                        for Z in _ranges]
                _ax.plot(_ranges, _pwr,
                         marker=MARKERS[_k], color=COLOURS[_j],
                         linestyle="-" if _k == 0 else "--",
                         label=f"{_medium.name[:20]} | {_beam.name[:12]}",
                         **LINE_KW)
        style_ax(_ax, "Link Range (m)", "Normalised Power (dB)", "Inhomogeneous Media")
        _ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    return _fig1


# ── Fig 2 — Channel Impulse Response ────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Fig 2 — Channel Impulse Response (CIR)")


@app.cell
def _(
    plt, cfg, RunKey, metrics_hom, metrics_inh,
    sim_waters, sim_beams, sim_media,
    COLOURS, style_ax,
):
    _ranges = list(cfg.link_ranges_m)
    _z_near, _z_far = _ranges[0], _ranges[-1]

    _entities = [(w.name, "hom") for w in sim_waters] + \
                [(m.name, "inh") for m in sim_media]
    _n_rows = max(len(_entities), 1)
    _n_cols = max(len(sim_beams) * 2, 1)
    _fig2, _axes = plt.subplots(_n_rows, _n_cols,
                                 figsize=(4.5 * _n_cols, 3.5 * _n_rows),
                                 squeeze=False)
    _fig2.suptitle(
        f"Channel Impulse Response  |  near={_z_near:.0f} m  far={_z_far:.0f} m",
        fontsize=12, fontweight="bold",
    )

    for _r, (_name, _kind) in enumerate(_entities):
        for _b, _beam in enumerate(sim_beams):
            for _col_off, _Z in enumerate([_z_near, _z_far]):
                _ax  = _axes[_r, _b * 2 + _col_off]
                _src = metrics_hom if _kind == "hom" else metrics_inh
                _key = RunKey(_name, _beam.name, float(_Z))
                if _key not in _src:
                    _ax.set_visible(False)
                    continue
                _m = _src[_key]
                _t = _m["t_axis"] * 1e9    # → ns
                _h = _m["cir"]
                _ax.fill_between(_t, _h, alpha=0.35, color=COLOURS[_b])
                _ax.plot(_t, _h, color=COLOURS[_b], linewidth=1.2)
                style_ax(
                    _ax, "ToF (ns)", "Normalised Weight",
                    f"{_name[:22]}  |  {_beam.name[:14]}  @{_Z:.0f} m",
                )

    plt.tight_layout()
    return _fig2


# ── Fig 3 — Frequency Response ───────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Fig 3 — Channel Frequency Response |H(f)|")


@app.cell
def _(
    plt, np, cfg, RunKey, metrics_hom, metrics_inh,
    sim_waters, sim_beams, sim_media,
    COLOURS, style_ax,
):
    _ranges = list(cfg.link_ranges_m)
    _z_near, _z_far = _ranges[0], _ranges[-1]

    _entities = [(w.name, "hom") for w in sim_waters] + \
                [(m.name, "inh") for m in sim_media]
    _n_rows = max(len(_entities), 1)
    _n_cols = max(len(sim_beams) * 2, 1)
    _fig3, _axes = plt.subplots(_n_rows, _n_cols,
                                 figsize=(4.5 * _n_cols, 3.5 * _n_rows),
                                 squeeze=False)
    _fig3.suptitle("Channel Frequency Response |H(f)|", fontsize=12, fontweight="bold")

    for _r, (_name, _kind) in enumerate(_entities):
        for _b, _beam in enumerate(sim_beams):
            for _col_off, _Z in enumerate([_z_near, _z_far]):
                _ax  = _axes[_r, _b * 2 + _col_off]
                _src = metrics_hom if _kind == "hom" else metrics_inh
                _key = RunKey(_name, _beam.name, float(_Z))
                if _key not in _src:
                    _ax.set_visible(False)
                    continue
                _m  = _src[_key]
                _f  = _m["freqs"] / 1e6    # → MHz
                _H  = _m["fr"]
                _bw = _m["bandwidth_hz"] / 1e6
                _ax.plot(_f, _H, color=COLOURS[_b], linewidth=1.4)
                _ax.axhline(1.0 / np.sqrt(2.0), color="grey", linestyle="--",
                            linewidth=1.0, label="−3 dB")
                _ax.axvline(_bw, color="red", linestyle=":", linewidth=1.0,
                            label=f"BW={_bw:.2f} MHz")
                _ax.set_xlim(left=0, right=max(10 * _bw, 1.0))
                style_ax(
                    _ax, "Frequency (MHz)", "|H(f)| (normalised)",
                    f"{_name[:22]}  |  {_beam.name[:14]}  @{_Z:.0f} m",
                )
                _ax.legend(fontsize=7)

    plt.tight_layout()
    return _fig3


# ── Fig 4 — RMS Delay Spread ─────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Fig 4 — RMS Delay Spread vs Link Range")


@app.cell
def _(
    plt, cfg, RunKey, metrics_hom, metrics_inh,
    sim_waters, sim_beams, sim_media,
    COLOURS, MARKERS, LINE_KW, semilogy_finite, style_ax, ticker,
):
    _ranges = list(cfg.link_ranges_m)
    _all_entities = [(w.name, "hom", i) for i, w in enumerate(sim_waters)] + \
                    [(m.name, "inh", i) for i, m in enumerate(sim_media)]
    _n_panels = max(len(sim_beams), 1)
    _fig4, _axes = plt.subplots(1, _n_panels, figsize=(7 * _n_panels, 5), squeeze=False)
    _fig4.suptitle("RMS Delay Spread vs Link Range", fontsize=12, fontweight="bold")

    for _b, _beam in enumerate(sim_beams):
        _ax = _axes[0, _b]
        for _name, _kind, _idx in _all_entities:
            _src = metrics_hom if _kind == "hom" else metrics_inh
            _ds  = [_src[RunKey(_name, _beam.name, float(Z))]["delay_spread_s"]
                    for Z in _ranges]
            _lbl = f"{_name[:28]} ({'hom' if _kind == 'hom' else 'inh'})"
            semilogy_finite(_ax, _ranges, _ds,
                            marker=MARKERS[_idx % len(MARKERS)],
                            color=COLOURS[_idx % len(COLOURS)],
                            label=_lbl, **LINE_KW)
        style_ax(_ax, "Link Range (m)", "RMS Delay Spread (s)", _beam.name)
        _ax.legend(fontsize=8)
        _ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1e}"))

    plt.tight_layout()
    return _fig4


# ── Fig 5 — 3 dB Bandwidth ───────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Fig 5 — 3 dB Channel Bandwidth vs Link Range")


@app.cell
def _(
    plt, cfg, RunKey, metrics_hom, metrics_inh,
    sim_waters, sim_beams, sim_media,
    COLOURS, MARKERS, LINE_KW, semilogy_finite, style_ax, ticker,
):
    _ranges = list(cfg.link_ranges_m)
    _all_entities = [(w.name, "hom", i) for i, w in enumerate(sim_waters)] + \
                    [(m.name, "inh", i) for i, m in enumerate(sim_media)]
    _n_panels = max(len(sim_beams), 1)
    _fig5, _axes = plt.subplots(1, _n_panels, figsize=(7 * _n_panels, 5), squeeze=False)
    _fig5.suptitle("3 dB Channel Bandwidth vs Link Range", fontsize=12, fontweight="bold")

    for _b, _beam in enumerate(sim_beams):
        _ax = _axes[0, _b]
        for _name, _kind, _idx in _all_entities:
            _src = metrics_hom if _kind == "hom" else metrics_inh
            _bw  = [_src[RunKey(_name, _beam.name, float(Z))]["bandwidth_hz"] / 1e6
                    for Z in _ranges]
            _lbl = f"{_name[:28]} ({'hom' if _kind == 'hom' else 'inh'})"
            semilogy_finite(_ax, _ranges, _bw,
                            marker=MARKERS[_idx % len(MARKERS)],
                            color=COLOURS[_idx % len(COLOURS)],
                            label=_lbl, **LINE_KW)
        style_ax(_ax, "Link Range (m)", "Bandwidth (MHz)", _beam.name)
        _ax.legend(fontsize=8)
        _ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))

    plt.tight_layout()
    return _fig5


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Metrics Summary Table
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("## 📋  Metrics Summary Table")


@app.cell
def _(mo, pd, np, cfg, RunKey, metrics_hom, metrics_inh, sim_waters, sim_beams, sim_media):
    _rows = []
    _all_combos = (
        [(w.name, b.name, "Homogeneous")   for w in sim_waters for b in sim_beams] +
        [(m.name, b.name, "Inhomogeneous") for m in sim_media  for b in sim_beams]
    )
    for _mname, _bname, _kind in _all_combos:
        _src = metrics_hom if _kind == "Homogeneous" else metrics_inh
        for _Z in cfg.link_ranges_m:
            _key = RunKey(_mname, _bname, float(_Z))
            if _key not in _src:
                continue
            _m = _src[_key]
            _rows.append({
                "Medium":            _mname,
                "Beam":              _bname,
                "Type":              _kind,
                "Range (m)":         int(_Z),
                "Power (dB)":        round(float(_m["power_dB"]),           2),
                "BL Power (dB)":     round(float(_m["beer_lambert_dB"]),     2),
                "Delay Spread (ns)": round(float(_m["delay_spread_s"]) * 1e9, 3)
                                     if np.isfinite(_m["delay_spread_s"]) else float("nan"),
                "BW (MHz)":          round(float(_m["bandwidth_hz"]) / 1e6,  3),
                "Captured":          int(_m["n_captured"]),
                "Launched":          int(_m["n_launched"]),
            })

    _df_metrics = pd.DataFrame(_rows)
    return mo.ui.table(_df_metrics, pagination=True, page_size=15)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Raw Photon Data Explorer
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("""
    ## 🔬  Raw Photon Data Explorer

    Per-photon DataFrame built from `to_dataframe()`.
    Filter by medium, beam, and link range to explore the raw photon
    statistics that underlie the channel metrics above.
    """)


@app.cell
def _(to_dataframe, raw_hom, raw_inh):
    _all_raw = {**raw_hom, **raw_inh}
    photon_df = to_dataframe(_all_raw)
    return (photon_df,)


@app.cell
def _(mo, photon_df):
    _media_opts = sorted(photon_df["medium_name"].unique().tolist())
    _beam_opts  = sorted(photon_df["beam_name"].unique().tolist())
    _range_opts = sorted(photon_df["link_range_m"].unique().tolist())

    filter_medium = mo.ui.multiselect(
        options=_media_opts, value=_media_opts[:1], label="**Medium**",
    )
    filter_beam = mo.ui.multiselect(
        options=_beam_opts, value=_beam_opts[:1], label="**Beam**",
    )
    filter_range = mo.ui.multiselect(
        options=_range_opts, value=_range_opts, label="**Link ranges (m)**",
    )
    return filter_beam, filter_medium, filter_range


@app.cell
def _(mo, filter_medium, filter_beam, filter_range):
    return mo.hstack([filter_medium, filter_beam, filter_range], gap=3)


@app.cell
def _(mo, photon_df, filter_medium, filter_beam, filter_range, np):
    _mask = (
        photon_df["medium_name"].isin(filter_medium.value) &
        photon_df["beam_name"].isin(filter_beam.value) &
        photon_df["link_range_m"].isin([np.float32(r) for r in filter_range.value])
    )
    filtered_df = photon_df[_mask].copy()
    return (
        filtered_df,
        mo.callout(
            mo.md(
                f"**{len(filtered_df):,}** photons selected "
                f"from {len(photon_df):,} total  "
                f"({100 * len(filtered_df) / max(len(photon_df), 1):.1f}%)"
            ),
            kind="info",
        ),
    )


# ── Explorer: Capture Statistics Table ───────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Capture Statistics")


@app.cell
def _(mo, filtered_df, raw_hom, raw_inh, capture_statistics_with_launched):
    _all_raw      = {**raw_hom, **raw_inh}
    _launched_map = {key: result.n_launched for key, result in _all_raw.items()}
    _stats        = capture_statistics_with_launched(filtered_df, _launched_map)
    return mo.ui.table(_stats.round(4), pagination=False)


# ── Explorer Plot 1 — Photon Weight Distribution ─────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Photon Weight Distribution")


@app.cell
def _(plt, np, filtered_df, style_ax):
    _ranges = sorted(filtered_df["link_range_m"].unique())
    _cmap   = plt.colormaps["plasma"].resampled(max(len(_ranges), 1))
    _fig_w, _ax = plt.subplots(figsize=(9, 4))

    for _i, _Z in enumerate(_ranges):
        _sub = filtered_df[filtered_df["link_range_m"] == _Z]
        if len(_sub) < 5:
            continue
        _w = _sub["weight"].to_numpy(dtype=np.float64)
        _w = _w[_w > 0]
        if _w.size < 5:
            continue
        _bins = np.geomspace(_w.min(), _w.max(), 60)
        _h, _e = np.histogram(_w, bins=_bins)
        _cx = 0.5 * (_e[:-1] + _e[1:])
        _ax.semilogx(_cx, _h / _h.max(),
                     color=_cmap(_i / max(len(_ranges) - 1, 1)),
                     linewidth=1.4, label=f"{_Z:.0f} m")

    style_ax(_ax, "Photon Weight (log scale)", "Normalised Count",
             "Weight Distribution by Link Range")
    _ax.legend(fontsize=8, title="Depth")
    plt.tight_layout()
    return _fig_w


# ── Explorer Plot 2 — Receiver Plane Spatial Distribution ────────────────────

@app.cell
def _(mo):
    return mo.md("### Receiver Plane Spatial Distribution")


@app.cell
def _(plt, np, filtered_df):
    _ranges = sorted(filtered_df["link_range_m"].unique())
    _n      = max(len(_ranges), 1)
    _fig_sp, _axes = plt.subplots(1, _n, figsize=(4 * _n, 4), squeeze=False)
    _fig_sp.suptitle("Photon Impact Position at Receiver Plane (x, y)", fontweight="bold")

    for _i, _Z in enumerate(_ranges):
        _ax  = _axes[0, _i]
        _sub = filtered_df[filtered_df["link_range_m"] == _Z]
        if len(_sub) < 5:
            _ax.set_title(f"{_Z:.0f} m — no data")
            continue
        _w = _sub["weight"].to_numpy(np.float64)
        _h, _xe, _ye = np.histogram2d(
            _sub["x_m"].to_numpy(np.float32),
            _sub["y_m"].to_numpy(np.float32),
            bins=60, weights=_w,
        )
        _ax.imshow(_h.T, origin="lower", aspect="equal",
                   extent=[_xe[0], _xe[-1], _ye[0], _ye[-1]], cmap="hot")
        _theta = np.linspace(0, 2 * np.pi, 200)
        _r_ap  = 0.1016 / 2
        _ax.plot(_r_ap * np.cos(_theta), _r_ap * np.sin(_theta),
                 "c--", linewidth=1.2, label="aperture")
        _ax.set_title(f"{_Z:.0f} m  ({len(_sub):,} photons)", fontsize=9)
        _ax.set_xlabel("x (m)", fontsize=8)
        _ax.set_ylabel("y (m)", fontsize=8)
        _ax.legend(fontsize=7)

    plt.tight_layout()
    return _fig_sp


# ── Explorer Plot 3 — ToF Distributions ─────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Time-of-Flight Distributions")


@app.cell
def _(plt, filtered_df, tof_histograms, style_ax):
    _ranges = sorted(filtered_df["link_range_m"].unique())
    _cmap   = plt.colormaps["viridis"].resampled(max(len(_ranges), 1))
    _fig_t, _ax = plt.subplots(figsize=(10, 4))

    for _i, _Z in enumerate(_ranges):
        _sub = filtered_df[filtered_df["link_range_m"] == _Z]
        if len(_sub) < 10:
            continue
        _tof_df = tof_histograms(_sub, n_bins=200, group_by="link_range_m")
        if _tof_df.empty:
            continue
        _g = _tof_df[_tof_df["link_range_m"] == _Z]
        _ax.plot(_g["tof_bin_ns"], _g["density"],
                 color=_cmap(_i / max(len(_ranges) - 1, 1)),
                 linewidth=1.4, label=f"{_Z:.0f} m")

    style_ax(_ax, "Time of Flight (ns)", "Normalised Weight Density",
             "ToF Histograms by Link Range")
    _ax.legend(fontsize=8, title="Depth")
    plt.tight_layout()
    return _fig_t


# ── Explorer Plot 4 — Scatter Count Histogram ────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Scattering Event Count vs Link Range")


@app.cell
def _(plt, filtered_df, style_ax):
    _ranges = sorted(filtered_df["link_range_m"].unique())
    _cmap   = plt.colormaps["plasma"].resampled(max(len(_ranges), 1))
    _fig_sc, _ax = plt.subplots(figsize=(9, 4))

    for _i, _Z in enumerate(_ranges):
        _sub = filtered_df[filtered_df["link_range_m"] == _Z]
        if len(_sub) < 5:
            continue
        _ns   = _sub["n_scatters"].to_numpy()
        _bins = range(0, int(_ns.max()) + 2)
        _ax.hist(_ns, bins=_bins, density=True, alpha=0.55,
                 color=_cmap(_i / max(len(_ranges) - 1, 1)),
                 label=f"{_Z:.0f} m", histtype="stepfilled")

    style_ax(_ax, "Number of Scattering Events", "Density",
             "Scattering Count Distribution by Link Range")
    _ax.legend(fontsize=8, title="Depth")
    plt.tight_layout()
    return _fig_sc


# ── Explorer Plot 5 — Excess Path vs Scatters ────────────────────────────────

@app.cell
def _(mo):
    return mo.md("### Excess Path Length vs Scatter Count")


@app.cell
def _(plt, filtered_df, style_ax):
    _ranges = sorted(filtered_df["link_range_m"].unique())
    _fig_ep, _ax = plt.subplots(figsize=(9, 4))

    for _i, _Z in enumerate(_ranges):
        _pool = filtered_df[filtered_df["link_range_m"] == _Z]
        _sub  = _pool.sample(min(2_000, len(_pool)), random_state=42)
        if len(_sub) < 5:
            continue
        _ax.scatter(
            _sub["n_scatters"], _sub["excess_path_m"],
            s=4, alpha=0.3,
            color=plt.colormaps["plasma"](_i / max(len(_ranges) - 1, 1)),
            label=f"{_Z:.0f} m",
        )

    style_ax(_ax, "Number of Scattering Events", "Excess Path Length (m)",
             "Excess Path Length vs Scatter Count  (2 k photon sample per range)")
    _ax.legend(fontsize=8, title="Depth", markerscale=4)
    plt.tight_layout()
    return _fig_ep


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    return mo.md("""
    ---
    **UOWC Monte Carlo** · Monte Carlo photon transport for underwater optical
    wireless communication ·
    Physics: HG scattering · Woodcock delta-tracking for inhomogeneous media ·
    Metrics: CIR · delay spread · 3 dB bandwidth · received power
    """)


if __name__ == "__main__":
    app.run()