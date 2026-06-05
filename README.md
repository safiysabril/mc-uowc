# monte-carlo-uowc

A modular **Monte Carlo photon-transport framework** for underwater optical wireless
communication (UOWC) channels. Photon packets are launched from a transmitter beam,
propagate through homogeneous or depth-stratified seawater, accumulate realistic
**absorption, scattering, optical turbulence, and Fresnel interface losses**, and are
collected at a finite-aperture receiver. From the captured photon ensemble the framework
derives the full set of channel descriptors used in UOWC link studies: the **channel
impulse response (CIR)**, **received power**, **RMS delay spread**, **3 dB bandwidth**,
**SNR/BER**, and turbulence coherence time — across configurable link ranges, beam
geometries, and water types.

The codebase is built around a strict **separation of concerns**: physics, transport,
media, turbulence, simulation control, metrics, data analysis, and plotting each live in
their own module and import only downward. A single shared transport kernel serves both the
homogeneous and the inhomogeneous (Woodcock delta-tracking) paths.

---

## Three comparison models

The framework is organised around **three scenarios** that share the same transport kernel
and differ only in the medium fed to it. This is the central design principle — each model
is a clean A/B/C of the previous one.

| Model | Mode (`main.py`) | Medium | IOPs | Turbulence | Transport |
|------:|------------------|--------|------|------------|-----------|
| **1** | `homogeneous`   | Constant water type            | Constant `a, b, c` | — | Beer-Lambert free flight |
| **2** | `inhomogeneous` | Layered / gradient / chlorophyll | Depth-varying `a(z), b(z), c(z)` | — | **Woodcock delta-tracking** |
| **3** | `turbulent`     | Model 2 IOP structure **+** ocean turbulence | Depth-varying `a(z), b(z), c(z)` | Nikishov `Cₙ²` (phase-screen kicks + scintillation) | Woodcock delta-tracking |

Each model writes its **own** Parquet file so the three scenarios never mix:
`photons_homogeneous.parquet`, `photons_inhomogeneous.parquet`, `photons_turbulent.parquet`.
Model 2 and Model 3 use distinct RNG sub-streams (seed offsets 1 and 2) so that, even
within a single `main.py all` invocation, they draw statistically independent photons.

> **Model 2 vs Model 3 as a true A/B:** the `KAMEDA_CHL_TURBULENT` (Model 3) medium reuses
> the *exact same* `KAMEDA_CHL_PROFILE` IOP object as Model 2 — the only difference between
> the two runs is the turbulence, so any change in the CIR/bandwidth is attributable solely
> to optical turbulence.

A full research-paper-style methodology for Model 3 lives in
[model3_methodology.txt](model3_methodology.txt).

---

## Features

- **Single shared transport kernel, two paths** — one `_propagate_chunk` loop handles both
  a homogeneous path (scalar `c`, standard Beer-Lambert free-flight sampling) and a Woodcock
  delta-tracking path (per-position `c(z)` with real/null acceptance), selected by whether a
  `MediumProfile` object is supplied.
- **Implicit absorption + Russian roulette** — photon weights are reduced by the
  single-scattering albedo `ω = b/c` at each real collision (MCML scheme); photons below the
  weight threshold (10⁻⁴) are terminated by the unbiased 1/m survival rule.
- **Henyey-Greenstein scattering** — analytic inverse-CDF sampling of the polar angle, with a
  scalar form for homogeneous media and a per-photon array form for depth-varying media.
- **Depth-varying media** — a `MediumProfile` interface with four concrete implementations:
  `HomogeneousMedium`, `LayeredMedium` (piecewise-constant slabs), `GradientMedium`
  (interpolated CTD/AC-9 profiles), and `ChlorophyllProfileMedium` (continuous IOPs derived
  from a **Kameda chlorophyll profile** via a Case-1 bio-optical model).
- **Optical turbulence** — Nikishov (2000) `Cₙ²` parameterisation from ocean parameters
  `(ε, χ_T)`; Markov-limit **phase-screen angular kicks** at every transport step;
  **scintillation fading** at the receiver (log-normal / Gamma-Gamma / negative-exponential),
  selected by the *path-integrated* Rytov variance `σ²_R`.
- **Coupled ocean layers** — `CoupledOceanMedium` bundles IOPs and turbulence into one depth
  grid so the boundaries are always aligned; `TurbulentChlorophyllMedium` composes a
  continuous chlorophyll IOP profile with depth-uniform turbulence.
- **Fresnel transmittance** — exact two-polarisation Fresnel equations at the water → glass
  receiver window.
- **Publication-quality CIR via weighted KDE** — the displayed CIR is a weighted Gaussian
  kernel density estimate with an outlier-robust bandwidth and time window, so it stays smooth
  and the ballistic peak stays visible even at modest photon counts.
- **Smoothing-independent bandwidth** — the 3 dB bandwidth is read from a *direct DTFT* of the
  raw weighted arrival train, not from the FFT of the smoothed CIR, so it tracks the true
  delay spread instead of saturating on near-ballistic channels.
- **Adaptive multi-metric convergence** — a batch-means relative-standard-error stopping rule
  with configurable per-metric tolerances (power, delay spread, bandwidth, CIR shape,
  frequency response); extensible via a metric registry.
- **Parallel execution** — `ProcessPoolExecutor` sweeps over (medium, beam, link-range)
  combinations using independent seeded RNG sub-streams.
- **Pandas + Parquet pipeline** — one row per captured photon, self-contained Parquet files
  (carrying `n_launched` and `c_ref`), automatic accumulation across runs, and CSV/`.npz`
  exports for paper tables.
- **Interactive notebook** — a Marimo notebook for live parameter exploration (Models 1–2).

---

## Installation

**With [uv](https://github.com/astral-sh/uv) (recommended):**

```bash
git clone https://github.com/safiysabril/mc-uowc.git
cd mc-uowc
uv sync
```

**With pip:**

```bash
git clone https://github.com/safiysabril/mc-uowc.git
cd mc-uowc
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

**Requirements** (from [pyproject.toml](pyproject.toml)): Python ≥ 3.12, NumPy ≥ 2.4,
SciPy ≥ 1.17, Pandas ≥ 3.0, Matplotlib ≥ 3.10, PyArrow ≥ 24.0, Marimo ≥ 0.23.

---

## Quick start

The workflow is a **three-stage pipeline**: simulate → (optionally) export metrics → plot.
Simulation and plotting are deliberately separate so you can accumulate photons over several
runs before drawing figures.

```bash
# Stage 1 — run the simulation (accumulates photons into Parquet on every call)
python main.py all --out ./outputs          # Models 1, 2 and 3
#   python main.py homogeneous --out ./outputs   # Model 1 only
#   python main.py inhomogeneous --out ./outputs # Model 2 only
#   python main.py turbulent --out ./outputs     # Model 3 only

# Re-run to accumulate more photons for sparse (long-range / turbid) combinations
python main.py all --out ./outputs
python main.py all --out ./outputs          # keep going until counts look "good"

# Stage 2 — export scalar metrics to CSV and CIR/FR arrays to .npz (paper tables)
python metrics.py --out ./outputs --arrays

# Stage 3 — generate all figures from the accumulated Parquet
python plot.py --out ./outputs
```

After each `main.py` run a **photon-count table** is printed with a quality label
(`good` / `sparse` / `very sparse` / `TOO FEW`) per (medium, beam, range). Run `main.py`
as many times as needed before invoking `metrics.py` or `plot.py`. Each run automatically
draws a **fresh random seed** (printed for reproducibility), so repeated runs add
independent photons rather than duplicates.

### Running from Python

```python
from uowc.config import SIM, CLEAR_WATER
from uowc.metrics import compute_all_metrics
from uowc.simulation import run_sweep_adaptive, RunKey
from uowc.analysis import to_dataframe, append_to_parquet

# Model 1 — adaptive sweep over all preset water types and beam geometries
results = run_sweep_adaptive(SIM, verbose=True)

# Inspect one combination
key     = RunKey("Clear Water", "Collimated (Laser)", 10.0)
metrics = compute_all_metrics(results[key], SIM, CLEAR_WATER.c, link_range=10.0)
print(f"Received power  : {metrics['power_dB']:.2f} dB")
print(f"RMS delay spread: {metrics['delay_spread_s']*1e9:.3f} ns")
print(f"3 dB bandwidth  : {metrics['bandwidth_hz']/1e6:.1f} MHz")

# Accumulate photons into Parquet (safe to call repeatedly; n_launched is summed)
c_ref_map = {RunKey("Clear Water", "Collimated (Laser)", float(Z)): CLEAR_WATER.c
             for Z in SIM.link_ranges_m}
df = to_dataframe(results, c_ref_map=c_ref_map)
append_to_parquet(df, "./outputs/photons_homogeneous.parquet")
```

```python
# Model 2 / Model 3 — Woodcock sweep over inhomogeneous media
from uowc.simulation import run_sweep_inhomogeneous_adaptive
from uowc.medium import ALL_INHOMOGENEOUS_MEDIA          # Model 2 media (4)
from uowc.turbulence import ALL_COUPLED_MEDIA            # Model 3 media (5)

raw_m2 = run_sweep_inhomogeneous_adaptive(SIM, media=ALL_INHOMOGENEOUS_MEDIA, seed_offset=1)
raw_m3 = run_sweep_inhomogeneous_adaptive(SIM, media=ALL_COUPLED_MEDIA,       seed_offset=2)
# Turbulence is decided per medium automatically: plain media run turbulence-free,
# CoupledOceanMedium / TurbulentChlorophyllMedium run with kicks + scintillation.
```

---

## CLI reference

### `main.py` — simulation only (no figures)

Runs photon transport, computes metrics, prints console summary tables, and **accumulates**
per-photon data into Parquet. No figures are generated here.

| Command | Description |
|---|---|
| `python main.py all` | Run all three models (Model 1 + Model 2 + Model 3) |
| `python main.py homogeneous` | **Model 1** — constant IOPs, no turbulence |
| `python main.py inhomogeneous` | **Model 2** — layered/gradient/chlorophyll IOPs via Woodcock, no turbulence |
| `python main.py turbulent` | **Model 3** — Model 2 IOP structure + ocean turbulence |
| `--out PATH` | Output directory (default `./outputs`); Parquet is appended on every run |
| `--seed INT` | Master RNG seed. Omit for a fresh random seed per run (printed). Reuse the *same* seed only to reproduce a run exactly — **never** reuse a seed across accumulation runs (it stacks duplicate photons → false convergence). |

### `metrics.py` — compute metrics and export for paper analysis

```bash
python metrics.py --out ./outputs                 # writes metrics_{model}.csv
python metrics.py --out ./outputs --arrays        # also writes cir_{model}.npz
python metrics.py --sigma-r2 0.15                 # OOK BER with weak turbulence fading
python metrics.py --noise-power 1e-8 --responsivity 0.6
python metrics.py --hom outputs/photons_homogeneous.parquet   # explicit path
```

| Flag | Default | Description |
|---|---|---|
| `--out PATH` | `./outputs` | Directory with Parquet inputs and where CSV/`.npz` are written |
| `--hom / --inh / --turb PATH` | auto | Explicit Parquet path for Model 1 / 2 / 3 |
| `--arrays` | off | Also save CIR & frequency-response arrays as compressed `.npz` |
| `--noise-power W²` | `1e-7` | Receiver noise variance σ²_noise (A²) for SNR/BER |
| `--responsivity A/W` | `0.5` | Photodetector responsivity R for SNR/BER |
| `--sigma-r2 σ²_R` | `0.0` | Rytov variance for turbulence-fading BER (0 = AWGN only) |

Outputs per model: `metrics_homogeneous.csv`, `metrics_inhomogeneous.csv`,
`metrics_turbulent.csv` (and `cir_*.npz` with `--arrays`). CSV columns:

| Column | Description |
|---|---|
| `power_dB` | MC received power (dB, normalised to launched power) |
| `beer_lambert_dB` | Beer-Lambert reference power (dB) |
| `power_excess_dB` | MC − Beer-Lambert (scattering gain/loss) |
| `delay_spread_ns` | RMS delay spread (ns) |
| `bandwidth_mhz` | 3 dB bandwidth (MHz) |
| `snr_db` | Electrical SNR (dB) — depends on `--noise-power`, `--responsivity` |
| `ber_ook` | OOK BER; AWGN unless `--sigma-r2` is given |
| `capture_rate` | n_captured / n_launched |
| `mean_n_scatters`, `std_n_scatters` | Real-collision count statistics per captured photon |
| `mean_excess_path_m`, `std_excess_path_m` | Extra path beyond straight-line (m) |
| `mean_r_m` | Mean radial offset at receiver plane (m) |
| `n_launched`, `n_captured`, `cir_quality` | Counts and quality label |

The `.npz` file stores per-run arrays (`t_axis_ns_0`, `cir_0`, `freqs_mhz_0`, `fr_0`,
`label_0`, …) keyed by run index, readable with `np.load(..., allow_pickle=True)`.

### `plot.py` — standalone figure regeneration (no simulation)

```bash
python plot.py --out ./outputs                    # plot every model found in --out
python plot.py --hom outputs/photons_homogeneous.parquet
python plot.py --inh outputs/photons_inhomogeneous.parquet
python plot.py --turb outputs/photons_turbulent.parquet     # Model 3 → outputs/turbulent/

# Merge several Parquet files first, then plot the merged result
python plot.py --merge run1/photons_homogeneous.parquet \
                       run2/photons_homogeneous.parquet \
               --merge-out merged/photons_homogeneous.parquet \
               --out merged
```

| Flag | Description |
|---|---|
| `--out PATH` | Directory with Parquet inputs and where figures are written (default `./outputs`) |
| `--hom / --inh / --turb PATH` | Explicit Parquet path for Model 1 / 2 / 3 |
| `--merge PATH...` | Two or more Parquet files to merge before plotting (requires `--merge-out`) |
| `--merge-out PATH` | Destination for the merged Parquet file |

Model 1 figures are written to `--out`, Model 2 to `--out`, and **Model 3 to a
`turbulent/` subdirectory** (Models 2 and 3 share the `fig_inh*` figure family, so Model 3
gets its own directory to avoid overwriting Model 2).

---

## Module structure

```
uowc/
├── config/        Physical constants, preset water types, beam/receiver geometry, SimConfig
├── physics/       Pure photon optics: step sampling, absorption, HG, MCML rotation, Fresnel,
│                  ToF, and the Case-1 chlorophyll → IOP bio-optical model
├── medium/        MediumProfile interface + HomogeneousMedium, LayeredMedium, GradientMedium,
│                  ChlorophyllProfileMedium (Kameda DCM) + inhomogeneous presets
├── turbulence/    TurbulenceProfile interface, Nikishov Cₙ², phase screens, scintillation,
│                  CoupledOceanMedium and TurbulentChlorophyllMedium + turbulence presets
├── transport/     _propagate_chunk loop + worker functions (picklable for ProcessPoolExecutor);
│                  owns the MCML / Woodcock stepping logic
├── simulation/    Parallel sweep orchestration (run_sweep_adaptive, …)
│   └── convergence.py   Multi-metric batch-means convergence framework + metric registry
├── metrics/       Channel metrics from RunResult: power, KDE CIR, delay spread, direct-DTFT
│                  frequency response, 3 dB bandwidth, OOK BER, SNR, coherence time
├── analysis/      Pandas DataFrame pipeline, Parquet/CSV export, accumulation/merge utilities
│   └── plots.py   Diagnostic figures (capture rate, ToF, scatter profile, spatial spread, …)
├── plotting/      Publication channel figures for all metrics (homogeneous + inhomogeneous)
├── reporting.py   Console summary tables
└── domain/        Reserved SoC scaffolding (not wired into the live pipeline)

main.py            Simulation entry point — simulate, summarise, accumulate Parquet (no figures)
metrics.py         Export scalar metrics → CSV and CIR/FR arrays → .npz (no simulation/figures)
plot.py            Figure generation from accumulated Parquet (no simulation required)
notebook.py        Interactive Marimo notebook (Models 1–2)
model3_methodology.txt   Research-paper-style methodology for Model 3
```

### Separation of concerns

Each module owns one domain and imports only downward — no cycles:

```
config
  ↑
physics ───────────────► metrics
  ↑                         ▲
medium                      │
  ↑                         │
turbulence                  │
  ↑                         │
transport                   │
  ↑                         │
simulation ─(convergence reuses metrics)
  ↑
analysis ──► plotting,  analysis.plots
            (reporting depends on config + simulation)
```

- **physics** — stateless functions; no awareness of receivers, files, or parallelism.
- **medium** — answers "what are the IOPs at depth z?"; imports only `config` + `physics`.
- **turbulence** — all turbulence physics; composes `medium` for the chlorophyll model.
- **transport** — advances photons; calls `physics` and `turbulence`; treats the medium
  through the duck-typed `MediumProfile` interface. Returns `PhotonRecord` dicts.
- **simulation** — distributes batches to workers, applies the convergence stopping rule.
- **metrics** — pure NumPy on `RunResult`; no transport or I/O.
- **analysis** — the Pandas / Parquet layer; no physics.
- **plotting / reporting** — rendering only; consume pre-computed metric dicts.

---

## Preset configurations

### Water types at 530 nm (Gabriel et al. 2013 / Petzold 1972)

| Name | c (m⁻¹) | a (m⁻¹) | b (m⁻¹) | g | ω = b/c |
|---|---|---|---|---|---|
| Clear Water | 0.241 | 0.151 | 0.090 | 0.924 | 0.373 |
| Coastal Water | 0.775 | 0.220 | 0.555 | 0.924 | 0.716 |
| Turbid Water | 2.190 | 0.366 | 1.824 | 0.945 | 0.833 |

### Beam geometries

| Name | Half-angle | Waist |
|---|---|---|
| Collimated (Laser) | 1.5 mrad | 1 mm |
| Diffused (LED) | 15° | 1 mm |

**Receiver:** 4-inch aperture (radius 50.8 mm), FOV = 180° (full hemisphere).
**Default link ranges:** 5, 10, 15, 20, 25 m.

### Model 2 — inhomogeneous media (`ALL_INHOMOGENEOUS_MEDIA`)

| Name | Kind | Structure | c_max (m⁻¹) |
|---|---|---|---|
| `STRATIFIED_OCEAN` | Layered | Clear (0–10 m) → Coastal (10+ m) | 0.775 |
| `DEEP_OCEAN_COLUMN` | Layered | Clear (0–8) → Coastal (8–18) → Turbid (18+ m) | 2.190 |
| `COASTAL_GRADIENT` | Gradient | Smooth c from 0.241 → 2.19 m⁻¹ (7 samples, 0–25 m) | 2.190 |
| `KAMEDA_CHL_PROFILE` | Chlorophyll | Continuous Case-1 IOPs from a Kameda DCM @ 12 m | 0.277 |

### Model 3 — coupled IOP + turbulence media (`ALL_COUPLED_MEDIA`)

| Name | ε (m²/s³) | χ_T (K²/s) | Cₙ² (m⁻²ᐟ³) | Regime |
|---|---|---|---|---|
| `CALM_OPEN_OCEAN` | 10⁻⁹ | 8.5×10⁻¹² | ~10⁻¹⁵ | Weak |
| `COASTAL_SURFACE` | 10⁻⁷ | 3.95×10⁻⁹ | ~10⁻¹³ | Weak–moderate |
| `STRATIFIED_THERMOCLINE` | 10⁻⁹ / 10⁻⁵ | 8.5×10⁻¹² / 9.16×10⁻⁷ | 10⁻¹⁵ / 5×10⁻¹² | Weak / Moderate |
| `DEEP_OCEAN_STRATIFIED` | 10⁻⁹ / 10⁻⁵ / 10⁻⁸ | layered | 10⁻¹⁵ / 5×10⁻¹² / 10⁻¹⁴ | Mixed |
| `KAMEDA_CHL_TURBULENT` | 10⁻⁷ | 3.95×10⁻⁹ | ~10⁻¹³ | Weak (σ²_R ≈ 0.008 @ 25 m) |

The first four couple `WaterParams` and `(ε, χ_T)` per depth slab via `CoupledOceanMedium`.
The last composes the continuous `KAMEDA_CHL_PROFILE` IOPs with depth-uniform turbulence via
`TurbulentChlorophyllMedium`, so Model 3's chlorophyll case is the same optics as Model 2's
with turbulence layered on top.

---

## Bio-optical chlorophyll model (Kameda + Case-1)

The `ChlorophyllProfileMedium` computes IOPs **continuously** from a depth-dependent
chlorophyll profile — no layers, no sampling — evaluating `c(z)` exactly at each photon's
position. It is the "functional" medium style, complementing `LayeredMedium`'s slabs and
`GradientMedium`'s interpolation.

**Step 1 — chlorophyll vs depth (Kameda DCM):** a constant background plus a Gaussian deep
chlorophyll maximum,

```
C(z) = C_b + h · exp[ −((z − z_m) / (√2 · σ))² ]      [mg m⁻³]
```

The `KAMEDA_CHL_PROFILE` preset uses C_b = 0.05, h = 0.50, z_m = 12 m, σ = 5 m
(oligotrophic open ocean with a DCM near 12 m).

**Step 2 — chlorophyll → IOPs (Case-1, 530 nm):**

```
a(C) = a_w + A_φ · C^E_φ                       (water + phytoplankton absorption)
b(C) = b_w + b_p550 · (550/λ) · C^0.62          (water + particulate scattering)
c(z) = a(C(z)) + b(C(z))
```

with `a_w = 0.0430`, `b_w = 0.0019`, `A_φ = 0.0180`, `E_φ = 0.650`, `b_p550 = 0.300`,
`λ = 530 nm`. For the preset this gives `c` ranging from ≈ 0.097 m⁻¹ (background) to
≈ 0.272 m⁻¹ (DCM peak), with majorant `c_max ≈ 0.277`. The asymmetry `g` is held constant
(0.924). The 530 nm constants are isolated in [uowc/physics/__init__.py](uowc/physics/__init__.py)
so they can be recalibrated in one place.

---

## Key algorithms

### Free-flight step sampling

```
s = −ln(ξ) / c          (homogeneous, Beer-Lambert inverse-CDF)
s = −ln(ξ) / c_max      (inhomogeneous, Woodcock majorant)
```

### Woodcock delta-tracking

Steps are sampled with the global majorant `c_max ≥ c(z)`. The collision at the new depth is
accepted as **real** with probability `c(z)/c_max`; rejected collisions are **null** — the
photon continues with no weight change and no scatter. By the thinning theorem of Poisson
processes, the sequence of real collisions reproduces the correct inhomogeneous Poisson
process with local rate `c(z)`, with no need to compute ray–boundary intersections. This is
what lets layered, gradient, and *continuous* (Kameda) media share one transport path.

### Implicit absorption

```
w ← w × ω(z)     where ω = b/c  (single-scattering albedo)
```

Applied in-place to the full weight array via integer-index assignment `w[active] *= ω`,
so NumPy fancy-index semantics never silently operate on a copy. Only **real** collisions
update the weight; null collisions leave it untouched.

### Henyey-Greenstein phase function

```
cos θ = 1/(2g) · [1 + g² − ((1−g²)/(1−g+2gξ))²]    g ≠ 0
cos θ = 1 − 2ξ                                          g = 0
```

### Direction rotation (MCML)

```
ux' = sinθ (ux uz cosφ − uy sinφ) / √(1−uz²) + ux cosθ
uy' = sinθ (uy uz cosφ + ux sinφ) / √(1−uz²) + uy cosθ
uz' = −sinθ cosφ √(1−uz²) + uz cosθ
```

The near-z singularity (|uz| > 1 − 10⁻⁵) is handled analytically.

### Nikishov turbulence parameterisation

```
Cₙ²(z) = B₀ · (dn/dT)² · ε(z)^(−1/3) · χ_T(z)
```

with `B₀ ≈ 3.63` (Kolmogorov-Corrsin structural constant) and `dn/dT = −1.80×10⁻⁴ K⁻¹` at
530 nm. The combined prefactor `B₀·(dn/dT)² = 1.176×10⁻⁷` matches published UOWC literature;
`dn_dT` is configurable for other wavelengths/temperatures (Quan & Fry 1995).

### Phase-screen angular kicks

```
σ²_α = 3.04 · l₀^(−1/3) · Cₙ²(z) · Δz      [rad²]
```

A Gaussian transverse kick (variance split equally between the two transverse axes) is drawn
at every transport step in the Markov continuous-random-walk limit and applied to the
direction vector, modelling beam wander and small-scale spreading along the path.

### Scintillation fading

At the receiver, each captured photon weight is multiplied by a fading factor `h_t`
(`E[h_t] = 1`). The regime is selected by the **path-integrated** Rytov variance,

```
σ²_R = 1.23 · k^(7/6) · L^(5/6) · ∫₀^L Cₙ²(z) dz
```

| σ²_R | Distribution | Notes |
|---|---|---|
| < 0.3 | Log-normal | weak turbulence |
| 0.3 – 5 | Gamma-Gamma (Andrews & Phillips 2001) | α, β from σ²_X, σ²_Y |
| ≥ 5 | Negative-exponential | fully developed speckle |

Using the path integral (rather than the local `Cₙ²` at the receiver plane) assigns the
correct regime on stratified links.

### Fresnel transmittance

Exact two-polarisation Fresnel equations are applied when photons reach the receiver,
modelling the water (n = 1.33) → glass window (n = 1.50) interface. At normal incidence
T ≈ 99.6%.

### Channel impulse response — weighted Gaussian KDE

The CIR is estimated as a **weighted Gaussian KDE** of the excess-delay distribution rather
than a histogram, so it stays smooth at low photon counts. Two robustness features keep the
sharp ballistic peak visible:

- **Bandwidth** — Silverman's robust rule using the Kish effective sample size
  `N_eff = (Σwᵢ)²/Σwᵢ²` and the *core* spread `min(σ_τ, IQR/1.349)`, immune to far-tail
  outliers.
- **Time window** — the grid runs from 0 to a weighted high-quantile (default 99.5%) of the
  delay energy plus a few bandwidths, so a handful of heavily-scattered late photons cannot
  stretch the x-axis and bury the peak.

The KDE uses a reflection boundary at τ = 0 (one-sided density), keeping the leading edge
sharp and conserving energy. Delay spread and received power are computed from the **raw**
weights/times and are unaffected by this smoothing.

### Frequency response & 3 dB bandwidth — direct DTFT

The frequency response is the exact discrete-time Fourier transform of the weighted arrival
train,

```
H(f) = Σ wᵢ · e^(−j2πf τᵢ) / Σ wᵢ ,     τᵢ = excess delay
```

evaluated on a grid scaled to the RMS delay spread (`f_max ∝ 1/τ_rms`). Because it uses no
histogram bin and no kernel, the bandwidth it yields is independent of any display smoothing
and correctly decreases with range — unlike the FFT of the smoothed CIR, whose cut-off would
saturate on near-ballistic channels. The 3 dB bandwidth is the linearly interpolated
frequency where |H(f)| first drops below 1/√2.

---

## Outputs

| Artifact | Location | Description |
|---|---|---|
| Console tables | stdout | Received power, delay spread, bandwidth per run |
| `fig1_received_power.png` … `fig5_bandwidth.png` | `outputs/` | **Model 1** channel figures |
| `fig_inh1_…png` … `fig_inh5_…png` | `outputs/` | **Model 2** channel figures |
| `fig_inh1_…png` … `fig_inh5_…png` | `outputs/turbulent/` | **Model 3** channel figures |
| `diagnostics/fig_diag1…6.png` | each figure dir | Capture rate, power+CI, ToF, scatter profile, spatial spread, excess path |
| `photons_homogeneous.parquet` | `outputs/` | Per-photon data, Model 1 |
| `photons_inhomogeneous.parquet` | `outputs/` | Per-photon data, Model 2 (Woodcock) |
| `photons_turbulent.parquet` | `outputs/` | Per-photon data, Model 3 (Woodcock + turbulence) |
| `metrics_{model}.csv` | `outputs/` | Scalar metrics, one row per (medium, beam, range) — from `metrics.py` |
| `cir_{model}.npz` | `outputs/` | CIR & frequency-response arrays — from `metrics.py --arrays` |

The five channel figures per model are: `1` received power, `2` CIR, `3` frequency response,
`4` RMS delay spread, `5` 3 dB bandwidth (Model 1 uses the `figN_` prefix; Models 2 & 3 use
`fig_inhN_`). CIR and frequency-response subplots carry a **photon-count quality badge**.

### Per-photon DataFrame schema

| Column | Type | Description |
|---|---|---|
| `photon_id` | int64 | Global unique ID across the sweep |
| `run_id` | int32 | Index of the (medium, beam, range) combination |
| `medium_name`, `beam_name` | category | Run metadata |
| `link_range_m` | float32 | Target receiver depth (m) |
| `weight` | float64 | Final photon weight after absorption (+ fading, Model 3) |
| `tof_s` / `tof_ns` | float64 | Absolute time of flight (s / ns) |
| `x_m`, `y_m`, `r_m` | float32 | Position / radius at capture plane |
| `path_length_m` | float32 | Total geometric path length |
| `n_scatters` | int32 | Real scattering events |
| `n_nulls` | int32 | Null (Woodcock virtual) collisions — 0 for Model 1 |
| `excess_path_m` | float32 | Extra path beyond straight-line distance |
| `n_launched` | int64 | Total photons launched for this run — power-normalisation denominator |
| `c_ref` | float32 | Beam-attenuation coefficient used for the Beer-Lambert reference |

`tof_s` is the **absolute** time of flight (`path_length / C_medium`, ~111 ns at 25 m). When
computing the CIR the code converts to **excess delay** (subtracting the first arrival) so the
response always starts at 0 regardless of range. The `n_launched` and `c_ref` columns make
each Parquet file fully self-contained: figures and metrics can be regenerated without the
original simulation objects.

---

## Accumulating photons for sparse runs

At long range in turbid water a single run may capture too few photons for a clean CIR or a
reliable bandwidth. The pipeline handles this with **automatic accumulation**: each
`main.py` run with the same `--out` directory **merges** new photons into the existing
Parquet file instead of overwriting it, and **sums `n_launched`** per (medium, beam, range)
so power normalisation stays exact.

```bash
# Each run adds to the same file — run as many times as needed
python main.py all --out ./outputs
python main.py all --out ./outputs
python main.py all --out ./outputs

# Then regenerate figures from the accumulated data
python plot.py --out ./outputs
```

Every run uses a fresh random seed by default, so accumulation adds **independent** photons.
To start fresh, delete the Parquet file first:

```bash
rm ./outputs/photons_*.parquet
python main.py all --out ./outputs
```

You can also merge Parquet files from different directories with `plot.py`:

```bash
python plot.py --merge run1/photons_homogeneous.parquet \
                       run2/photons_homogeneous.parquet \
                       run3/photons_homogeneous.parquet \
               --merge-out merged/photons_homogeneous.parquet \
               --out merged
```

### CIR photon-quality thresholds

| Captured photons | Quality | Action |
|---|---|---|
| ≥ 2 000 | good | None |
| 250 – 1 999 | sparse | CIR shape usable; bandwidth approximate |
| 50 – 249 | very sparse | Run 2–3 more times and accumulate |
| < 50 | TOO FEW | CIR is noise — accumulate more runs |

---

## Adaptive convergence

The adaptive sweep launches batches until **all required** metrics satisfy a batch-means
relative-standard-error criterion. Each batch is an independent RNG sub-stream, so per-batch
estimates are i.i.d. draws of the metric:

```
mean      = (1/n) Σ mᵢ
SE        = std(mᵢ, ddof=1) / √n
rel_error = ‖SE‖₂ / ‖mean‖₂  <  tolerance      (and n ≥ min_conv_batches)
```

For scalar metrics this is the familiar `SE/|mean|`; for vector metrics (CIR, frequency
response) it is the relative L2 size of the shape uncertainty, so near-empty tail bins do not
dominate. Recognised metric names: `power`, `delay_spread`, `bandwidth`, `cir`,
`frequency_response` — and you can register your own.

Configure via `SimConfig`:

```python
from uowc.config import SimConfig

cfg = SimConfig(
    n_photons            = 1_000_000,   # batch size for the non-adaptive run_one / run_sweep
    link_ranges_m        = (5, 10, 15, 20, 25),
    dt_bin_s             = 1e-11,
    n_time_bins          = 3000,
    weight_threshold     = 1e-4,
    roulette_m           = 10,
    n_workers            = 8,
    master_seed          = 42,
    chunk_size           = 10_000,
    min_captured_photons = 10_000,
    max_launched_photons = 1_000_000_000,   # emergency cap for near-zero capture rates
    conv_batch_photons   = 5_000_000,       # photons per adaptive round
    conv_metrics         = ("power", "delay_spread"),   # ALL must converge to stop
    rel_error_tol        = 0.05,            # default per-metric tolerance
    conv_tols            = (("delay_spread", 0.05),),   # per-metric overrides
    min_conv_batches     = 3,
    # CIR / FR display knobs:
    cir_kde_bw_scale     = 1.0,   # >1 smooths more, <1 sharpens
    cir_n_grid           = 512,   # delay-grid resolution
    cir_tail_quantile    = 0.995, # outlier-robust CIR time window
)
```

> In the adaptive sweep (what `main.py` uses), each round launches `conv_batch_photons`
> photons; `n_photons` is the per-run count for the non-adaptive `run_one` / `run_sweep`
> helpers. `max_launched_photons` is only an emergency safeguard for combinations whose
> capture rate is so low a metric never reaches its tolerance.

---

## Extending the framework

### Custom water type

```python
from uowc.config import WaterParams
from uowc.medium import HomogeneousMedium

my_water = WaterParams(name="Harbour", c=1.1, a=0.3, b=0.8, g=0.93)
medium   = HomogeneousMedium(params=my_water)
```

### Custom stratified medium

```python
from uowc.medium import LayeredMedium
from uowc.config import CLEAR_WATER, COASTAL_WATER, TURBID_WATER
import numpy as np

custom = LayeredMedium(
    layers=(
        (5.0,    CLEAR_WATER),
        (15.0,   COASTAL_WATER),
        (np.inf, TURBID_WATER),
    ),
    name="Custom Three-Layer",
)
```

### Custom gradient medium (from in-situ CTD/AC-9 data)

```python
from uowc.medium import GradientMedium

profile = GradientMedium(
    z_samples=(0.0, 5.0, 10.0, 20.0),
    c_samples=(0.24, 0.40, 0.80, 1.60),
    b_samples=(0.09, 0.18, 0.56, 1.30),
    g_samples=(0.92, 0.93, 0.93, 0.94),
    name="Measured CTD Profile",
)
```

### Custom chlorophyll (Kameda) medium

```python
from uowc.medium import ChlorophyllProfileMedium

my_dcm = ChlorophyllProfileMedium(
    chl_background=0.08, peak_height=0.7, peak_depth_m=15.0, peak_width_m=4.0,
    z_max_m=30.0, name="Coastal DCM @15 m",
)
```

### Custom coupled ocean layer (IOPs + turbulence co-located)

```python
from uowc.turbulence import CoupledOceanMedium, OceanLayer
from uowc.config import CLEAR_WATER, COASTAL_WATER
import numpy as np

channel = CoupledOceanMedium(
    layers=(
        OceanLayer(12.0,   CLEAR_WATER,   epsilon=1e-9, chi_T=8.5e-12, v_current=0.05),
        OceanLayer(np.inf, COASTAL_WATER, epsilon=1e-5, chi_T=9.0e-7,  v_current=0.12),
    ),
    name="Custom Thermocline",
)
```

### Custom turbulent chlorophyll medium (Model 3 A/B)

```python
from uowc.turbulence import TurbulentChlorophyllMedium
from uowc.medium import KAMEDA_CHL_PROFILE

m3 = TurbulentChlorophyllMedium(
    iop=KAMEDA_CHL_PROFILE,            # reuse the Model-2 optics verbatim
    epsilon=1e-6, chi_T=1e-7,          # stronger turbulence
    current_speed=0.15, name="Kameda + moderate turbulence",
)
```

### Additional convergence metric

```python
from uowc.simulation.convergence import register_metric

def _make_mean_scatters(cfg):
    def estimator(record, n_launched):
        if record["n_scatters"].size == 0:
            return None
        return float(record["n_scatters"].mean())
    return estimator

register_metric("mean_scatters", _make_mean_scatters)
# Then set cfg.conv_metrics = ("power", "mean_scatters")
```

---

## Reproducibility & seeding

- `main.py` draws a fresh 32-bit seed from system entropy on every run and **prints it**;
  pass `--seed N` to reproduce a specific run exactly.
- Within a run, RNG sub-streams are derived with `numpy.random.SeedSequence`: each model uses
  a distinct seed offset (0/1/2), each (medium, beam, range) combination spawns its own
  child, and each adaptive round spawns again — so every batch is statistically independent
  (verified: repeated `spawn` calls are non-overlapping).
- **Do not** reuse a seed across accumulation runs — identical photons stack and inflate
  apparent convergence while adding no real statistics.

---

## References

- Gabriel et al. (2013). *Channel modeling for underwater optical wireless communications*. IEEE/OSA JOCN 5(1):1–12.
- Wang, Jacques & Zheng (1995). *MCML — Monte Carlo modeling of light transport in multi-layered tissues*. Computer Physics Communications 47:131–146.
- Woodcock et al. (1965). *Techniques used in the GEM code for Monte Carlo neutron transport* — delta-tracking majorant method.
- Lux & Koblinger (1991). *Monte Carlo Particle Transport Methods*. CRC Press.
- Mobley & Preisendorfer (1994). *Light and Water: Radiative Transfer in Natural Waters*. Academic Press.
- Petzold (1972). *Volume Scattering Functions for Selected Ocean Waters*. SIO Ref 72-78.
- Morel (1991); Gordon & Morel (1983); Bricaud et al. (1998); Pope & Fry (1997) — Case-1 bio-optics / pure-water absorption.
- Haltrin (1999). *Chlorophyll-based model of seawater optical properties*. Applied Optics 38(33):6826.
- Kameda & Matsumura (1998) — vertical chlorophyll (deep chlorophyll maximum) profile form.
- Nikishov & Nikishov (2000). *Spectrum of turbulent fluctuations of the sea-water refraction index*. Int. J. Fluid Mech. Res. 27(1):82–98.
- Andrews & Phillips (2001). *Laser Beam Scintillation with Applications*. SPIE Press.
- Korotkova et al. (2012). *Light scintillation in oceanic turbulence*. Waves Random Complex Media 22(2):260–266.
- Yi et al. (2015). *Underwater optical communication performance under oceanic turbulence*. Opt. Express 23(4):4886–4895.
- Quan & Fry (1995). *Empirical equation for the index of refraction of seawater*. Applied Optics 34(18):3477.
- Thorpe (2005). *The Turbulent Ocean*. Cambridge University Press.
- Born & Wolf (1999). *Principles of Optics*, 7th ed., §1.5 — Fresnel equations.
</content>
</invoke>
