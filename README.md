# monte-carlo-uowc

A modular Monte Carlo simulation framework for underwater optical wireless communication (UOWC) channels. Photon packets propagate through homogeneous and depth-stratified seawater, accumulating realistic absorption, scattering, optical turbulence, and Fresnel interface losses. The framework produces channel impulse responses, received power curves, delay spread, and 3 dB bandwidth over configurable link ranges and beam types.

---

## Features

- **Dual-kernel transport** — a homogeneous kernel using standard Beer-Lambert free-flight sampling and a Woodcock delta-tracking kernel for arbitrary depth-varying IOPs, sharing a single unified propagation loop.
- **Implicit absorption + Russian roulette** — photon weights are reduced by the single-scattering albedo ω = b/c at each real collision (MCML scheme); photons with weights below 10⁻⁴ are terminated with the unbiased 1/m survival rule.
- **Henyey-Greenstein scattering** — analytic inverse-CDF sampling of the polar angle; vectorised scalar form for homogeneous media and per-photon array form for stratified media.
- **Optical turbulence** — Nikishov (2000) C_n² parameterisation from (ε, χ_T) ocean parameters; Markov-limit phase-screen angular kicks at every transport step; Gamma-Gamma / log-normal / negative-exponential scintillation fading at the receiver, selected by the path-integrated Rytov variance σ²_R.
- **Coupled ocean layers** — `CoupledOceanMedium` bundles IOPs and turbulence parameters into a single depth grid so IOP and turbulence boundaries are always aligned.
- **Fresnel transmittance** — exact two-polarisation Fresnel equations applied at the water → glass receiver window.
- **Adaptive convergence** — multi-metric batch-means stopping criterion; configurable per-metric relative-error tolerances for power, delay spread, bandwidth, and CIR shape.
- **Parallel execution** — Python `ProcessPoolExecutor` sweeps over (medium, beam, link range) combinations; results are aggregated from independent seeded sub-streams.
- **Pandas data pipeline** — per-photon DataFrames with Parquet / CSV export; per-run capture statistics with correct n_launched denominators.
- **Publication figures** — Matplotlib plots for received power, CIR, frequency response, RMS delay spread, and 3 dB bandwidth.

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

**Requirements:** Python ≥ 3.12, NumPy ≥ 2.4, SciPy ≥ 1.17, Pandas ≥ 3.0, Matplotlib ≥ 3.10, PyArrow ≥ 24.0, Marimo ≥ 0.23.

---

## Quick start

```bash
# Run the full pipeline — homogeneous + inhomogeneous, metrics, plots, Parquet export
python -m uowc.main all

# Homogeneous simulation only (Clear Water / Coastal Water, two beams, five ranges)
python -m uowc.main homogeneous

# Inhomogeneous simulation only (Woodcock delta-tracking, three stratified profiles)
python -m uowc.main inhomogeneous

# Write outputs to a custom directory
python -m uowc.main all --out ./results
```

### Running from Python

```python
import numpy as np
from uowc.config import SIM, CLEAR_WATER, COLLIMATED, RECEIVER
from uowc.transport import propagate_batch
from uowc.metrics import compute_all_metrics
from uowc.simulation import run_sweep_adaptive, RunKey

# One-shot adaptive sweep over all preset water types and beam geometries
results = run_sweep_adaptive(SIM, verbose=True)

key = RunKey("Clear Water", "Collimated (Laser)", 10.0)
metrics = compute_all_metrics(results[key], SIM, CLEAR_WATER.c, link_range=10.0)
print(f"Received power : {metrics['power_dB']:.2f} dB")
print(f"RMS delay spread: {metrics['delay_spread_s']*1e9:.3f} ns")
print(f"3 dB bandwidth  : {metrics['bandwidth_hz']/1e6:.1f} MHz")
```

---

## CLI reference

| Command | Description |
|---|---|
| `python -m uowc.main all` | Full pipeline: both simulations, metrics, plots, Parquet |
| `python -m uowc.main homogeneous` | Homogeneous simulation only |
| `python -m uowc.main inhomogeneous` | Woodcock (stratified) simulation only |
| `--out PATH` | Output directory (default: `./outputs`) |

---

## Module structure

```
uowc/
├── config/        Physical constants, preset water types, beam/receiver geometry, SimConfig
├── physics/       Pure photon-physics functions — sampling, absorption, rotation, Fresnel
├── medium/        MediumProfile interface + HomogeneousMedium, LayeredMedium, GradientMedium
├── turbulence/    TurbulenceProfile interface, Nikishov C_n², phase screens, scintillation,
│                  CoupledOceanMedium (aligned IOP + turbulence per depth layer)
├── transport/     _propagate_chunk loop + four public worker functions (picklable for
│                  ProcessPoolExecutor); owns the MCML stepping logic
├── simulation/    Parallel sweep orchestration (run_sweep_adaptive, etc.) + multi-metric
│                  convergence framework (batch-means relative standard error)
├── metrics/       Derived channel metrics from RunResult: power, CIR, delay spread,
│                  frequency response, bandwidth, OOK BER, coherence time
├── analysis/      Pandas DataFrame pipeline, Parquet/CSV export, capture statistics
├── plotting/      Matplotlib figure generation for all channel metrics
├── reporting.py   Console summary tables
main.py            CLI entry point
notebook.py        Interactive Marimo diagnostics notebook
```

### Separation of concerns

Each module owns exactly one domain and imports downward only:

```
config  ←  physics  ←  transport  ←  simulation  ←  main
               ↑            ↑
            turbulence    medium
               ↓            ↓
            metrics  ←  analysis  ←  plotting
```

- `physics` — stateless functions; no awareness of receivers, files, or parallelism.
- `transport` — advances photons; calls physics and turbulence; returns `PhotonRecord` dicts.
- `simulation` — distributes photon batches to workers; applies convergence stopping.
- `metrics` — pure NumPy functions on `RunResult`; no transport or I/O.
- `analysis` — Pandas / Parquet layer; no physics.

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

**Receiver:** 4 inch aperture (radius 50.8 mm), FOV = 180° (hemisphere).

### Inhomogeneous medium presets

| Name | Layers | Depth |
|---|---|---|
| `STRATIFIED_OCEAN` | Clear → Coastal | 0–10 m / 10+ m |
| `DEEP_OCEAN_COLUMN` | Clear → Coastal → Turbid | 0–8 / 8–18 / 18+ m |
| `COASTAL_GRADIENT` | Smooth c from 0.241 to 2.19 m⁻¹ | 7 sample points, 0–25 m |

### Turbulence presets (CoupledOceanMedium)

| Name | ε (m²/s³) | χ_T (K²/s) | C_n² (m⁻²/³) | Regime |
|---|---|---|---|---|
| `CALM_OPEN_OCEAN` | 10⁻⁹ | 8.5×10⁻¹² | ~10⁻¹⁵ | Weak |
| `COASTAL_SURFACE` | 10⁻⁷ | 3.95×10⁻⁹ | ~10⁻¹³ | Weak–moderate |
| `STRATIFIED_THERMOCLINE` | 10⁻⁹ / 10⁻⁵ | 8.5×10⁻¹² / 9.16×10⁻⁷ | 10⁻¹⁵ / 5×10⁻¹² | Weak / Moderate |
| `DEEP_OCEAN_STRATIFIED` | 10⁻⁹ / 10⁻⁵ / 10⁻⁸ | layered | 10⁻¹⁵ / 5×10⁻¹² / 10⁻¹⁴ | Mixed |

---

## Key algorithms

### Free-flight step sampling

```
s = −ln(ξ) / c          (homogeneous, Beer-Lambert inverse-CDF)
s = −ln(ξ) / c_max      (inhomogeneous, Woodcock majorant)
```

### Woodcock delta-tracking

Steps are sampled with the global majorant c_max ≥ c(z). The collision at depth z is accepted as real with probability c(z)/c_max; rejected collisions are null (the photon continues without weight change or scatter). By the thinning theorem of Poisson processes the sequence of real collisions reproduces the correct inhomogeneous Poisson process with local rate c(z).

### Implicit absorption

```
w ← w × ω(z)     where ω = b/c  (single-scattering albedo)
```

Applied in-place to the full weight array via integer-index assignment `w[active] *= omega`, ensuring NumPy's fancy-index semantics do not silently create a copy.

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

Near-z singularity (|uz| > 1 − 10⁻⁵) is handled analytically.

### Nikishov turbulence parameterisation

```
C_n²(z) = B₀ · (dn/dT)² · ε(z)^(−1/3) · χ_T(z)
```

where B₀ ≈ 3.63 is the Kolmogorov-Corrsin structural constant, dn/dT = −1.80×10⁻⁴ K⁻¹ at 530 nm, ε is kinetic energy dissipation rate, and χ_T is temperature variance dissipation rate. The parameter `dn_dT` is configurable for other wavelengths (Quan & Fry 1995).

### Phase-screen angular kicks

```
σ²_α = 3.04 · l₀^(−1/3) · C_n²(z) · Δz      [rad²]
```

Gaussian transverse kicks are drawn at every transport step and applied to the direction vector in the Markov (continuous random-walk) limit.

### Scintillation fading

The path-integrated Rytov variance drives regime selection:

```
σ²_R = 1.23 · k^(7/6) · (∫₀^L C_n²(z) dz) / L · L^(11/6)
```

| σ²_R | Distribution | Notes |
|---|---|---|
| < 0.3 | Log-normal | E[h_t] = 1 guaranteed |
| 0.3 – 5 | Gamma-Gamma (Andrews & Phillips 2001) | α, β from σ²_X, σ²_Y |
| ≥ 5 | Negative-exponential | Fully developed speckle |

### Fresnel transmittance

Exact two-polarisation Fresnel equations are applied when photons reach the receiver, modelling the water (n = 1.33) → glass window (n = 1.50) interface. At normal incidence T ≈ 99.6%.

---

## Outputs

| Artifact | Location | Description |
|---|---|---|
| Console tables | stdout | Received power, delay spread, bandwidth per run |
| `power_vs_range_*.png` | `outputs/` | MC and Beer-Lambert power vs link range |
| `cir_*.png` | `outputs/` | Normalised channel impulse response |
| `frequency_response_*.png` | `outputs/` | \|H(f)\| magnitude |
| `delay_spread_*.png` | `outputs/` | RMS delay spread vs range |
| `bandwidth_*.png` | `outputs/` | 3 dB bandwidth vs range |
| `photons_homogeneous.parquet` | `outputs/` | Per-photon DataFrame, homogeneous runs |
| `photons_inhomogeneous.parquet` | `outputs/` | Per-photon DataFrame, Woodcock runs |
| `diagnostics/` | `outputs/` | Weight distribution, scatter histograms, spatial maps |

### Per-photon DataFrame schema

| Column | Type | Description |
|---|---|---|
| `photon_id` | int64 | Global unique ID across the sweep |
| `weight` | float64 | Final photon weight after absorption |
| `tof_s` / `tof_ns` | float64 | Time of flight in seconds / nanoseconds |
| `x_m`, `y_m`, `r_m` | float32 | Position at capture plane |
| `path_length_m` | float32 | Total geometric path length |
| `n_scatters` | int32 | Real scattering events |
| `n_nulls` | int32 | Null (Woodcock virtual) collisions |
| `excess_path_m` | float32 | Extra path beyond straight-line distance |
| `medium_name`, `beam_name`, `link_range_m` | category / float | Run metadata |

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

### Custom coupled ocean layer (IOPs + turbulence co-located)

```python
from uowc.turbulence import CoupledOceanMedium, OceanLayer
from uowc.config import CLEAR_WATER, COASTAL_WATER
import numpy as np

channel = CoupledOceanMedium(
    layers=(
        OceanLayer(12.0,   CLEAR_WATER,   epsilon=1e-9,  chi_T=8.5e-12, v_current=0.05),
        OceanLayer(np.inf, COASTAL_WATER, epsilon=1e-5,  chi_T=9.0e-7,  v_current=0.12),
    ),
    name="Custom Thermocline",
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

## Adaptive convergence

The simulation runs in batches until all configured metrics satisfy a batch-means relative standard error criterion:

```
rel_error = std(batch_samples) / (sqrt(n_batches) · |mean|) < tolerance
```

Configure via `SimConfig`:

```python
from uowc.config import SimConfig

cfg = SimConfig(
    n_photons            = 1_000_000,   # photons per batch
    link_ranges_m        = (5, 10, 15, 20, 25),
    dt_bin_s             = 1e-11,
    n_time_bins          = 3000,
    weight_threshold     = 1e-4,
    roulette_m           = 10,
    n_workers            = 8,
    master_seed          = 42,
    chunk_size           = 10_000,
    min_captured_photons = 10_000,
    max_launched_photons = 100_000_000,
    conv_metrics         = ("power", "delay_spread", "bandwidth"),
    rel_error_tol        = 0.02,        # 2% relative error on all metrics
    min_conv_batches     = 3,
)
```

---

## References

- Gabriel et al. (2013). *Channel modeling for underwater optical wireless communications*. IEEE/OSA JOCN 5(1):1–12.
- Wang, Jacques & Zheng (1995). *MCML — Monte Carlo modeling of light transport in multi-layered tissues*. Computer Physics Communications 47:131–146.
- Woodcock et al. (1965). *Techniques used in the GEM code for the neutron transport problems*. — delta-tracking majorant method.
- Mobley & Preisendorfer (1994). *Light and Water: Radiative Transfer in Natural Waters*. Academic Press.
- Petzold (1972). *Volume Scattering Functions for Selected Ocean Waters*. SIO Ref 72-78.
- Haltrin (1999). *Chlorophyll-based model of seawater optical properties*. Applied Optics 38(33):6826.
- Nikishov & Nikishov (2000). *Spectrum of turbulent fluctuations of the sea-water refraction index*. Int. J. Fluid Mech. Res. 27(1):82–98.
- Andrews & Phillips (2001). *Laser Beam Scintillation with Applications*. SPIE Press.
- Korotkova et al. (2012). *Light scintillation in oceanic turbulence*. Waves Random Complex Media 22(2):260–266.
- Quan & Fry (1995). *Empirical equation for the index of refraction of seawater*. Applied Optics 34(18):3477.
- Born & Wolf (1999). *Principles of Optics*, 7th ed., §1.5. — Fresnel equations.
