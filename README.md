# monte-carlo-uowc

A modular Monte Carlo simulation framework for underwater optical wireless communication (UOWC) channels. Models photon propagation through homogeneous and inhomogeneous seawater, accounts for optical turbulence, computes channel metrics, and generates publication-ready analysis figures.

## Key features

- **Dual-kernel photon transport**: Homogeneous (standard free-flight) and inhomogeneous (Woodcock delta-tracking) Monte Carlo engines
- **Turbulence modeling**: Phase-screen angular kicks, scintillation fading, and coupled ocean-layer dynamics via Nikishov parameterisation
- **Flexible media**: Preset Clear and Coastal water types; supports custom depth-dependent IOPs via `MediumProfile` interface
- **Configurable geometries**: Collimated (laser-like) and diffused (LED-like) transmitters
- **Parallel sweeps**: Multi-process parameter sweeps over link ranges, beam types, and media
- **Comprehensive metrics**: Received power, RMS delay spread, channel impulse response (CIR), frequency response, and 3 dB bandwidth
- **Data pipeline**: Pandas-based photon event analysis and Parquet/CSV export for downstream statistical processing
- **Interactive notebooks**: Marimo-based exploration tools for simulation diagnostics and result visualization

## Requirements

- Python 3.12 or newer
- NumPy, Matplotlib, Pandas, SciPy, Marimo (see `pyproject.toml`)

## Installation

```bash
uv sync
```

Or with pip in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the complete pipeline (homogeneous + inhomogeneous simulations, metrics, diagnostics, plots):

```bash
python -m uowc.main all
```

Run specific workflows:

```bash
python -m uowc.main homogeneous      # Homogeneous simulation only
python -m uowc.main inhomogeneous    # Inhomogeneous (turbulent) simulation only
python -m uowc.main plots            # Generate figures from existing results
python -m uowc.main diagnostics      # Compute diagnostic tables and statistics
```

Specify output directory:

```bash
python -m uowc.main all --out ./custom_outputs
```

## Project structure

- **`uowc/`** — Main package
  - **`transport/`** — Monte Carlo photon propagation (homogeneous, inhomogeneous, turbulent, non-turbulent workers)
  - **`medium/`** — Spatial optical medium profiles (homogeneous, inhomogeneous, Woodcock delta-tracking support)
  - **`turbulence/`** — Optical turbulence physics (phase screens, scintillation, Nikishov parameterisation, coupled ocean model)
  - **`metrics/`** — Channel metric computation (CIR, delay spread, frequency response, bandwidth)
  - **`physics/`** — Pure optical physics utilities (Henyey-Greenstein scattering, absorption, Fresnel, refractive index)
  - **`simulation/`** — Parallel sweep orchestration and result aggregation
  - **`plotting/`** — Figure generation (received power, CIR, frequency response, delay spread, bandwidth)
  - **`analysis/`** — Pandas DataFrame pipeline and statistical analysis (photon-level events, Parquet/CSV export)
  - **`config/`** — Physical constants, preset media, beam geometries, simulation parameters
  - **`reporting.py`** — Console summary tables and result formatting
- **`main.py`** — CLI entry point (flexible workflow orchestration)
- **`notebook.py`** — Interactive Marimo notebook for diagnostics
- **`__marimo__/`** — Marimo-specific notebooks and state

## Simulation pipeline

1. **Configure**: Load media profiles, beam geometry, receiver parameters, and link ranges
2. **Transport**: Run adaptive sweep across (medium, beam, range) combinations
   - Homogeneous kernel (no turbulence)
   - Inhomogeneous kernel (Woodcock method for depth-dependent IOPs)
   - Optional turbulence: phase screens + scintillation fading
3. **Metrics**: Compute CIR, delay spread, bandwidth, and frequency response from photon weights and TOFs
4. **Analysis**: Export photon-level events to DataFrame; compute aggregate statistics
5. **Report**: Print summary tables (delay spread, received power, bandwidth)
6. **Plot**: Generate and save analysis figures

## Output

- **Console**: Summary tables for delay spread, received power, and bandwidth
- **Figures**: Matplotlib PNG files in output directory:
  - Received power vs. link range and beam/medium combinations
  - Channel impulse response (CIR) time-domain traces
  - Frequency response magnitude and phase
  - RMS delay spread vs. range
  - 3 dB bandwidth vs. range
- **Data**: Parquet/CSV datasets of photon events (photon_id, weight, TOF, position, medium, beam, range)

## Key algorithms

- **Monte Carlo transport**: Free-flight sampling with Russian roulette and implicit absorption
- **Scattering**: Henyey-Greenstein phase function with forward-bias parameter *g*
- **Turbulence**: Phase-screen Markov-limit angular kicks; Gamma-distributed scintillation fading
- **Delta-tracking**: Woodcock method for inhomogeneous media (depth-dependent attenuation)
- **Receiver**: Circular aperture with acceptance angle (FOV) filtering and Fresnel reflection

## Architecture notes

- **Separation of Concerns**: Modules own disjoint responsibilities (transport, media, turbulence, metrics, plotting, analysis) with minimal coupling
- **Vectorization**: All medium, physics, and analysis functions use NumPy array operations to eliminate Python loops in hot paths
- **Immutable Data**: Medium profiles and simulation configs are frozen dataclasses (hashable, picklable, safe for multiprocessing)
- **Extensibility**: Add new media via `MediumProfile` subclass; new turbulence models via `TurbulenceProfile`; new metrics via analysis functions
