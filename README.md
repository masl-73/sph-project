# Neon RTI: SPH Rayleigh-Taylor Instability Simulation

A high-performance Smoothed Particle Hydrodynamics (SPH) simulation of Rayleigh-Taylor Instability (RTI) featuring hexagonal particle packing, Numba-accelerated physics, and "Neon-style" visualizations.


## 🚀 Features

- **High Performance**: Physics kernels accelerated with **Numba** (JIT compiled) and parallelized.
- **Physics Engine**:
    - Tait Equation of State (EOS) for pressure.
    - Monaghan Artificial Viscosity for stability.
    - Ghost Particle boundary conditions (slip/reflect).
    - Energy conservation tracking (Internal Energy dissipation).
- **Visualization**:
    - "Smooth" grid interpolation mode with bicubic filtering.
    - "Particles" mode for high-contrast point visualization.
    - Dark "Neon" aesthetic (Cyan/Magenta colors).
- **Analysis**:
    - Live energy balance plots (Kinetic, Potential, Internal).
    - Mixing layer width calculation.
    - Velocity distribution and vorticity analysis.

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/masl-73/sph-project.git
   cd sph-project
   ```

2. **Setup virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### Run Simulation
Use the provided launch script:
```bash
./run_simulation.sh
```
Or run manually with options:
```bash
python3 src/simulation.py --viz-mode smooth --clear
```

### Run Analysis
Generate comprehensive data plots from checkpoints:
```bash
./run_analysis.sh
```

## 📊 Simulation Results (Step 17000)

| Fluid Density | Velocity Field |
|:---:|:---:|
| ![Step 17000](step_17000.png) | ![Velocity 17000](vel_step_17000.png) |

### Real-time Analysis
| Energy Conservation | Mixing Layer Width |
|:---:|:---:|
| ![Energy Latest](energy_latest.png) | ![Mixing Latest](mixing_latest.png) |

## 📊 Project Structure

- `src/`: Core implementation.
  - `physics.py`: SPH kernels and force calculations.
  - `sph_solver.py`: Time integration and simulation orchestration.
  - `rti_setup.py`: Initial condition configuration.
- `data/`: Checkpoint storage (.npz).
- `output/`: Generated visualization frames (.png).
- `output_analysis/`: Statistical plots and metric logs.

## 📜 License
GPL-3.0
