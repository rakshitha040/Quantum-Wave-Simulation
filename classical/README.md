# Classical Finite Difference Solver

This module implements the **classical numerical solution of the one-dimensional wave equation** using the **Explicit Second-Order Finite Difference Method (FDM)** with **Leapfrog Time Integration**.

It serves as the classical baseline for the research project:

> **Classical vs. Quantum: A Comparative Study of Finite Difference and Hamiltonian Simulation Approaches to the 1D Wave Equation**

---

# Objective

The objective of this module is to solve the one-dimensional wave equation numerically and study its physical and numerical properties using finite-difference methods.

The classical results obtained here will later be compared with Hamiltonian-based quantum simulations.

---

# Governing Equation

The one-dimensional wave equation is

```text
∂²u(x,t)
──────── = c² ∂²u(x,t)
  ∂t²          ───────
                 ∂x²
```

where

- **u(x,t)** → wave displacement
- **x** → spatial coordinate
- **t** → time
- **c** → wave propagation speed

---

# Numerical Method

The solver uses

- Explicit Second-Order Finite Difference Method (FDM)
- Leapfrog time integration
- Bootstrap initialization
- Fixed (Dirichlet) boundary conditions
- CFL stability condition

The numerical update equation is

```text
u(i,n+1) = 2u(i,n) − u(i,n−1) + r² [u(i+1,n) − 2u(i,n) + u(i−1,n)]
```

where

```text
r = cΔt / Δx
```

For stability,

```text
r ≤ 1
```

---

# Initial Conditions

Two different physical scenarios are implemented.

## 1. Standing Wave

Initial displacement:

```text
u(x,0) = sin(πx/L)
```

Characteristics:

- Fundamental vibration mode
- Fixed endpoints
- Exact analytical solution available
- Single Fourier mode

---

## 2. Gaussian Pulse

Initial displacement:

```text
u(x,0) = exp[-200(x − 0.5)²]
```

Characteristics:

- Localized wave packet
- Propagates in both directions
- Reflects at boundaries
- Contains multiple Fourier modes

---

# Analyses Performed

The classical study consists of four major analyses.

---

## 1. Wave Propagation Analysis

Generated outputs:

- Wave animation
- Wave snapshot
- Midpoint displacement
- Space-time heatmap
- 3D surface plot

---

## 2. Fourier Modal Analysis

The numerical solution is decomposed into sine modes:

```text
u(x,t) = Σ bₙ(t) sin(nπx/L)
```

Generated outputs:

- Fourier spectrum for the standing wave
- Fourier spectrum for the Gaussian pulse

Observations:

- Standing wave occupies a single mode.
- Gaussian pulse excites multiple modes.

---

## 3. Energy Conservation Analysis

The total energy is

```text
E(t) = K(t) + P(t)
```

where

```text
K(t) = ½ ∫ (u_t)² dx
```

and

```text
P(t) = ½ c² ∫ (u_x)² dx
```

Generated outputs:

- Kinetic energy
- Potential energy
- Total energy
- Combined energy plot
- Energy summary table

Observations:

- Total energy remains approximately constant.
- Kinetic and potential energies periodically exchange.

---

## 4. Stability Analysis

The CFL number is

```text
CFL = cΔt / Δx
```

The following cases are studied:

- CFL = 0.5
- CFL = 0.9
- CFL = 1.0
- CFL = 1.1

Generated outputs:

- Stability summary table
- Maximum amplitude vs time
- Stability profiles

Observations:

- CFL < 1 → stable
- CFL = 1 → marginally stable
- CFL > 1 → unstable

---

# Features

- Explicit second-order FDM solver
- Leapfrog time integration
- Standing-wave simulation
- Gaussian-pulse simulation
- Fourier modal decomposition
- Energy conservation analysis
- CFL stability verification
- Wave animation
- Space-time heatmaps
- 3D visualizations

---

# Solver Specifications

| Property | Description |
|---|---|
| PDE | One-Dimensional Wave Equation |
| Numerical Method | Explicit Finite Difference Method |
| Time Integration | Leapfrog Scheme |
| Spatial Accuracy | Second Order |
| Temporal Accuracy | Second Order |
| Boundary Condition | Dirichlet |
| Initial Conditions | Standing Wave, Gaussian Pulse |
| Language | Python |
| Libraries | NumPy, Matplotlib |

---

# Project Structure

```text
classical/

├── fdm_wave_solver.py
├── fourier_analysis.py
├── energy_analysis.py
├── stability_analysis.py
├── README.md
│
└── results/
    │
    ├── gaussian_pulse/
    │   ├── energy_analysis/
    │   ├── fourier_analysis/
    │   ├── stability_analysis/
    │   ├── animation_gaussian.gif
    │   ├── midpoint_displacement.png
    │   ├── space_time_heatmap.png
    │   ├── wave_snapshot.png
    │   └── 3d_surface.png
    │
    └── standing_wave/
        ├── energy_analysis/
        │   ├── energy_combined.png
        │   ├── energy_kinetic.png
        │   ├── energy_potential.png
        │   ├── energy_total.png
        │   └── table.png
        │
        ├── fourier_analysis/
        │   └── fourier_spectrum_standing_wave.png
        │
        ├── stability_analysis/
        │   ├── amplitude_vs_time.png
        │   ├── stability_profiles.png
        │   └── stability_table.png
        │
        ├── animation_sine.gif
        ├── midpoint_displacement.png
        ├── space_time_heatmap.png
        ├── wave_snapshot.png
        └── 3d_surface.png
```

---

# Running the Simulations

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the wave solver:

```bash
python fdm_wave_solver.py
```

Run Fourier analysis:

```bash
python fourier_analysis.py
```

Run energy analysis:

```bash
python energy_analysis.py
```

Run stability analysis:

```bash
python stability_analysis.py
```

---

# Phase 1 Status

✅ Finite Difference Solver

✅ Standing Wave Simulation

✅ Gaussian Pulse Simulation

✅ Fourier Modal Analysis

✅ Energy Conservation Analysis

✅ Stability Analysis

✅ Visualization and Validation

---

# Future Work (Phase 2)

The next phase includes:

- Matrix formulation of the wave equation
- Schrödingerization
- Hamiltonian construction
- Quantum circuit implementation
- Quantum simulation
- Classical–quantum comparison

---

# Author

**Rakshitha Jagu**

**Research Intern – Quantum Algorithms**

Enginuvity Nexus Technologies

**B.Tech – Information Technology**

MVGR College of Engineering

---

# License

This project is distributed under the MIT License.