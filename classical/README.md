# Classical Finite Difference Solver

This module implements the **classical numerical solution** of the **One-Dimensional Wave Equation** using the **Explicit Second-Order Finite Difference Method (FDM)** with **Leapfrog Time Integration**.

It serves as the **classical baseline** for the research project:

> **Classical vs. Quantum: A Comparative Study of Finite Difference and Hamiltonian Simulation Approaches to the 1D Wave Equation**

---

# Objective

The objective of this module is to numerically solve the one-dimensional wave equation and visualize wave propagation using the Finite Difference Method (FDM). The results obtained from this solver will later be compared with the Hamiltonian-based quantum simulation.

---

# Governing Equation

The one-dimensional wave equation is

```
∂²u(x,t)
──────── = c² × ∂²u(x,t)
  ∂t²            ───────
                   ∂x²
```

where

- **u(x,t)** → Displacement of the string
- **x** → Spatial position
- **t** → Time
- **c** → Wave propagation speed

---

# Numerical Method

The solver uses the following numerical techniques:

- Explicit Second-Order Finite Difference Method (FDM)
- Leapfrog Time Integration
- Bootstrap Initialization for the First Time Step
- Fixed (Dirichlet) Boundary Conditions
- CFL (Courant-Friedrichs-Lewy) Stability Condition

The numerical update equation used in the simulation is

```
u(i,n+1) = 2u(i,n) - u(i,n-1) + r² × [u(i+1,n) - 2u(i,n) + u(i-1,n)]
```

where

```
r = cΔt / Δx
```

For stability,

```
r ≤ 1
```

---

# Initial Conditions

Two different physical scenarios are implemented.

## 1. Standing Wave

Initial displacement

```
u(x,0) = sin(πx/L)
```

Characteristics

- Fundamental vibration mode
- Fixed ends
- Exact analytical solution available
- Used for numerical validation

---

## 2. Gaussian Pulse

Initial displacement

```
u(x,0) = exp[-200(x − 0.5)²]
```

Characteristics

- Localized wave packet
- Travels in both directions
- Demonstrates wave propagation and reflection

---

# Solver Features

- Explicit Second-Order Finite Difference Method
- Leapfrog Time Integration
- CFL Stability Verification
- Standing Wave Simulation
- Gaussian Pulse Simulation
- Analytical Solution Validation
- RMS Error Calculation
- Wave Animation
- Wave Snapshot
- Midpoint Displacement Plot
- Space-Time Heatmap
- 3D Surface Visualization

---

# Solver Specifications

| Property | Description |
|----------|-------------|
| PDE | One-Dimensional Wave Equation |
| Numerical Method | Explicit Finite Difference Method |
| Time Integration | Leapfrog Scheme |
| Spatial Accuracy | Second Order |
| Temporal Accuracy | Second Order |
| Boundary Condition | Fixed Ends (Dirichlet) |
| Initial Conditions | Standing Wave, Gaussian Pulse |
| Programming Language | Python |
| Libraries | NumPy, Matplotlib |

---

# Project Structure

```
classical/
│
├── fdm_wave_solver.py
├── README.md
│
└── results/
    │
    ├── standing_wave/
    │   ├── animation.gif
    │   ├── wave_snapshot.png
    │   ├── midpoint_displacement.png
    │   ├── space_time_heatmap.png
    │   └── surface3d.png
    │
    └── gaussian_pulse/
        ├── animation.gif
        ├── wave_snapshot.png
        ├── midpoint_displacement.png
        ├── space_time_heatmap.png
        └── surface3d.png
```

---

# Simulation Results

The generated simulation outputs are organized according to the initial condition.

## Standing Wave

Located in

```
results/standing_wave/
```

Contains

- Wave Animation
- Wave Snapshot
- Midpoint Displacement
- Space-Time Heatmap
- 3D Surface Plot

---

## Gaussian Pulse

Located in

```
results/gaussian_pulse/
```

Contains

- Wave Animation
- Wave Snapshot
- Midpoint Displacement
- Space-Time Heatmap
- 3D Surface Plot

---

# Running the Solver

Install the required packages

```bash
pip install -r ../requirements.txt
```

Run the solver

```bash
python fdm_wave_solver.py
```

Inside the program, change

```python
WAVE_TYPE = "sine"
```

or

```python
WAVE_TYPE = "gaussian"
```

to simulate the desired initial condition.

---

# Future Work

This classical solver serves as the reference implementation for the research project.

The next phase includes

- Matrix formulation of the One-Dimensional Wave Equation
- Hamiltonian construction
- Quantum Hamiltonian Simulation
- Comparative analysis between classical and quantum approaches

---

# Author

**Rakshitha Jagu**

**Research Intern – Quantum Algorithms**  
Enginuvity Nexus Technologies

B.Tech Information Technology  
MVGR College of Engineering

---

# License

This module is distributed under the MIT License.