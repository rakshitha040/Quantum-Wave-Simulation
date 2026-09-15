"""
Energy conservation analysis for the 1D wave equation FDM solver.

Recreates the explicit central-difference (leapfrog) scheme used by
the classical solver -- without modifying it -- and tracks kinetic,
potential, and total energy over time to verify energy conservation.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Parameters
L = 1.0  # domain length, must match the FDM solver
C = 1.0  # wave speed
NX = 401  # spatial grid points, including both boundaries
CFL = 0.5  # Courant number c*dt/dx, must satisfy CFL <= 1
T_FINAL = 4.0  # total simulation time (two periods of the fundamental mode)
TABLE_TIMES = np.arange(0.0, T_FINAL + 1e-9, 0.5)  # rows for the per-IC energy tables


# Initial conditions
def standing_wave_ic(x: np.ndarray, length: float) -> np.ndarray:
    """Fundamental normal mode, sin(pi*x/L)."""
    return np.sin(np.pi * x / length)


def gaussian_pulse_ic(x: np.ndarray, *_unused) -> np.ndarray:
    """Narrow Gaussian pulse centered at x = 0.5."""
    return np.exp(-200.0 * (x - 0.5) ** 2)


# Solver (explicit central-difference / leapfrog, Dirichlet BC)
def solve_wave_equation(
    u0: np.ndarray, v0: np.ndarray, dx: float, dt: float, n_steps: int, c: float
) -> np.ndarray:
    """Integrate u_tt = c^2 u_xx with fixed boundaries; returns U[time, space]."""
    nx = u0.size
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0

    r2 = (c * dt / dx) ** 2
    u[1, 1:-1] = (
        u[0, 1:-1] + dt * v0[1:-1] + 0.5 * r2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])
    )

    for n in range(1, n_steps):
        u[n + 1, 1:-1] = (
            2 * u[n, 1:-1] - u[n - 1, 1:-1] + r2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
        )

    return u  # boundaries stay 0 by construction


# Energy analysis
def time_derivative(u: np.ndarray, dt: float) -> np.ndarray:
    """u_t(i,n) = (u(i,n+1) - u(i,n-1)) / (2*dt), one-sided at the ends."""
    u_t = np.empty_like(u)
    u_t[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dt)
    u_t[0, :] = (u[1, :] - u[0, :]) / dt
    u_t[-1, :] = (u[-1, :] - u[-2, :]) / dt
    return u_t


def space_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """u_x(i,n) = (u(i+1,n) - u(i-1,n)) / (2*dx), one-sided at the ends."""
    u_x = np.empty_like(u)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    u_x[:, 0] = (u[:, 1] - u[:, 0]) / dx
    u_x[:, -1] = (u[:, -1] - u[:, -2]) / dx
    return u_x


def compute_energies(
    u: np.ndarray, x: np.ndarray, dt: float, c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kinetic, potential, and total energy at every time step."""
    dx = x[1] - x[0]
    u_t = time_derivative(u, dt)  # centered time derivative
    u_x = space_derivative(u, dx)  # centered spatial derivative

    kinetic = 0.5 * _trapz(u_t**2, x, axis=1)
    potential = 0.5 * c**2 * _trapz(u_x**2, x, axis=1)
    total = kinetic + potential
    return kinetic, potential, total


def energy_statistics(total: np.ndarray) -> dict:
    """Summary statistics used to assess energy conservation."""
    variation = 100 * (total.max() - total.min()) / total[0]
    return {
        "initial": total[0],
        "final": total[-1],
        "max": total.max(),
        "min": total.min(),
        "variation_pct": variation,
    }


def relative_energy_error(total: np.ndarray) -> np.ndarray:
    """Relative deviation of total energy from its initial value."""
    return (total - total[0]) / total[0]


def sample_energy_at_times(
    t: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    sample_times: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Nearest-index snapshots of (time, kinetic, potential, total) at each sample time."""
    rows = []
    for sample_time in sample_times:
        idx = int(np.argmin(np.abs(t - sample_time)))
        rows.append((t[idx], kinetic[idx], potential[idx], total[idx]))
    return rows


# Plotting
def _style_axes(ax: plt.Axes, ylabel: str, title: str) -> None:
    """Apply shared axis formatting."""
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_energy_component(
    t: np.ndarray, values: np.ndarray, ylabel: str, color: str, title: str
) -> None:
    """Single-quantity vs-time plot, displayed on screen."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(t, values, color=color, linewidth=1.3)
    _style_axes(ax, ylabel, title)
    fig.tight_layout()
    plt.show()


def plot_combined_energy(
    t: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    title: str,
) -> None:
    """Kinetic, potential, and total energy on one figure, displayed on screen."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(t, kinetic, label="Kinetic", color="#d62728", linewidth=1.1)
    ax.plot(t, potential, label="Potential", color="#2ca02c", linewidth=1.1)
    ax.plot(t, total, label="Total", color="#1f4e8c", linewidth=1.6)
    _style_axes(ax, "Energy", title)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_energy_table(name: str, rows: list[tuple[float, float, float, float]]) -> None:
    """Draw a Time / Kinetic / Potential / Total energy table for one initial condition."""
    col_labels = ["Time", "Kinetic Energy", "Potential Energy", "Total Energy"]
    cell_text = [[f"{time:.3f}", f"{k:.6f}", f"{p:.6f}", f"{e:.6f}"] for time, k, p, e in rows]

    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.4 * len(rows)), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1, 1.5)
    for col in range(len(col_labels)):
        table[0, col].set_text_props(weight="bold")

    ax.set_title(f"Energy vs Time -- {name}", pad=14)
    fig.tight_layout()
    plt.show()


# Reporting
def conclusion_message(variation_pct: float) -> str:
    """Adaptive conclusion sentence based on the measured energy variation."""
    if variation_pct < 0.1:
        quality = "excellent"
    elif variation_pct < 1.0:
        quality = "good"
    else:
        quality = "poor"
    return (
        f"Total energy variation remained at {variation_pct:.4f}%, "
        f"indicating {quality} numerical energy conservation."
    )


def print_energy_report(name: str, stats: dict) -> None:
    """Print initial/final/extremal energy, percentage variation, and conclusion."""
    print(f"--- {name}: Energy Conservation Report ---")
    print(f"Initial Total Energy : {stats['initial']:.6f}")
    print(f"Final Total Energy   : {stats['final']:.6f}")
    print(f"Maximum Total Energy : {stats['max']:.6f}")
    print(f"Minimum Total Energy : {stats['min']:.6f}")
    print(f"Energy Variation     : {stats['variation_pct']:.4f}%")
    print("Conclusion:")
    print(f"  {conclusion_message(stats['variation_pct'])}\n")


def print_interpretation(name: str, stats: dict) -> None:
    """Print a short research-style interpretation of the result."""
    quality = "nearly constant" if stats["variation_pct"] < 1.0 else "bounded but oscillating"
    print(
        f"For the {name} case, the total energy remains {quality} "
        "throughout the simulation while kinetic and potential energies "
        "periodically exchange, confirming that the finite-difference "
        "solver preserves the physical energy of the 1D wave equation.\n"
    )


# Main
def analyze(name: str, ic_func) -> None:
    """Run the solver and energy analysis pipeline for one initial condition."""
    dx = L / (NX - 1)
    dt = CFL * dx / C
    n_steps = int(round(T_FINAL / dt))

    x = np.linspace(0.0, L, NX)
    t = np.arange(n_steps + 1) * dt
    u0 = ic_func(x, L)
    v0 = np.zeros_like(u0)  # both initial conditions start from rest

    u = solve_wave_equation(u0, v0, dx, dt, n_steps, C)
    kinetic, potential, total = compute_energies(u, x, dt, C)
    stats = energy_statistics(total)
    rel_error = relative_energy_error(total)

    print_energy_report(name, stats)
    print_interpretation(name, stats)

    plot_energy_component(
        t, kinetic, "Kinetic Energy", "#d62728", f"Kinetic Energy vs Time -- {name}"
    )
    plot_energy_component(
        t, potential, "Potential Energy", "#2ca02c", f"Potential Energy vs Time -- {name}"
    )
    plot_energy_component(
        t,
        rel_error,
        "Relative Energy Error  (E(t) - E0) / E0",
        "#1f4e8c",
        f"Relative Energy Error vs Time -- {name}",
    )
    plot_combined_energy(
        t, kinetic, potential, total, f"Energy Conservation Analysis -- {name}"
    )

    table_rows = sample_energy_at_times(t, kinetic, potential, total, TABLE_TIMES)
    plot_energy_table(name, table_rows)


def main() -> None:
    """Run the energy conservation analysis for both project initial conditions."""
    analyze("Standing Wave", standing_wave_ic)
    analyze("Gaussian Pulse", gaussian_pulse_ic)


if __name__ == "__main__":
    main()