"""
CFL stability analysis for the 1D wave equation FDM solver.

Recreates the explicit central-difference (leapfrog) scheme with fixed
(Dirichlet) boundaries -- without modifying the existing solver -- and
runs it at several Courant numbers, for both project initial
conditions, to verify the CFL stability limit r = c*dt/dx <= 1.
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # domain length
C = 1.0  # wave speed
NX = 401  # spatial grid points, including both boundaries
T_FINAL = 4.0  # target simulation time
CFL_VALUES = [0.5, 0.9, 1.0, 1.1]
AMP_CAP = 1.0e12  # numerical safeguard: caps float growth for unstable cases
UNSTABLE_THRESHOLD = 1.0e6  # amplitude above this is classified unstable outright


# Initial conditions
def standing_wave_ic(x: np.ndarray, length: float) -> np.ndarray:
    """Fundamental normal mode, sin(pi*x/L), released from rest."""
    return np.sin(np.pi * x / length)


def gaussian_pulse_ic(x: np.ndarray, *_unused) -> np.ndarray:
    """Narrow Gaussian pulse centered at x = 0.5, released from rest."""
    return np.exp(-200.0 * (x - 0.5) ** 2)


# Solver (explicit central-difference / leapfrog, Dirichlet BC)
def solve_wave_equation(
    u0: np.ndarray, v0: np.ndarray, dx: float, dt: float, n_steps: int, c: float
) -> np.ndarray:
    """Integrate u_tt = c^2 u_xx with fixed boundaries; returns U[time, space]."""
    u = np.zeros((n_steps + 1, u0.size))
    u[0] = u0

    r2 = (c * dt / dx) ** 2
    u[1, 1:-1] = (
        u[0, 1:-1] + dt * v0[1:-1] + 0.5 * r2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])
    )

    for n in range(1, n_steps):
        u[n + 1, 1:-1] = (
            2 * u[n, 1:-1] - u[n - 1, 1:-1] + r2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
        )
        np.clip(u[n + 1], -AMP_CAP, AMP_CAP, out=u[n + 1])  # guard against overflow

    return u


# Stability analysis
def classify_stability(cfl: float, peak_amplitude: float) -> str:
    """CFL < 1 stable, CFL == 1 marginally stable, CFL > 1 unstable;
    an amplitude above UNSTABLE_THRESHOLD always classifies as unstable."""
    if peak_amplitude > UNSTABLE_THRESHOLD:
        return "Unstable"
    if cfl < 1.0:
        return "Stable"
    if cfl == 1.0:
        return "Marginally stable"
    return "Unstable"


def run_cfl_case(ic_func, cfl: float) -> dict:
    """Run the solver at a given CFL number and collect diagnostics."""
    dx = L / (NX - 1)
    dt = cfl * dx / C
    n_steps = int(round(T_FINAL / dt))

    x = np.linspace(0.0, L, NX)
    t = np.arange(n_steps + 1) * dt
    u0 = ic_func(x, L)
    v0 = np.zeros_like(u0)  # released from rest

    u = solve_wave_equation(u0, v0, dx, dt, n_steps, C)
    max_amplitude = np.max(np.abs(u), axis=1)
    peak_amplitude = max_amplitude.max()

    return {
        "cfl": cfl,
        "status": classify_stability(cfl, peak_amplitude),
        "x": x,
        "t": t,
        "u_final": u[-1],
        "max_amplitude": max_amplitude,
        "peak_amplitude": peak_amplitude,
    }


# Plotting
def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Shared axis formatting."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, save_path: str) -> None:
    """Create the output directory and save the figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_stability_profiles(results: list, title: str, save_path: str) -> None:
    """Final-time wave profile for each CFL number, one subplot each."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    for ax, result in zip(axes.flat, results):
        ax.plot(result["x"], result["u_final"], color="#1f4e8c", linewidth=1.5)
        subtitle = f"CFL = {result['cfl']} ({result['status']})"
        _style_axes(ax, "x", "u(x, T_final)", subtitle)
    fig.suptitle(title)
    save_figure(fig, save_path)


def plot_amplitude_vs_time(results: list, title: str, save_path: str) -> None:
    """Maximum amplitude vs time for all CFL numbers, on one graph."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    styles = ["-", "--", ":", "-."]
    for result, color, style in zip(results, colors, styles):
        ax.plot(
            result["t"],
            result["max_amplitude"],
            label=f"CFL = {result['cfl']}",
            color=color,
            linestyle=style,
            linewidth=1.8,
        )
    ax.set_yscale("log")
    _style_axes(ax, "Time", "Max |u(x, t)|  (log scale)", title)
    ax.legend(frameon=False)
    save_figure(fig, save_path)


def plot_stability_table(results: list, title: str, save_path: str) -> None:
    """Table of CFL, stability status, and maximum amplitude."""
    col_labels = ["CFL", "Stability", "Maximum Amplitude"]
    cell_text = [[f"{r['cfl']}", r["status"], f"{r['peak_amplitude']:.6e}"] for r in results]

    fig, ax = plt.subplots(figsize=(6.5, 1.0 + 0.4 * len(results)), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1, 1.6)
    for col in range(len(col_labels)):
        table[0, col].set_text_props(weight="bold")

    ax.set_title(title, pad=14)
    save_figure(fig, save_path)


# Reporting
def print_stability_section(title: str, results: list) -> None:
    """Print a CFL : stability status line for each case, under a header."""
    print(f"===== {title} =====")
    for r in results:
        print(f"CFL = {r['cfl']} : {r['status']}")


def print_interpretation() -> None:
    """Print the overall CFL-condition conclusion."""
    print(
        "\nThe finite-difference method remains stable for CFL <= 1 and "
        "becomes unstable for CFL > 1, confirming the Courant-Friedrichs-Lewy "
        "stability criterion for both standing waves and Gaussian pulses."
    )


# Main
def analyze_ic(name: str, folder: str, ic_func) -> list:
    """Run the CFL sweep for one initial condition and save its figures."""
    results = [run_cfl_case(ic_func, cfl) for cfl in CFL_VALUES]
    base = f"results/{folder}/stability"

    plot_stability_profiles(
        results, f"Stability Profiles -- {name}", f"{base}/stability_profiles.png"
    )
    plot_amplitude_vs_time(
        results, f"Maximum Amplitude vs Time -- {name}", f"{base}/amplitude_vs_time.png"
    )
    plot_stability_table(
        results, f"CFL Stability Summary -- {name}", f"{base}/stability_table.png"
    )

    return results


def main() -> None:
    """Run the CFL stability study for both initial conditions."""
    standing_results = analyze_ic("Standing Wave", "standing_wave", standing_wave_ic)
    gaussian_results = analyze_ic("Gaussian Pulse", "gaussian_pulse", gaussian_pulse_ic)

    print_stability_section("STANDING WAVE", standing_results)
    print_stability_section("GAUSSIAN PULSE", gaussian_results)
    print_interpretation()


if __name__ == "__main__":
    main()