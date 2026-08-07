"""
Fourier sine modal analysis of the 1D wave equation initial conditions.

Since the 1D wave equation satisfies fixed (Dirichlet) boundary
conditions, a Fourier sine decomposition provides the physically
appropriate modal representation, using eigenfunctions sin(n*pi*x/L).
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

_trapz = getattr(np, "trapezoid", None) or np.trapz


# Parameters
L = 1.0          # domain length, must match the FDM solver
NX = 401         # spatial grid points, including both boundaries
N_MODES = 100    # number of Fourier sine modes to compute


# Initial condition
def build_grid(length: float, n_points: int) -> np.ndarray:
    """Uniform grid on [0, length]."""
    return np.linspace(0.0, length, n_points)


def standing_wave_ic(x: np.ndarray, length: float) -> np.ndarray:
    """Fundamental normal mode, sin(pi*x/L)."""
    return np.sin(np.pi * x / length)


def gaussian_pulse_ic(x: np.ndarray, *_unused) -> np.ndarray:
    """Narrow Gaussian pulse centered at x = 0.5."""
    return np.exp(-200.0 * (x - 0.5) ** 2)


# Fourier analysis
def fourier_sine_coefficients(
    u0: np.ndarray, x: np.ndarray, length: float, n_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourier sine series coefficients.

    b_n = (2 / L) * integral(u0(x) * sin(n*pi*x/L) dx), for n = 1 .. n_max.
    """
    n = np.arange(1, n_max + 1)
    basis = np.sin(np.outer(n, x) * np.pi / length)
    b = (2.0 / length) * _trapz(basis * u0, x, axis=1)
    return n, b


def normalize_spectrum(b: np.ndarray) -> np.ndarray:
    """Scale spectral magnitude to a [0, 1] range."""
    magnitude = np.abs(b)
    return magnitude / np.max(magnitude)


def real_space_energy(u0: np.ndarray, x: np.ndarray) -> float:
    """Signal energy in real space, integral(u0(x)^2 dx)."""
    return _trapz(u0 ** 2, x)


def modal_space_energy(b: np.ndarray, length: float) -> float:
    """Signal energy in modal space via Parseval's theorem."""
    return (length / 2.0) * np.sum(b ** 2)


def report_energy_conservation(
    u0: np.ndarray, x: np.ndarray, b: np.ndarray, length: float
) -> None:
    """Print real-space vs. modal-space energy as a consistency check."""
    real_energy = real_space_energy(u0, x)
    modal_energy = modal_space_energy(b, length)
    captured = 100 * modal_energy / real_energy

    print(f"Real-space energy : {real_energy:.6f}")
    print(f"Modal-space energy: {modal_energy:.6f}")
    print(f"Energy captured   : {captured:.3f}%\n")


# Plotting
def plot_spectrum(n: np.ndarray, b_norm: np.ndarray, title: str) -> None:
    """Stem plot of the normalized Fourier sine spectrum."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    markerline, stemlines, _ = ax.stem(n, b_norm, basefmt=" ")
    plt.setp(markerline, marker="o", markersize=4, color="#1f4e8c")
    plt.setp(stemlines, color="#1f4e8c", linewidth=1.0)

    ax.set_xlabel("Mode Number n")
    ax.set_ylabel("Normalized Modal Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, n[-1] + 1)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# Main
def compute_spectrum(ic_func) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the grid, evaluate the IC, and compute its sine spectrum."""
    x = build_grid(L, NX)
    u0 = ic_func(x, L)
    n, b = fourier_sine_coefficients(u0, x, L, N_MODES)
    return x, u0, n, b


def analyze(name: str, ic_func) -> None:
    """Run the full Fourier sine analysis pipeline for one initial condition."""
    x, u0, n, b = compute_spectrum(ic_func)
    b_norm = normalize_spectrum(b)

    print(f"--- {name} ---")
    report_energy_conservation(u0, x, b, L)

    title = f"Fourier Sine Modal Spectrum -- {name}"
    plot_spectrum(n, b_norm, title)


def main() -> None:
    """Run the analysis for both project initial conditions."""
    analyze("Standing Wave", standing_wave_ic)
    analyze("Gaussian Pulse", gaussian_pulse_ic)


if __name__ == "__main__":
    main()