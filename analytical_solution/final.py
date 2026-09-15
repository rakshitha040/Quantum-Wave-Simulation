"""
analytical_reference.py

Combined analytical reference solutions for the 1D wave equation, covering
both initial-condition cases used in the FDM study, on the SAME spatial grid
convention as the FDM script (x = 0, dx, 2*dx, ..., L with dx = L/(Nx-1),
Nx odd so x = L/2 lands exactly on a grid point).

Governing equation:
    d^2u/dt^2 = c^2 * d^2u/dx^2,   0 <= x <= L

Boundary conditions (homogeneous Dirichlet):
    u(0, t) = 0
    u(L, t) = 0

CASE 1 -- Sine-wave IC (exact closed-form solution):
    u(x, 0)     = A * sin(m*pi*x/L),   du/dt(x,0) = 0
    u(x, t)     = A * sin(m*pi*x/L) * cos(m*pi*c*t/L)
    (single Dirichlet eigenfunction -> series collapses to one term, no
    truncation, no numerical integration)

CASE 2 -- Gaussian-pulse IC (Fourier sine-series solution):
    u(x, 0)     = A * exp(-(x-x0)^2 / (2*sigma^2)),   du/dt(x,0) = 0
    u(x, t)     = sum_n A_n * cos(n*pi*c*t/L) * sin(n*pi*x/L)
    A_n = (2/L) * integral_0^L f(x) sin(n*pi*x/L) dx   (scipy.integrate.quad)
    (not an eigenfunction -> infinite superposition, truncated to N_modes
    terms; this is a semi-analytical solution: truncation + quadrature +
    floating-point error only, never time-stepping error)

No time-stepping is used anywhere in this script for either case.
All plots are displayed only (plt.show()) -- nothing is written to disk.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# ==========================================================================
# 1. Physical and numerical parameters
#    (kept identical in name and value to the FDM script, so x below is the
#    exact same spatial grid used by the FDM solver)
# ==========================================================================
L = 1.0                 # Length of string (m)
T = 2.0                 # Total simulation time (s)
c = 1.0                 # Wave speed

Nx = 201                # Number of spatial grid points (ODD, so x=L/2 lands exactly on a grid point)
dx = L / (Nx - 1)

Nt = 401                # number of time samples for the analytical solution
                         # (set this to match FDM's Nt, or pass FDM's own t
                         #  array into the functions below, when you need to
                         #  difference U_FDM and U_analytical element-wise)

# Case 1 (sine-wave) parameters
A_SINE = 1.0             # amplitude
M_MODE = 1               # mode number (sin(m*pi*x/L))

# Case 2 (Gaussian pulse) parameters
A_GAUSS = 1.0            # pulse amplitude
X0 = 0.5                 # pulse center
SIGMA = 0.05             # pulse width
N_MODES_DEFAULT = 200    # default number of Fourier modes for the main run


# ==========================================================================
# 2. Spatial and temporal grids (shared by both cases)
# ==========================================================================
def make_grids(L, T, Nx, Nt, dx=None):
    """
    Create the spatial grid using the same convention as the FDM script
    (x = 0, dx, 2*dx, ..., L with dx = L/(Nx-1)) and a uniform time grid.

    Passing dx explicitly guarantees this grid is bit-for-bit identical to
    the FDM spatial grid whenever the same L, Nx are used -- required before
    U_FDM and U_analytical can be differenced element-wise for error metrics.
    """
    if dx is None:
        dx = L / (Nx - 1)
    x = np.arange(Nx) * dx          # equivalent to np.linspace(0, L, Nx)
    t = np.linspace(0.0, T, Nt)
    return x, t


# ==========================================================================
# CASE 1: Sine-wave IC -- exact closed-form solution
# ==========================================================================
def sine_wave_solution(x, t, A, m, L, c):
    """u(x,t) = A * sin(m*pi*x/L) * cos(m*pi*c*t/L), evaluated on a full grid.

    Returns U with shape (Nt, Nx): U[i, j] = u(x[j], t[i]).
    """
    k = m * np.pi / L                    # spatial wavenumber
    omega = k * c                        # angular frequency (omega = c*k)
    spatial = A * np.sin(k * x)          # shape (Nx,)
    temporal = np.cos(omega * t)         # shape (Nt,)
    return np.outer(temporal, spatial)   # shape (Nt, Nx)


def sine_wave_derivatives(x, t, A, m, L, c):
    """
    u_t(x,t) = -A * omega * sin(k x) * sin(omega t)
    u_x(x,t) =  A * k     * cos(k x) * cos(omega t)
    """
    k = m * np.pi / L
    omega = k * c
    Ut = -A * omega * np.outer(np.sin(omega * t), np.sin(k * x))
    Ux = A * k * np.outer(np.cos(omega * t), np.cos(k * x))
    return Ut, Ux


def sine_wave_mode_spectrum(x, U_snapshot, L, n_check=10):
    """Project a spatial profile onto sin(n*pi*x/L), n=1..n_check."""
    coeffs = np.zeros(n_check)
    for n in range(1, n_check + 1):
        basis = np.sin(n * np.pi * x / L)
        coeffs[n - 1] = (2.0 / L) * np.trapezoid(U_snapshot * basis, x)
    return np.arange(1, n_check + 1), coeffs


# ==========================================================================
# CASE 2: Gaussian-pulse IC -- Fourier sine-series solution
# ==========================================================================
def gaussian_initial_condition(x, A, x0, sigma):
    """f(x) = A * exp(-(x - x0)^2 / (2*sigma^2))."""
    return A * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))


def fourier_sine_coefficients(N_modes, A, x0, sigma, L):
    """
    A_n = (2/L) * int_0^L f(x) sin(n*pi*x/L) dx, n = 1..N_modes.

    scipy.integrate.quad (adaptive Gauss-Kronrod quadrature) is used instead
    of a fixed-panel rule so the narrow Gaussian peak is resolved accurately
    without needing a wastefully dense uniform grid.
    """
    def f(x):
        return A * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))

    coeffs = np.zeros(N_modes)
    for n in range(1, N_modes + 1):
        integrand = lambda x, n=n: f(x) * np.sin(n * np.pi * x / L)
        value, _ = quad(integrand, 0.0, L, limit=200)
        coeffs[n - 1] = (2.0 / L) * value
    return coeffs


def gaussian_reconstruct(x, t, coeffs, L, c):
    """u(x,t) = sum_n A_n * cos(n*pi*c*t/L) * sin(n*pi*x/L), vectorized."""
    N_modes = len(coeffs)
    n = np.arange(1, N_modes + 1)
    k_n = n * np.pi / L
    omega_n = c * k_n

    spatial_basis = np.sin(np.outer(x, k_n))          # (Nx, N_modes)
    temporal_basis = np.cos(np.outer(t, omega_n))     # (Nt, N_modes)
    weighted_spatial = spatial_basis * coeffs
    return temporal_basis @ weighted_spatial.T         # (Nt, Nx)


def gaussian_derivatives(x, t, coeffs, L, c):
    """
    u_t(x,t) = -sum_n A_n * omega_n * sin(omega_n t) * sin(k_n x)
    u_x(x,t) =  sum_n A_n * k_n     * cos(omega_n t) * cos(k_n x)
    """
    N_modes = len(coeffs)
    n = np.arange(1, N_modes + 1)
    k_n = n * np.pi / L
    omega_n = c * k_n

    spatial_sin = np.sin(np.outer(x, k_n))
    spatial_cos = np.cos(np.outer(x, k_n))
    temporal_sin = np.sin(np.outer(t, omega_n))
    temporal_cos = np.cos(np.outer(t, omega_n))

    Ut = -(temporal_sin @ (spatial_sin * (coeffs * omega_n)).T)
    Ux = temporal_cos @ (spatial_cos * (coeffs * k_n)).T
    return Ut, Ux


def gaussian_convergence_study(x, t, mode_counts, reference_modes, A, x0, sigma, L, c):
    """Compare truncated reconstructions against a high-resolution reference."""
    print(f"[Convergence] building reference solution with {reference_modes} modes ...")
    coeffs_ref = fourier_sine_coefficients(reference_modes, A, x0, sigma, L)
    U_ref = gaussian_reconstruct(x, t, coeffs_ref, L, c)

    results = []
    for N in mode_counts:
        coeffs_N = fourier_sine_coefficients(N, A, x0, sigma, L)
        U_N = gaussian_reconstruct(x, t, coeffs_N, L, c)
        diff = U_N - U_ref
        l2_error = np.linalg.norm(diff) / np.linalg.norm(U_ref)
        rmse = np.sqrt(np.mean(diff ** 2))
        max_abs_error = np.max(np.abs(diff))
        results.append((N, l2_error, rmse, max_abs_error))
        print(f"[Convergence] N_modes={N:4d}  L2_rel={l2_error:.3e}  "
              f"RMSE={rmse:.3e}  max_abs_err={max_abs_error:.3e}")
    return results, U_ref, coeffs_ref


# ==========================================================================
# Shared: boundary-condition check, energy, plotting
# ==========================================================================
def verify_boundary_conditions(U, label, tol=1e-8):
    left_max = np.max(np.abs(U[:, 0]))
    right_max = np.max(np.abs(U[:, -1]))
    ok = (left_max < tol) and (right_max < tol)
    print(f"[{label}] max|u(0,t)| = {left_max:.3e}   max|u(L,t)| = {right_max:.3e}   "
          f"Dirichlet BCs satisfied: {ok}")
    return ok


def compute_energy(x, Ut, Ux, c):
    """E(t) = 1/2 * integral[ u_t^2 + c^2 u_x^2 ] dx."""
    density = Ut**2 + (c**2) * Ux**2
    return 0.5 * np.trapezoid(density, x, axis=1)


def plot_snapshots(x, t, U, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for st in np.linspace(0, t[-1], 6):
        idx = np.argmin(np.abs(t - st))
        ax.plot(x, U[idx, :], label=f"t = {t[idx]:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x, t)"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()


def plot_heatmap(x, t, U, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(x, t, U, shading="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label="u(x, t)")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title)
    fig.tight_layout(); plt.show()


def plot_surface(x, t, U, title):
    X, Tm = np.meshgrid(x, t)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Tm, U, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("u(x, t)"); ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="u(x, t)")
    fig.tight_layout(); plt.show()


def plot_energy(t, E, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, E, color="darkred")
    ax.set_xlabel("t"); ax.set_ylabel("Total Energy E(t)"); ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()
    rel_drift = (np.max(E) - np.min(E)) / np.mean(E)
    print(f"[Energy] mean E = {np.mean(E):.6e},  relative drift = {rel_drift:.3e}")


def plot_mode_spectrum(mode_numbers, coeffs, title, log=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    if log:
        ax.semilogy(mode_numbers, np.abs(coeffs), marker=".", linestyle="none")
    else:
        ax.stem(mode_numbers, np.abs(coeffs))
    ax.set_xlabel("Mode number n"); ax.set_ylabel("|A_n|"); ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); plt.show()


def plot_convergence(results):
    mode_counts = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(mode_counts, [r[1] for r in results], "o-", label="Relative L2 error")
    ax.loglog(mode_counts, [r[2] for r in results], "s-", label="RMSE")
    ax.loglog(mode_counts, [r[3] for r in results], "^-", label="Max absolute error")
    ax.set_xlabel("Number of Fourier modes N"); ax.set_ylabel("Error")
    ax.set_title("Fourier-Mode Convergence (Gaussian Pulse)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); plt.show()


# ==========================================================================
# Reusable FDM-vs-analytical comparison metrics
# ==========================================================================
def mean_squared_error(U_FDM, U_analytical):
    return np.mean((U_FDM - U_analytical) ** 2)


def root_mean_squared_error(U_FDM, U_analytical):
    return np.sqrt(mean_squared_error(U_FDM, U_analytical))


def l2_relative_error(U_FDM, U_analytical):
    return np.linalg.norm(U_FDM - U_analytical) / np.linalg.norm(U_analytical)


def max_absolute_error(U_FDM, U_analytical):
    return np.max(np.abs(U_FDM - U_analytical))


def time_dependent_error(U_FDM, U_analytical):
    """RMSE(t) for each time row -- useful for later FDM comparison."""
    diff = U_FDM - U_analytical
    return np.sqrt(np.mean(diff ** 2, axis=1))


# ==========================================================================
# Main driver -- runs both cases end to end
# ==========================================================================
def run_sine_case(x, t):
    print("\n===== CASE 1: Sine-wave IC (exact closed form) =====")
    U = sine_wave_solution(x, t, A_SINE, M_MODE, L, c)
    verify_boundary_conditions(U, label="Sine BC check")

    plot_snapshots(x, t, U, "Analytical Standing-Wave: Snapshots")
    plot_heatmap(x, t, U, "Analytical Standing-Wave: Space-Time Heatmap")
    plot_surface(x, t, U, "Analytical Standing-Wave: 3D Surface")

    Ut, Ux = sine_wave_derivatives(x, t, A_SINE, M_MODE, L, c)
    E = compute_energy(x, Ut, Ux, c)
    plot_energy(t, E, "Analytical Standing-Wave: Energy Conservation")

    mode_numbers, coeffs = sine_wave_mode_spectrum(x, U[0, :], L)
    plot_mode_spectrum(mode_numbers, coeffs, "Spatial Fourier Spectrum (Sine IC)")
    dominant_mode = mode_numbers[np.argmax(np.abs(coeffs))]
    print(f"[Fourier] dominant mode = {dominant_mode} (expected m = {M_MODE}) -- "
          "single mode because the IC is itself a Dirichlet eigenfunction.")

    return U, Ut, Ux, E


def run_gaussian_case(x, t):
    print("\n===== CASE 2: Gaussian-pulse IC (Fourier sine series) =====")
    coeffs = fourier_sine_coefficients(N_MODES_DEFAULT, A_GAUSS, X0, SIGMA, L)
    U = gaussian_reconstruct(x, t, coeffs, L, c)

    f_exact = gaussian_initial_condition(x, A_GAUSS, X0, SIGMA)
    ic_err = np.max(np.abs(f_exact - U[0, :]))
    print(f"[IC check] max|f(x) - reconstruction at t=0| = {ic_err:.3e}")
    verify_boundary_conditions(U, label="Gaussian BC check")

    plot_snapshots(x, t, U, "Gaussian-Pulse Solution: Snapshots")
    plot_heatmap(x, t, U, "Gaussian-Pulse Solution: Space-Time Heatmap")
    plot_surface(x, t, U, "Gaussian-Pulse Solution: 3D Surface")
    plot_mode_spectrum(np.arange(1, len(coeffs) + 1), coeffs,
                        "Gaussian Pulse: Modal Spectrum", log=True)

    Ut, Ux = gaussian_derivatives(x, t, coeffs, L, c)
    E = compute_energy(x, Ut, Ux, c)
    plot_energy(t, E, "Gaussian-Pulse Solution: Energy vs Time")

    mode_counts = [10, 20, 50, 100, 200]
    results, U_ref, coeffs_ref = gaussian_convergence_study(
        x, t, mode_counts, reference_modes=500,
        A=A_GAUSS, x0=X0, sigma=SIGMA, L=L, c=c
    )
    plot_convergence(results)

    print("[Explanation] The Gaussian is not a single Dirichlet eigenfunction, "
          "so it needs a superposition of many sine modes; narrower pulses "
          "(smaller sigma) decay more slowly in |A_n| and need more modes.")

    return U, Ut, Ux, E, coeffs


def main():
    x, t = make_grids(L, T, Nx, Nt, dx=dx)

    U_sine, Ut_sine, Ux_sine, E_sine = run_sine_case(x, t)
    U_gauss, Ut_gauss, Ux_gauss, E_gauss, coeffs_gauss = run_gaussian_case(x, t)

    return {
        "x": x, "t": t,
        "sine": {"U": U_sine, "Ut": Ut_sine, "Ux": Ux_sine, "E": E_sine},
        "gaussian": {"U": U_gauss, "Ut": Ut_gauss, "Ux": Ux_gauss,
                     "E": E_gauss, "coeffs": coeffs_gauss},
    }


if __name__ == "__main__":
    main()