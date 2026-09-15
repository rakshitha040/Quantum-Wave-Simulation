import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# PARAMETERS
# =====================================================

L = 1.0                 # Length of string (m)
T = 2.0                 # Total simulation time (s)
c = 1.0                 # Wave speed

Nx = 201                 # Number of spatial grid points (ODD, so x=L/2 lands exactly on a grid point)
dx = L / (Nx - 1)

WAVE_TYPE = "sine"    # change to "sine" or "gaussian" to try the other initial condition
SHOW_ANIMATION = True     # set False if you just want the static plots

# =====================================================
# CFL CONDITION
# =====================================================

CFL = 0.9                # Must be <= 1 for stability
dt = CFL * dx / c        # Automatically computed time step
Nt = int(T / dt)
T_actual = Nt * dt
r = c * dt / dx

print("=" * 40)
print("Simulation Parameters")
print("=" * 40)
print(f"Wave Speed (c)      = {c}")
print(f"dx                  = {dx:.6f}")
print(f"dt                  = {dt:.6f}")
print(f"CFL Number (r)      = {r:.3f}")
print(f"Number of Time Steps= {Nt}")

if r > 1:
    raise ValueError("CFL condition violated! Simulation unstable.")

print("Simulation is Stable.")
print("=" * 40)

# =====================================================
# SPATIAL GRID
# =====================================================

x = np.linspace(0, L, Nx)

# =====================================================
# INITIAL CONDITIONS
# =====================================================

if WAVE_TYPE == "gaussian":
    u0 = np.exp(-200 * (x - 0.5) ** 2)          # Gaussian pulse, centered at x=0.5
elif WAVE_TYPE == "sine":
    u0 = np.sin(np.pi * x / L)                  # fundamental standing-wave mode
else:
    raise ValueError("WAVE_TYPE must be 'gaussian' or 'sine'")

v0 = np.zeros(Nx)   # released from rest (zero initial velocity)

# =====================================================
# ARRAYS
# =====================================================

u_prev = np.copy(u0)
u_curr = np.copy(u0)
u_next = np.zeros(Nx)

# =====================================================
# FIRST TIME STEP  (bootstrap formula, needed because the
# scheme normally needs TWO previous time layers, and at
# the very start we only have ONE — u0 — plus the initial velocity)
# =====================================================

for i in range(1, Nx - 1):
    u_curr[i] = (
        u_prev[i]
        + dt * v0[i]
        + 0.5 * r**2 *
        (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
    )

u_curr[0] = 0
u_curr[-1] = 0

# =====================================================
# STORE SOLUTION
# =====================================================

solution = [u_prev.copy(), u_curr.copy()]

# =====================================================
# MAIN FDM LOOP  (the leapfrog update — this is the actual wave equation)
# =====================================================

for n in range(1, Nt):
    for i in range(1, Nx - 1):
        u_next[i] = (
            2*u_curr[i]
            - u_prev[i]
            + r**2 *
            (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        )

    u_next[0] = 0
    u_next[-1] = 0

    solution.append(u_next.copy())

    u_prev[:] = u_curr
    u_curr[:] = u_next

U = np.array(solution)
time = np.arange(len(solution)) * dt

# =====================================================
# ERROR CHECK AGAINST THE EXACT SOLUTION
# =====================================================

if WAVE_TYPE == "sine":
    # exact solution is valid for ALL time here, since a sine mode
    # is an exact eigenmode of a pinned string
    U_exact = np.array([np.cos(np.pi * c * t / L) * np.sin(np.pi * x / L) for t in time])
    rms_error = np.sqrt(np.mean((U - U_exact) ** 2, axis=1))
    print(f"[sine] max RMS error over whole run : {rms_error.max():.3e}")
    print(f"[sine] RMS error at final time       : {rms_error[-1]:.3e}")

elif WAVE_TYPE == "gaussian":
    # exact (d'Alembert) solution only holds BEFORE the wave reaches
    # the walls (after that it's a reflection problem, which needs
    # the method of images -- out of scope here, so we just check
    # the early, pre-reflection part of the run)
    def gaussian_shape(xx):
        return np.exp(-200 * (xx - 0.5) ** 2)

    def dalembert(xx, t):
        return 0.5 * gaussian_shape(xx - c * t) + 0.5 * gaussian_shape(xx + c * t)

    t_check = 0.3   # well before the pulse reaches either wall (~t=0.5)
    idx_check = int(round(t_check / dt))
    u_exact_check = dalembert(x, idx_check * dt)
    rms_error = np.sqrt(np.mean((U[idx_check] - u_exact_check) ** 2))
    print(f"[gaussian] RMS error vs analytic (pre-reflection) at t={idx_check*dt:.3f} : {rms_error:.3e}")

# =====================================================
# ANIMATION
# =====================================================

if SHOW_ANIMATION:
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x, solution[0], color='blue', linewidth=2)
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"1D Wave Equation using FDM ({WAVE_TYPE})")
    ax.grid(True)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=13,
                         bbox=dict(facecolor="white", alpha=0.8))

    display_dt = 0.01
    frame_step = max(1, int(display_dt / dt))
    frames = range(0, len(solution), frame_step)

    def update(frame):
        line.set_ydata(solution[frame])
        time_text.set_text(f"Time = {frame*dt:.3f} s")
        return line, time_text

    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=False, repeat=True)
    plt.show()

# =====================================================
# PLOT 1 : u vs x  (snapshot at a fixed physical time)
# =====================================================

t_snapshot = 0.3
frame = min(int(round(t_snapshot / dt)), len(solution) - 1)

plt.figure(figsize=(8, 5))
plt.plot(x, solution[frame], linewidth=2)
plt.xlabel("Position (x)")
plt.ylabel("Displacement (u)")
plt.title(f"u vs x at t = {frame*dt:.4f} s")
plt.grid(True)
plt.show()

# =====================================================
# PLOT 2 : u vs t  (at the TRUE center of the string)
# =====================================================

mid = int(np.argmin(np.abs(x - L/2)))   # FIX: nearest grid point to x=L/2, works for any Nx
u_time = [sol[mid] for sol in solution]

plt.figure(figsize=(8, 5))
plt.plot(time, u_time)
plt.xlabel("Time (t)")
plt.ylabel("Displacement (u)")
plt.title(f"Displacement at Midpoint (x = {x[mid]:.4f})")
plt.grid(True)
plt.show()

# =====================================================
# PLOT 3 : Space-Time Heatmap (FIXED: time now increases upward)
# =====================================================

plt.figure(figsize=(10, 6))
plt.imshow(
    U,
    extent=[0, L, 0, T_actual],   # FIX: was [0, L, T_actual, 0]
    aspect='auto',
    cmap='viridis',
    origin='lower'                # FIX: explicitly draw t=0 row at the bottom
)
plt.colorbar(label="Displacement")
plt.xlabel("Position (x)")
plt.ylabel("Time (t)")
plt.title("Space-Time Evolution of the Wave")
plt.show()

# =====================================================
# PLOT 4 : 3D SURFACE
# =====================================================

X, TT = np.meshgrid(x, time)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, TT, U, cmap='viridis', edgecolor='none')
ax.set_xlabel("Position (x)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("Displacement (u)")
ax.set_title("3D Wave Propagation")
plt.show()