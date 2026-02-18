"""
plot_vanishing_cbf_qp.py

Load vanishing CBF-QP trajectory data and generate plots using plot_trajectory.py style.
This script mirrors plot_trajectory.py but for the new algorithm results.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Load trajectory data from the new algorithm
data = np.load('vanishing_cbf_qp_trajectory_data.npy', allow_pickle=True).item()
tgrid = data['tgrid']
X = data['X']
params = data['params']
h_B_vals = data.get('h_B_vals', None)
h_bar_U_vals = data.get('h_bar_U_vals', None)
us = data.get('us', None)
l_t = data.get('l_t', None)

# Extract parameters
rB = params['rB']

# Recreate symbolic functions (same as plot_trajectory.py)
x_sym, y_sym, t_sym = sp.symbols("x y t", real=True)

def r_center(t):
    return 50/(sp.log(t + 1) + 1)

hU_sym = (
    4*x_sym**4
    - 20*x_sym**2*(y_sym - 25)
    - 13*x_sym**2
    + 25*(y_sym - 25)**2
    + 35*(y_sym - 25)
    - 2
)

hU = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")

# Plot settings (same as plot_trajectory.py)
xlim = (-60, 60)
ylim = (-60, 60)
grid_n = 300

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Trajectory in state space (main trajectory plot)
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

# Create barrier field
xs = np.linspace(*xlim, grid_n)
ys = np.linspace(*ylim, grid_n)
XX, YY = np.meshgrid(xs, ys)
HU_grid = hU(XX, YY)

# Unsafe region (obstacle) - same style as plot_trajectory.py
ax1.contourf(XX, YY, HU_grid, levels=[-1e10, 0.0], colors=['lightcoral'], alpha=0.3)
ax1.contour(XX, YY, HU_grid, levels=[0.0], colors='red', linewidths=2.5)

# Ball constraint
th = np.linspace(0, 2*np.pi, 400)
ax1.plot(rB*np.cos(th), rB*np.sin(th), linestyle="--", linewidth=2.5, color='blue', label=f"Ball (r={rB})")

# Trajectory
ax1.plot(X[:, 0], X[:, 1], linewidth=2.5, color='green', label='Vanishing CBF-QP trajectory')
ax1.plot(X[0, 0], X[0, 1], marker="o", markersize=10, color='green', label='Start')
ax1.plot(X[-1, 0], X[-1, 1], marker="^", markersize=10, color='darkgreen', label='End')

# Equilibrium shift l(t)
if l_t is not None:
    ax1.plot(l_t[:, 0], l_t[:, 1], linestyle=":", linewidth=2.0, color='black', label='$l(t)$')

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect("equal", adjustable="box")
ax1.grid(True, alpha=0.3)
ax1.set_xlabel("$x_1$", fontsize=12)
ax1.set_ylabel("$x_2$", fontsize=12)
ax1.set_title("Vanishing CBF-QP: State Trajectory", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)

# ============================================================================
# Plot 2: Ball barrier h_B(x(t)) over time
# ============================================================================
ax2 = plt.subplot(2, 2, 2)
if h_B_vals is not None:
    ax2.plot(tgrid, h_B_vals, 'b-', linewidth=2.5, label='$h_B(x(t))$')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Constraint boundary')
    ax2.fill_between(tgrid, 0, h_B_vals, where=(h_B_vals >= 0), alpha=0.2, color='green', label='Safe region')
    ax2.set_xlabel('Time $t$ (s)', fontsize=11)
    ax2.set_ylabel('$h_B$ (Ball barrier)', fontsize=11)
    ax2.set_title('Ball Barrier: $h_B(x(t)) \\geq 0$ must hold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
else:
    ax2.text(0.5, 0.5, 'Ball barrier data not available', ha='center', va='center', transform=ax2.transAxes)

# ============================================================================
# Plot 3: Safety barrier h_bar_U(x(t)) over time
# ============================================================================
ax3 = plt.subplot(2, 2, 3)
if h_bar_U_vals is not None:
    ax3.plot(tgrid, h_bar_U_vals, 'g-', linewidth=2.5, label='$h_{\\bar{U}}(x(t))$')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Constraint boundary')
    ax3.fill_between(tgrid, 0, h_bar_U_vals, where=(h_bar_U_vals >= 0), alpha=0.2, color='green', label='Safe region')
    ax3.set_xlabel('Time $t$ (s)', fontsize=11)
    ax3.set_ylabel('$h_{\\bar{U}}$ (Safety barrier)', fontsize=11)
    ax3.set_title('Safety Barrier: $h_{\\bar{U}}(x(t)) \\geq 0$ must hold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
else:
    ax3.text(0.5, 0.5, 'Safety barrier data not available', ha='center', va='center', transform=ax3.transAxes)

# ============================================================================
# Plot 4: Control input norm and components
# ============================================================================
ax4 = plt.subplot(2, 2, 4)
if us is not None:
    u_norm = np.linalg.norm(us, axis=1)
    t_us = np.linspace(0, tgrid[-1], len(us))
    ax4.plot(t_us, us[:, 0], 'b-', linewidth=2, label='$u_1(t)$', alpha=0.8)
    ax4.plot(t_us, us[:, 1], 'r-', linewidth=2, label='$u_2(t)$', alpha=0.8)
    ax4.plot(t_us, u_norm, 'k--', linewidth=2.5, label='$\\|u(t)\\|$', alpha=0.7)
    ax4.set_xlabel('Time $t$ (s)', fontsize=11)
    ax4.set_ylabel('Control input', fontsize=11)
    ax4.set_title('Applied Control: Vanishing Feedback', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
else:
    ax4.text(0.5, 0.5, 'Control data not available', ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('vanishing_cbf_qp_analysis.png', dpi=150, bbox_inches='tight')
print("[Saved] vanishing_cbf_qp_analysis.png")
plt.close()

# ============================================================================
# Print summary
# ============================================================================
print("\n" + "="*70)
print("Vanishing CBF-QP Safety Filter: Simulation Summary")
print("="*70)
print(f"Algorithm: {params.get('algorithm', 'VanishingCBFQP')}")
print(f"Initial condition: x0 = {params['x0']}")
print(f"Final state: x_f = {X[-1]}")
print(f"Simulation time: {params['T']} s")
print(f"Time step: {params['dt']} s")
print(f"\nBall constraint B_{rB}(0):")
print(f"  Initial radius: ||x_0|| = {np.linalg.norm(X[0]):.4f}")
print(f"  Final radius:   ||x_f|| = {np.linalg.norm(X[-1]):.4f}")
print(f"  Max radius:     max||x|| = {np.linalg.norm(X, axis=1).max():.4f}")
if h_B_vals is not None:
    print(f"  Min h_B: {h_B_vals.min():.6f} (constraint: h_B >= 0)")
    h_B_violations = np.sum(h_B_vals < -1e-6)
    print(f"  h_B violations: {h_B_violations}")

print(f"\nSafety constraint (unsafe set avoidance):")
if h_bar_U_vals is not None:
    print(f"  Min h_bar_U: {h_bar_U_vals.min():.6f} (constraint: h_bar_U >= 0)")
    h_bar_U_violations = np.sum(h_bar_U_vals < -1e-6)
    print(f"  h_bar_U violations: {h_bar_U_violations}")

print(f"\nControl effort:")
if us is not None:
    u_norm = np.linalg.norm(us, axis=1)
    print(f"  Max ||u||: {u_norm.max():.6f}")
    print(f"  Mean ||u||: {u_norm.mean():.6f}")
    print(f"  Total energy: {np.sum(u_norm**2) * params['dt']:.4f}")
    print(f"  Final ||u||: {u_norm[-1]:.6f} (should vanish to ~0)")

print("="*70 + "\n")
