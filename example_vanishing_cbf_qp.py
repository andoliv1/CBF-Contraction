"""
Example: Vanishing CBF-QP Safety Filter with 2D Dynamics

Demonstrates the algorithm on a simple 2D system with:
- Autonomous dynamics: x_dot = -x
- Control-affine: g(x,t) = I (identity)
- Ball constraint: B_r(0) with r = 1.5
- Unsafe set: U = {x : h_U(x) > 0} (from vanishing-controller.py)
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from vanishing_cbf_qp_safety_filter import VanishingCBFQPFilter, run_simulation


# ============================================================================
# Define the system using symbolic computation (like vanishing-controller.py)
# ============================================================================

# Symbolic variables
x_sym, y_sym = sp.symbols("x y", real=True)

# Unsafe set barrier (from vanishing-controller.py)
hU_sym = (
    4*x_sym**4
    - 20*x_sym**2*(y_sym - 25)
    - 13*x_sym**2
    + 25*(y_sym - 25)**2
    + 35*(y_sym - 25)
    - 2
)

# Compute partial derivatives symbolically
dhdx_U_sym = sp.diff(hU_sym, x_sym)
dhdy_U_sym = sp.diff(hU_sym, y_sym)

# Create lambdified functions for h_U and its gradient
h_U_func = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")
grad_hU_func = sp.lambdify((x_sym, y_sym), (dhdx_U_sym, dhdy_U_sym), "numpy")


def f(x: np.ndarray) -> np.ndarray:
    """Autonomous dynamics: x_dot = -x (stable at origin)."""
    return -x


def g(x: np.ndarray, t: float) -> np.ndarray:
    """Control matrix: 2x2 identity (full actuation)."""
    return np.eye(2)


def h_U(x: np.ndarray) -> float:
    """
    Unsafe set barrier using sympy lambdified expression.
    U = {x : h_U(x) > 0}
    """
    return float(h_U_func(x[0], x[1]))


def h_U_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of h_U using sympy partial derivatives."""
    grad_vals = grad_hU_func(x[0], x[1])
    return np.array([float(grad_vals[0]), float(grad_vals[1])])


# ============================================================================
# Create and configure the filter
# ============================================================================

# Parameters
r = 50.0  # ball radius
l_0 = np.array([-30.0, 30.0])  # equilibrium shift (unscaled)
Q = np.eye(2)  # weight matrix

# Class-K functions (simple linear)
def kappa_B(h):
    return 2.0 * h

def kappa_U(h):
    return 5.0 * h

# Class-K_infinity schedules (from vanishing-controller.py style)
def alpha(t):
    return 1

def alpha_prime(t):
    return t + 1


# Create filter
filter_obj = VanishingCBFQPFilter(
    f=f,
    g=g,
    l_0=-l_0,
    h_U=h_U,
    h_U_grad=h_U_grad,
    r=r,
    Q=Q,
    kappa_B=kappa_B,
    kappa_U=kappa_U,
    alpha=alpha,
    alpha_prime=alpha_prime,
)

# ============================================================================
# Run simulation
# ============================================================================

# Initial condition (from vanishing-controller.py)
# x0 = np.array([0.0, 26.0])
x0 = np.array([1.0, 1.0])

# Simulation parameters
t_sim = 10.0  # simulation time (from vanishing-controller.py: T=40)
dt = 0.01  # time step (from vanishing-controller.py)

# Nominal controller (e.g., set-tracking)
def u_nom(x, t):
    """Simple stabilizing controller."""
    return np.zeros(2)


# Run simulation
ts, xs, us, h_B_vals = run_simulation(
    filter_obj=filter_obj,
    x0=x0,
    t_sim=t_sim,
    dt=dt,
    u_nom_func=u_nom,
)

# Equilibrium shift trajectory l(t) = l_0 / (1 + alpha(t))
l_t = np.array([l_0 / (1.0 + alpha(t)) for t in ts])

# ============================================================================
# Save trajectory data (compatible with plot_trajectory.py format)
# ============================================================================

# Compute barrier values for saving
h_bar_U_vals = np.array([filter_obj.h_bar_U(x) for x in xs])

# Save trajectory data
data = {
    'tgrid': ts,
    'X': xs,
    'X_origin': None,  # No origin trajectory in this example
    'params': {
        'x0': x0,
        'T': t_sim,
        'dt': dt,
        'rB': r,
        'ENABLE_SAFETY': True,
        'ENABLE_REACHABILITY': False,
        'PLOT_ORIGIN_TRAJECTORY': False,
        'algorithm': 'VanishingCBFQP',
        'l0': l_0,
    },
    'h_B_vals': h_B_vals,
    'h_bar_U_vals': h_bar_U_vals,
    'us': us,
    'l_t': l_t,
}
np.save('vanishing_cbf_qp_trajectory_data.npy', data, allow_pickle=True)
print("[Saved] vanishing_cbf_qp_trajectory_data.npy")
print("[Info] Use plot_trajectory.py or plot_vanishing_cbf_qp.py to generate plots")
print("\n" + "="*60)
print("Vanishing CBF-QP Safety Filter: Simulation Complete")
print("="*60)
print(f"Initial condition: x0 = {x0}")
print(f"Final state: x_f = {xs[-1]}")
print(f"Simulation time: {t_sim} s")
print(f"Time step: {dt} s")
print(f"\nTrajectory data saved to: vanishing_cbf_qp_trajectory_data.npy")
print("To generate plots, run: python plot_vanishing_cbf_qp.py")
print("="*60 + "\n")
