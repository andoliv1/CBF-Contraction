"""
Example: Vanishing CBF-QP Safety Filter with 2D Dynamics

Demonstrates the algorithm on a simple 2D system with:
- Autonomous dynamics: x_dot = -x
- Control-affine: g(x,t) = I (identity)
- Ball constraint: B_r(0) with r = 1.5
- Unsafe set: U = {x : h_U(x) > 0} (from vanishing-controller.py)

Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from vanishing_cbf_qp_safety_filter import VanishingCBFQPFilter, run_simulation


# ========== Define the system using symbolic computation ==========

x_sym, y_sym = sp.symbols("x y", real=True)     # Symbolic variables

# Unsafe set barrier (from vanishing-controller.py)
hU_sym = (4*x_sym**4 - 20*x_sym**2*(y_sym - 25) - 13*x_sym**2 + 25*(y_sym - 25)**2 + 35*(y_sym - 25) - 2)

dhdx_U_sym = sp.diff(hU_sym, x_sym)                             # Symbolic partial derivatives
dhdy_U_sym = sp.diff(hU_sym, y_sym)
h_U_func = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")         # Lambdify for numerical evaluation
grad_hU_func = sp.lambdify((x_sym, y_sym), 
                           (dhdx_U_sym, dhdy_U_sym), "numpy")

# ========== Parameters ==========

r = 100.0                       # Ball radius
l_0 = np.array([-50.0, 50.0])   # Equilibrium shift (unscaled)
Q = np.eye(2)                   # Weight matrix
x0 = np.array([0.0, 26.0])      # Initial state
t_sim = 10.0                    # Simulation time
dt = 0.01                       # Time step

# Class-K/K_infty functions
def kappa_B(h): return 2.0 * h
def kappa_U(h): return h ** 4 
def alpha(t):   return t

def s_fun(t):   return 1 / (1 + alpha(t))   # Scaling for equilibrium shift (vanishing schedule)
def l_fun(t):   return l_0 * s_fun(t)       # Time-varying equilibrium shift

# ========= Define system dynamics and barrier functions ==========

def f(x: np.ndarray) -> np.ndarray:
    """Autonomous dynamics: x_dot = -x (stable at origin)."""
    return -x

def g(x: np.ndarray, t: float) -> np.ndarray:
    """Control matrix: 2x2 identity (full actuation)."""
    return np.eye(2)

def h_U(x: np.ndarray) -> float:
    """ Unsafe set barrier using sympy lambdified expression:   U = {x : h_U(x) > 0}."""
    return float(h_U_func(x[0], x[1]))

def h_U_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of h_U using sympy partial derivatives."""
    grad_vals = grad_hU_func(x[0], x[1])
    return np.array([float(grad_vals[0]), float(grad_vals[1])])

# ========== Create the filter ==========

filter_obj = VanishingCBFQPFilter(
    f=f,
    g=g,
    l_0=l_0,
    h_U=h_U,
    h_U_grad=h_U_grad,
    r=r,
    Q=Q,
    kappa_B=kappa_B,
    kappa_U=kappa_U,
    alpha=alpha,
    s_fun=s_fun,
    l_fun=l_fun,
)

# ========== Run simulation ==========

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
l_t = np.array([l_0 / (1 + alpha(t)) for t in ts])

# ========== Save trajectory data ==========

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
