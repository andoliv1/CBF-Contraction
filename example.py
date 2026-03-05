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
from vanishing_filter import VanishingFilter, run_simulation


# ========== Define the system using symbolic computation ==========

x_sym, y_sym = sp.symbols("x y", real=True)

# Obstacle shape parameters
x_scale = 1.5    # > 1 stretches wider, < 1 narrows
y_scale = 1.0    # > 1 stretches taller, < 1 flattens
x_shift = 0.0    # shifts obstacle left/right
y_shift = 15.0   # shifts obstacle up/down
# Scale/Shift
x_s = x_sym / x_scale - x_shift
y_s = y_sym / y_scale - y_shift
# Unsafe set barrier (from vanishing-controller.py)
hU_sym = (4*x_s**4 - 20*x_s**2*y_s - 13*x_s**2
          + 25*y_s**2 + 35*y_s - 2)

# Symbolic derivatives and lambdify for numerical evaluation
dhdx_U_sym   = sp.diff(hU_sym, x_sym)
dhdy_U_sym   = sp.diff(hU_sym, y_sym)
h_U_func     = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")
grad_hU_func = sp.lambdify((x_sym, y_sym), (dhdx_U_sym, dhdy_U_sym), "numpy")

# ========== Parameters ==========

r = 30.0                        # Ball radius
l_0 = np.array([-20.0, 20.0])   # Equilibrium shift (unscaled)
Q = np.eye(2)                   # Weight matrix
t_sim = 20.0                    # Simulation time
dt = 0.001                      # Time step

# Initial-conditions
x0_center = np.array([0.0, 18.0])
x0_radius = 2
n_points = 10

# Class-K/K_infty functions
def kappa_B(h): return 2.0 * h
def kappa_U(h): return h ** 10
def alpha(t):   return t

def s_fun(t):   return 1 / (1 + alpha(t))   # Scaling for equilibrium shift (vanishing schedule)
def l_fun(t):   return l_0 / (1 + t/4)      # Time-varying equilibrium shift

# Control constraints
u_max = 100.0
u_norm_type = 'inf'

# ========== System dynamics and barrier functions ==========

def f(x):        return -x                              # Autonomous dynamics: stable at origin                    
def g(x, t):     return np.eye(2)                       # Control matrix: 2x2 identity (full actuation)
def h_U(x):      return float(h_U_func(x[0], x[1]))     # Unsafe set barrier function
def h_U_grad(x):                                        # Gradient of h_U (returns tuple), convert to numpy array
    gv = grad_hU_func(x[0], x[1])                   
    return np.array([float(gv[0]), float(gv[1])])       
def u_nom(x, t): return np.zeros(2)                     # Nominal control (zero in this example)

# ========== Grid of initial conditions inside IC ball ==========

def sample_initial_conditions(center, radius, n_points, seed=None):
    """Sample n_points uniformly at random inside a ball of given radius around center."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, n_points)
    r = radius * np.sqrt(rng.uniform(0, 1, n_points))  # sqrt for uniform area distribution
    pts = center + np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    return pts

x0_list = sample_initial_conditions(x0_center, x0_radius, n_points=n_points, seed=42)
print(f"[Info] Running {len(x0_list)} trajectories (n_points={len(x0_list)}, r={x0_radius})")

# ========== Create filter (shared, stateless) ==========

filter_obj = VanishingFilter(
    f=f, g=g, l_0=l_0, h_U=h_U, h_U_grad=h_U_grad,
    r=r, Q=Q, kappa_B=kappa_B, kappa_U=kappa_U,
    alpha=alpha, s_fun=s_fun, l_fun=l_fun,
    u_max=u_max, u_norm_type=u_norm_type,
)

# ========== Run simulations ==========

all_results = []
for i, x0 in enumerate(x0_list):
    print(f"  [{i+1}/{len(x0_list)}] x0 = {x0}")
    ts, xs, us = run_simulation(filter_obj, x0, t_sim, dt, u_nom)
    h_B_vals = np.array([filter_obj.h_B(x) for x in xs])
    h_U_vals = np.array([filter_obj.h_U(x) for x in xs])
    l_t_vals = np.array([filter_obj.l_fun(t) for t in ts])
    all_results.append({
        'x0':           x0,
        'tgrid':        ts,
        'X':            xs,
        'us':           us,
        'h_B_vals':     h_B_vals,
        'h_U_vals':     h_U_vals,
        'l_t_vals':     l_t_vals,
    })

# ========== Save ==========

data = {
    'all_results':    all_results,
    'x0_center': x0_center,
    'x0_radius': x0_radius,
    'params': {
        'T': t_sim, 'dt': dt, 'rB': r,
        'algorithm': 'VanishingCBFQP',
        'l0': l_0,
        'n_trajectories': len(x0_list),
    },
    'obstacle': {
        'x_scale': x_scale,
        'y_scale': y_scale,
        'x_shift': x_shift,
        'y_shift': y_shift,
    },
}
np.save('trajectory_data.npy', data, allow_pickle=True)
print("[Saved] trajectory_data.npy")

# ========== Summary ==========

print("\n" + "="*60)
print(f"Trajectories: {len(x0_list)}, IC ball: center={x0_center}, radius={x0_radius}")
print(f"Simulation: T={t_sim}s, dt={dt}s")
for r_ in all_results:
    viol_B = np.sum(r_['h_B_vals'] < -1e-6)
    viol_U = np.sum(r_['h_U_vals'] < -1e-6)
    print(f"  x0={np.round(r_['x0'], 2)}  h_B_viol={viol_B}  h_U_viol={viol_U}  x_f={np.round(r_['X'][-1], 4)}")
print("="*60 + "\n")