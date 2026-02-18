
"""
vanishing_clf_cbf_qp_noslack_gif_fast2.py

No-slack Algorithm 1 + GIF output, optimized (2D minimal-QP solved by checking boundary candidates).

QP:
  min 0.5 ||u||_Q^2
  s.t.
    Lf h_C + Lg h_C u >= -kC h_C
    Lf h_U + Lg h_U u >= -kU h_U         (enforces safety: keep h_U >= 0)
    ∂t V + Lf V + Lg V u <= -cV V^γ      (enforces reachability to S_t)

Algorithm 1:
  if x ∉ int(B): apply u = u*
  else: apply u = α(t) u*

Produces:
  vanishing_trajectory.gif  (obstacle solid, S_t dashed, ball B dashed)
  vanishing_final.png
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import minimize

# -----------------------------
# Symbolic definitions
# -----------------------------
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

# hS_sym = - (
#     4*(sp.Rational(1, 2)*x_sym)**4
#     - 20*(sp.Rational(1, 2)*x_sym)**2*(y_sym - 5)
#     - 13*(sp.Rational(1, 2)*x_sym)**2
#     + 25*(y_sym - 5)**2
#     + 35*(y_sym - 5)
#     - 100/(sp.log(t_sym + 1) + 1)
# )

hS_sym = - ((x_sym + r_center(t_sym))**2 + (y_sym - r_center(t_sym))**2 - 10)

dhdxU_sym = sp.diff(hU_sym, x_sym)
dhdyU_sym = sp.diff(hU_sym, y_sym)

dhdxS_sym = sp.diff(hS_sym, x_sym)
dhdyS_sym = sp.diff(hS_sym, y_sym)
dtdhS_sym = sp.diff(hS_sym, t_sym)

hU = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")
grad_hU = sp.lambdify((x_sym, y_sym), (dhdxU_sym, dhdyU_sym), "numpy")

hS = sp.lambdify((x_sym, y_sym, t_sym), hS_sym, "numpy")
grad_hS = sp.lambdify((x_sym, y_sym, t_sym), (dhdxS_sym, dhdyS_sym), "numpy")
dt_hS = sp.lambdify((x_sym, y_sym, t_sym), dtdhS_sym, "numpy")

# -----------------------------
# QP solver using scipy
# -----------------------------
def min_norm_qp_2d(Q, A, b, tol=1e-9):
    """
    Solve min 0.5 u^T Q u s.t. A u <= b using scipy.optimize.minimize.
    
    Args:
        Q: 2x2 cost matrix
        A: m x 2 constraint matrix
        b: m-dim constraint vector
    
    Returns:
        u: optimal control input
    """
    def objective(u):
        return 0.5 * float(u @ Q @ u)
    
    def objective_grad(u):
        return Q @ u
    
    # Constraint: A @ u <= b  =>  A @ u - b <= 0
    constraints = []
    for i in range(A.shape[0]):
        constraints.append({
            'type': 'ineq',
            'fun': lambda u, i=i: b[i] - (A[i, :] @ u),
            'jac': lambda u, i=i: -A[i, :]
        })
    
    # Initial guess
    u0 = np.zeros(2)
    
    # Solve
    result = minimize(
        objective,
        u0,
        method='SLSQP',
        jac=objective_grad,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not result.success:
        raise RuntimeError(f"QP solver failed: {result.message}")
    
    return result.x


# -----------------------------
# Parameters
# -----------------------------
R_C = 20.0
rB = 14.0
kC = 2.0
kU = 2.0
cV = 100.0
gammaV = 0.5
umax = 70
Q = np.eye(2)

# Constraint toggles
ENABLE_SAFETY = True        # Include obstacle CBF (h_U >= 0)
ENABLE_REACHABILITY = True  # Include reachability constraint (h_S)

# Plotting toggles
PLOT_ORIGIN_TRAJECTORY = False  # Plot trajectory starting from origin

# Simulation tuned to finish quickly (edit up if you want more time)
x0 = np.array([0.0,26])
T = 40
dt = 0.001
steps = int(T/dt)

# Plot window / contour grid
xlim = (-60,60)
ylim = (-60,60)
grid_n = 300  # Increased resolution for detailed contours

def alpha_schedule(t):
    return 1/(1 + t**2)

def kappa(k, s):
    return k*s


def compute_constraints(t, x):
    x1, x2 = float(x[0]), float(x[1])
    xvec = np.array([x1, x2], dtype=float)
    f = -xvec

    A_list = []
    b_list = []

    # bounds
    A_list += [[ 1.0, 0.0],
               [-1.0, 0.0],
               [ 0.0, 1.0],
               [ 0.0,-1.0]]
    b_list += [umax, umax, umax, umax]

    # obstacle CBF for safety: h_U >= 0 invariant
    if ENABLE_SAFETY:
        hU_val = float(hU(x1, x2))
        dhdxU, dhdyU = grad_hU(x1, x2)
        gU = np.array([float(dhdxU), float(dhdyU)], dtype=float)
        Lf_hU = float(gU @ f)
        # -(gU·u) <= Lf_hU + kU*hU
        A_list.append([-gU[0], -gU[1]])
        b_list.append(Lf_hU + kappa(kU, hU_val))

    # V constraint when hS<0:  gS·u >= cV V^γ - dtS - gS·f  =>  -(gS·u) <= -rhs
    if ENABLE_REACHABILITY:
        hS_val = float(hS(x1, x2, t))
        if hS_val < 0.0:
            print("[Info] t={:.2f}: (hS={:.3f})".format(t, hS_val))
            V = -hS_val
            dhdxS, dhdyS = grad_hS(x1, x2, t)
            gS = np.array([float(dhdxS), float(dhdyS)], dtype=float)
            dtS = float(dt_hS(x1, x2, t))
            Lf_hS = float(gS @ f)  # Lie derivative Lf(hS) = grad(hS) @ f
            rhs = cV*(V**gammaV) - dtS - Lf_hS  # Combined: dtS + Lf_hS + Lg_hS·u <= -cV*V^γ
            A_list.append([-gS[0], -gS[1]])
            b_list.append(-rhs)

    A = np.array(A_list, dtype=float)
    b = np.array(b_list, dtype=float)
    return A, b


def u_star(t, x):
    A, b = compute_constraints(t, x)
    try:
        return min_norm_qp_2d(Q, A, b)
    except RuntimeError as e:
        print(f"[QP Infeasible] t={t:.2f}, x={x}, using fallback u=0")
        return np.zeros(2)


def algorithm1_control(t, x):
    u = u_star(t, x)
    if np.linalg.norm(x) < rB:
        return alpha_schedule(t) * u
    return u


# RK4 integration
def dyn(x, u):
    return -x + u

def rk4_step(t, x, dt):
    u1 = algorithm1_control(t, x)
    k1 = dyn(x, u1)

    x2 = x + 0.5*dt*k1
    u2 = algorithm1_control(t + 0.5*dt, x2)
    k2 = dyn(x2, u2)

    x3 = x + 0.5*dt*k2
    u3 = algorithm1_control(t + 0.5*dt, x3)
    k3 = dyn(x3, u3)

    x4 = x + dt*k3
    u4 = algorithm1_control(t + dt, x4)
    k4 = dyn(x4, u4)

    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate():
    tgrid = np.linspace(0, T, steps+1)
    X = np.zeros((steps+1, 2))
    X[0,:] = x0
    
    # Also simulate from origin if enabled
    X_origin = None
    if PLOT_ORIGIN_TRAJECTORY:
        X_origin = np.zeros((steps+1, 2))
        X_origin[0,:] = np.array([0.0, 0.0])

    hits = 0
    for k in range(steps):
        try:
            X[k+1,:] = rk4_step(tgrid[k], X[k,:], dt)
            if PLOT_ORIGIN_TRAJECTORY:
                X_origin[k+1,:] = rk4_step(tgrid[k], X_origin[k,:], dt)
        except Exception as e:
            print(f"[Exception at step {k}] {e}")
            X[k+1,:] = X[k,:]  # Stay at current position
            if PLOT_ORIGIN_TRAJECTORY:
                X_origin[k+1,:] = X_origin[k,:]
        if float(hU(X[k+1,0], X[k+1,1])) < 0.0:
            hits += 1
    if hits > 0:
        print(f"[WARN] Entered unsafe (hU<0) in {hits} steps.")
    
    # Save trajectory data
    data = {
        'tgrid': tgrid,
        'X': X,
        'X_origin': X_origin,
        'params': {
            'x0': x0,
            'T': T,
            'dt': dt,
            'rB': rB,
            'ENABLE_SAFETY': ENABLE_SAFETY,
            'ENABLE_REACHABILITY': ENABLE_REACHABILITY,
            'PLOT_ORIGIN_TRAJECTORY': PLOT_ORIGIN_TRAJECTORY
        }
    }
    np.save('trajectory_data.npy', data, allow_pickle=True)
    print("[Saved] trajectory_data.npy")
    
    return tgrid, X, X_origin


if __name__ == "__main__":
    print("[Sanity] hU(0,10)=", float(hU(0,10)), "hU(0,12)=", float(hU(0,12)), "hU(0,0)=", float(hU(0,0)))
    tgrid, X, X_origin = simulate()
    print("[Done] Simulation complete. Use plot_trajectory.py to generate plots.")

