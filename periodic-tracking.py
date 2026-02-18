import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Limit system: xdot = -x + sin(t) is contracting and has a unique 2π-periodic orbit:
# x_p(t) = 0.5 (sin t - cos t)

B = 0.25

def s(t):
    return 1.0 - 1.0/(1.0 + t)          # bounded, non-periodic drift -> 1

def r(t):
    return 1.0/(1.0 + t)**2             # shrinking width

def alpha(t):
    rt = r(t)
    return B/(rt**2)                    # huge gain near the moving center

def delta(x, t):
    z = (x - s(t)) / r(t)
    return alpha(t) * (s(t) - x) * np.exp(-(z**2))

def xdot_scalar(t, x):
    return -x + np.sin(t) + delta(x, t)

def fun_ivp(t, y):
    return np.array([xdot_scalar(t, y[0])])

t0, tf = 0.0, 80.0
t_eval = np.linspace(t0, tf, 8001)
x0_list = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

traj = []
for x0 in x0_list:
    sol = solve_ivp(fun_ivp, (t0, tf), np.array([x0]), t_eval=t_eval,
                    method="Radau", rtol=1e-8, atol=1e-10)
    traj.append(sol.y[0])
traj = np.vstack(traj)

x_p = 0.5*(np.sin(t_eval) - np.cos(t_eval))  # limit periodic orbit

# 1) trajectories vs x_p and s(t)
plt.figure()
for i, x0 in enumerate(x0_list):
    plt.plot(t_eval, traj[i], label=f"x0={x0:g}")
plt.plot(t_eval, x_p, linewidth=2, label="limit periodic orbit x_p(t)")
plt.plot(t_eval, s(t_eval), linewidth=2, label="drift target s(t)")
plt.xlabel("t"); plt.ylabel("x(t)")
plt.title("Bounded counterexample: trajectories track s(t) (non-periodic), not x_p(t)")
plt.legend(ncol=2, fontsize=9)
plt.show()

# 2) error to x_p(t)
plt.figure()
for i, x0 in enumerate(x0_list):
    plt.semilogy(t_eval, np.abs(traj[i] - x_p) + 1e-14, label=f"|x - x_p|, x0={x0:g}")
plt.xlabel("t"); plt.ylabel("absolute error (log scale)")
plt.title("Error to the limit periodic orbit does NOT vanish")
plt.legend(ncol=2, fontsize=9)
plt.show()

# 3) bounded envelope
plt.figure()
plt.plot(t_eval, np.max(traj, axis=0), label="max over initial conditions")
plt.plot(t_eval, np.min(traj, axis=0), label="min over initial conditions")
plt.xlabel("t"); plt.ylabel("envelope")
plt.title("Trajectories remain bounded (compact envelope)")
plt.legend()
plt.show()

# 4) pointwise convergence delta(x_fixed,t)->0
t_check = np.linspace(0, tf, 3001)
x_fixed_list = [-1.0, 0.0, 0.7, 1.0, 1.5]
plt.figure()
for xf in x_fixed_list:
    plt.semilogy(t_check, np.abs(delta(xf, t_check)) + 1e-300, label=f"|delta({xf:g},t)|")
plt.xlabel("t"); plt.ylabel("magnitude (log scale)")
plt.title("Pointwise convergence: delta(x,t)->0 for each fixed x")
plt.legend(ncol=2, fontsize=9)
plt.show()
