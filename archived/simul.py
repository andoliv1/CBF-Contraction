import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---

R0 = 1.2       # initial radius of S_t
Rf = 0.2       # final radius of S_t
t_shrink_start = 1.0
t_shrink_end   = 11.0   # slow shrink so tracking is easier

T  = 12.0      # simulation horizon
dt = 0.001
k_gain = 15.0  # gain for set-tracking control
eps = 1e-6

# Unsafe set (put far away so it is avoided trivially)
c_u = np.array([2.0, 2.0])
R_u = 0.5

# --- Time-varying safe radius ---

def R_of_t(t: float) -> float:
    """Radius of the time-varying safe set S_t."""
    if t <= t_shrink_start:
        return R0
    elif t >= t_shrink_end:
        return Rf
    else:
        alpha = (t - t_shrink_start) / (t_shrink_end - t_shrink_start)
        return (1 - alpha) * R0 + alpha * Rf

# --- Autonomous dynamics and controller ---

def f_aut(x: np.ndarray) -> np.ndarray:
    """Autonomous dynamics: dot x = -x."""
    return -x

def u_set(x: np.ndarray, t: float) -> np.ndarray:
    """
    Controller that tries to minimize F(x,t) = 0.5 (||x|| - R(t))^2,
    i.e., to track the circle of radius R(t).
    """
    r = np.linalg.norm(x)
    R = R_of_t(t)
    if r < eps:
        # Small radius: kick along x1-axis to get off the origin.
        return k_gain * np.array([R, 0.0])
    grad = (r - R) * x / (r + eps)  # gradient of F wrt x
    return -k_gain * grad

def f_cl(x: np.ndarray, t: float) -> np.ndarray:
    """Closed-loop dynamics: dot x = -x + u(x,t)."""
    return f_aut(x) + u_set(x, t)

# --- Simulation ---

def simulate(x0: np.ndarray):
    N = int(T / dt)
    xs  = np.zeros((N+1, 2))
    Vs  = np.zeros(N+1)
    dS  = np.zeros(N+1)
    Rs  = np.zeros(N+1)

    x = x0.copy()
    xs[0] = x
    Vs[0] = np.dot(x, x)
    Rs[0] = R_of_t(0.0)
    dS[0] = abs(np.linalg.norm(x) - Rs[0])

    for k in range(N):
        t = k * dt
        dx = f_cl(x, t)
        x = x + dt * dx
        xs[k+1] = x
        Vs[k+1] = np.dot(x, x)
        Rs[k+1] = R_of_t((k+1) * dt)
        dS[k+1] = abs(np.linalg.norm(x) - Rs[k+1])

    ts = np.linspace(0, T, N+1)
    return ts, xs, Vs, dS, Rs

# Initial condition: very close to the origin
x0 = np.array([0.05, 0.0])
ts, xs, Vs, dS, Rs = simulate(x0)

radii = np.linalg.norm(xs, axis=1)
d_to_u = np.linalg.norm(xs - c_u, axis=1)

print("Initial radius:", np.linalg.norm(x0))
print("Max radius along trajectory:", radii.max())
print("Final radius:", radii[-1])
print("Initial distance to S_0:", dS[0])
print("Final distance to S_T:", dS[-1])
print("Min distance to unsafe center:", d_to_u.min(), "(R_u =", R_u, ")")

# Check where V increases and inspect distance behavior (optional)
dV   = np.diff(Vs) / dt
ddS  = np.diff(dS) / dt
print("Does V ever increase?        ", np.any(dV > 1e-4), " max dV:", dV.max())
print("Distance to S_t mostly hill-climbs down; max ddS:", ddS.max())

# --- Plots ---

theta = np.linspace(0, 2*np.pi, 300)
# Ball B of radius 1
circle_B = np.vstack([1.0*np.cos(theta), 1.0*np.sin(theta)])
# Unsafe set U
circle_U = c_u.reshape(2,1) + np.vstack([R_u*np.cos(theta), R_u*np.sin(theta)])

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))

# 1) State trajectory in the plane
ax[0].plot(circle_B[0], circle_B[1], 'k--', label='Ball B (r=1)')
ax[0].plot(circle_U[0], circle_U[1], 'r',   label='Unsafe set U')
ax[0].plot(xs[:, 0], xs[:, 1], 'b',        label='Trajectory')
ax[0].plot(0, 0, 'ko', label='Origin')
ax[0].set_aspect('equal', 'box')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_title('State trajectory')
ax[0].legend(loc='best')

# 2) Natural Lyapunov V(t) = ||x||^2
ax[1].plot(ts, Vs)
ax[1].set_xlabel('t')
ax[1].set_ylabel('V(x) = ||x||^2')
ax[1].set_title('Natural Lyapunov V(t)')

# 3) Distance to moving set S_t and radius R(t)
ax[2].plot(ts, dS, label='dist(x(t), S_t)')
ax[2].plot(ts, Rs, '--', label='R(t)')
ax[2].set_xlabel('t')
ax[2].set_ylabel('distance / radius')
ax[2].set_title('Distance to moving safe set S_t')
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()
