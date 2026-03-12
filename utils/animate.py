"""
Create a GIF animation from vanishing CBF-QP trajectory data.
Saves output to ./Figures/trajectory.gif.

Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.barriers import make_hU


# ========== Setup ==========

figures_dir = ROOT / "Figures"
figures_dir.mkdir(exist_ok=True)

output_file = figures_dir / "trajectory.gif"


def load_trajectory_data():
    path = ROOT / "trajectory_data.npy"
    try:
        loaded = np.load(path, allow_pickle=True).item()
        print(f"[Info] Loaded data from {path}")
        return loaded
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find trajectory_data.npy in the current directory."
        ) from exc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not load trajectory_data.npy due to environment incompatibility "
            "(e.g., 'numpy._core'). Re-run your simulation script to regenerate the file "
            "in this environment."
        ) from exc


data = load_trajectory_data()
params = data["params"]
rB = params["rB"]
all_results = data.get("all_results", None)

# Support single-trajectory legacy format
if all_results is None:
    all_results = [{
        "x0": data["params"]["x0"],
        "tgrid": data["tgrid"],
        "X": data["X"],
        "us": data["us"],
        "h_B_vals": data["h_B_vals"],
        "h_U_vals": data["h_U_vals"],
        "l_t_vals": data["l_t_vals"],
    }]

n_traj = len(all_results)
colors = plt.cm.viridis(np.linspace(0.3, 0.7, n_traj))

x0_center = data["x0_center"]
x0_radius = data["x0_radius"]

# ========== Barrier field ==========

x_scale = data["hU"]["x_scale"]
y_scale = data["hU"]["y_scale"]
x_shift = data["hU"]["x_shift"]
y_shift = data["hU"]["y_shift"]
sigma = data["hU"].get("sigma", None)
mode = data["hU"].get("mode", "raw")
delta = data["hU"].get("delta", None)

_, _, h_U_grid_func = make_hU(
    x_scale=x_scale,
    y_scale=y_scale,
    x_shift=x_shift,
    y_shift=y_shift,
    mode=mode,
    sigma=sigma,
    delta=delta,
)

all_X = np.vstack([r_["X"] for r_ in all_results])
extent = max(float(np.abs(all_X).max()), rB) + 5.0
xlim, ylim = (-extent, extent), (-extent, extent)

grid_n = 300
xs_g = np.linspace(*xlim, grid_n)
ys_g = np.linspace(*ylim, grid_n)
XX, YY = np.meshgrid(xs_g, ys_g)
h_U_grid = h_U_grid_func(XX, YY)

th = np.linspace(0, 2 * np.pi, 400)
ic_th = np.linspace(0, 2 * np.pi, 400)

# ========== Animation setup ==========

fig, ax = plt.subplots(figsize=(8, 8))

# Static background
ax.contourf(XX, YY, h_U_grid, levels=[-1e10, 0.0], colors=["lightcoral"], alpha=0.25)
ax.contour(XX, YY, h_U_grid, levels=[0.0], colors="red", linewidths=1.0)
ax.plot(rB * np.cos(th), rB * np.sin(th), "--", linewidth=1.5, color="red", label=f"$B_r(0)$, $r={rB}$")
ax.plot(
    x0_center[0] + x0_radius * np.cos(ic_th),
    x0_center[1] + x0_radius * np.sin(ic_th),
    "--",
    linewidth=1.2,
    color="green",
    label=f"$x_0$ Ball, $r={x0_radius}$",
)
ax.plot(0, 0, "*", markersize=10, color="red", zorder=5)

l_t = all_results[0]["l_t_vals"]
ax.plot(l_t[:, 0], l_t[:, 1], "--", linewidth=1.5, color="black", label="$l(t)$")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.set_xlabel("$x_1$", fontsize=16)
ax.set_ylabel("$x_2$", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=12, loc="upper right")

# Dynamic artists
traj_lines = []
head_markers = []
for i in range(n_traj):
    line, = ax.plot([], [], linewidth=1.5, color=colors[i], alpha=1.0)
    head, = ax.plot([], [], "o", markersize=4, color=colors[i])
    traj_lines.append(line)
    head_markers.append(head)

time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=12)

# Assume common time grid length (as produced by the simulation script)
N = min(len(r_["X"]) for r_ in all_results)
common_t = all_results[0]["tgrid"][:N]

# Downsample frames for smaller/faster GIF
max_frames = 300
step = max(1, N // max_frames)
frame_indices = list(range(0, N, step))
if frame_indices[-1] != N - 1:
    frame_indices.append(N - 1)


def init():
    for line, head in zip(traj_lines, head_markers):
        line.set_data([], [])
        head.set_data([], [])
    time_text.set_text("")
    return [*traj_lines, *head_markers, time_text]


def update(frame_idx):
    k = frame_indices[frame_idx]

    for i, result in enumerate(all_results):
        X = result["X"][:N]
        traj_lines[i].set_data(X[: k + 1, 0], X[: k + 1, 1])
        head_markers[i].set_data([X[k, 0]], [X[k, 1]])

    time_text.set_text(f"t = {common_t[k]:.2f} s")
    return [*traj_lines, *head_markers, time_text]


ani = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    init_func=init,
    blit=True,
    interval=40,
)

writer = PillowWriter(fps=20)
ani.save(output_file, writer=writer)
plt.close(fig)

print(f"[Saved] {output_file}")
