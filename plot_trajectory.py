"""
plot_trajectory.py

Load saved trajectory data and generate plots without re-running simulation.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Load trajectory data
data = np.load('trajectory_data.npy', allow_pickle=True).item()
tgrid = data['tgrid']
X = data['X']
X_origin = data['X_origin']
params = data['params']

# Extract parameters
rB = params['rB']

# Recreate symbolic functions (needed for plotting contours)
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

hS_sym = - ((x_sym + r_center(t_sym))**2 + (y_sym - r_center(t_sym))**2 - 10)

hU = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")
hS = sp.lambdify((x_sym, y_sym, t_sym), hS_sym, "numpy")

# Plot settings
xlim = (-60, 60)
ylim = (-60, 60)
grid_n = 300


def make_gif(tgrid, X, X_origin, out_path="vanishing_trajectory.gif"):
    xs = np.linspace(*xlim, grid_n)
    ys = np.linspace(*ylim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    HU_grid = hU(XX, YY)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
    ax.set_title("Algorithm 1 (no slack): obstacle solid, S_t dashed", fontsize=14)

    # obstacle boundary - filled unsafe region
    ax.contourf(XX, YY, HU_grid, levels=[-1e10, 0.0], colors=['lightcoral'], alpha=0.3)
    ax.contour(XX, YY, HU_grid, levels=[0.0], colors='red', linewidths=2.5)

    # ball B
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(rB*np.cos(th), rB*np.sin(th), linestyle="--", linewidth=2.5, color='blue', label=f"B (r={rB})")

    traj_line, = ax.plot([], [], linewidth=2.5, color='green', label="trajectory")
    point, = ax.plot([], [], marker="o", markersize=8, color='green')
    
    # Origin trajectory (shows origin is not an equilibrium) - only if enabled
    origin_traj_line = None
    origin_point = None
    if X_origin is not None:
        origin_traj_line, = ax.plot([], [], linewidth=2.5, color='orange', linestyle=':', label="trajectory from origin")
        origin_point, = ax.plot([], [], marker="s", markersize=8, color='orange')

    st = {"obj": None}
    ax.legend(loc="upper right", fontsize=10)

    nframes = 100
    frame_idx = np.linspace(0, len(tgrid)-1, nframes, dtype=int)

    def update(i):
        k = int(frame_idx[i])
        t = float(tgrid[k])
        traj_line.set_data(X[:k+1,0], X[:k+1,1])
        point.set_data([X[k,0]], [X[k,1]])
        
        # Update origin trajectory if it exists
        if X_origin is not None:
            origin_traj_line.set_data(X_origin[:k+1,0], X_origin[:k+1,1])
            origin_point.set_data([X_origin[k,0]], [X_origin[k,1]])

        if st["obj"] is not None:
            for coll in st["obj"].collections:
                coll.remove()

        HS_grid = hS(XX, YY, t)
        st["obj"] = ax.contour(XX, YY, HS_grid, levels=[0.0], colors='purple', linewidths=2.5, linestyles="--")
        
        return_objs = [traj_line, point]
        if X_origin is not None:
            return_objs.extend([origin_traj_line, origin_point])
        return return_objs

    anim = FuncAnimation(fig, update, frames=nframes, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=25, bitrate=1800))
    plt.close(fig)
    return out_path


def final_plot(tgrid, X, X_origin, out_path="vanishing_final.png"):
    xs = np.linspace(*xlim, grid_n)
    ys = np.linspace(*ylim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    
    fig = plt.figure(figsize=(16, 14))
    
    # Unsafe set contour with fill
    plt.contourf(XX, YY, hU(XX, YY), levels=[-1e10, 0.0], colors=['lightcoral'], alpha=0.3)
    plt.contour(XX, YY, hU(XX, YY), levels=[0.0], colors='red', linewidths=2.5, label='Unsafe set (hU=0)')
    
    # Target set S_T at t=0 (initial)
    t_start = float(tgrid[0])
    contour_start = plt.contour(XX, YY, hS(XX, YY, t_start), levels=[0.0], colors='purple', linewidths=2.0, linestyles=":", alpha=0.6)
    plt.clabel(contour_start, inline=True, fontsize=10, fmt='S(0)')
    
    # Target set S_T at t=T (final)
    t_end = float(tgrid[-1])
    contour_end = plt.contour(XX, YY, hS(XX, YY, t_end), levels=[0.0], colors='purple', linewidths=2.5, linestyles="--")
    plt.clabel(contour_end, inline=True, fontsize=10, fmt='S(T)')
    
    # Add line showing S_t trajectory center movement with small outline arrows
    # Calculate center positions at start and end
    # For hS = -((x + r_center(t))^2 + (y - r_center(t))^2 - 10), center is at (-r_center(t), r_center(t))
    def get_center(t_val):
        r = 50 / (np.log(t_val + 1) + 1)
        return np.array([-r, r])
    
    center_start = get_center(t_start)
    center_end = get_center(t_end)
    
    # Plot center trajectory as a dashed line with small outline arrows
    dx = center_end[0] - center_start[0]
    dy = center_end[1] - center_start[1]
    
    # Draw the dashed line
    plt.plot([center_start[0], center_end[0]], [center_start[1], center_end[1]], 
             linewidth=1.5, color='purple', linestyle='--', alpha=0.7, label='S_t center trajectory')
    
    # Add small arrow heads along the trajectory to show direction
    n_arrows = 4
    for i in range(1, n_arrows):
        t_frac = i / n_arrows
        x_pos = center_start[0] + t_frac * dx
        y_pos = center_start[1] + t_frac * dy
        
        # Normalize direction vector
        dir_len = np.sqrt(dx**2 + dy**2)
        dir_x = dx / dir_len
        dir_y = dy / dir_len
        
        # Perpendicular vector
        perp_x = -dir_y
        perp_y = dir_x
        
        # Arrow head triangle: point and two base points
        arrow_length = 0.8
        arrow_width = 0.5
        point = [x_pos + arrow_length * dir_x, y_pos + arrow_length * dir_y]
        base_left = [x_pos - 0.3 * arrow_length * dir_x - arrow_width * perp_x, 
                     y_pos - 0.3 * arrow_length * dir_y - arrow_width * perp_y]
        base_right = [x_pos - 0.3 * arrow_length * dir_x + arrow_width * perp_x, 
                      y_pos - 0.3 * arrow_length * dir_y + arrow_width * perp_y]
        
        arrow_head = plt.Polygon([point, base_left, base_right], 
                                 closed=True, edgecolor='purple', facecolor='purple',
                                 linewidth=1.0, alpha=0.7, zorder=4)
        plt.gca().add_patch(arrow_head)
    
    plt.scatter([center_start[0]], [center_start[1]], s=150, marker='^', color='purple', 
                edgecolors='black', linewidths=2, zorder=5, label='S_t center (t=0)')
    plt.scatter([center_end[0]], [center_end[1]], s=150, marker='v', color='purple', 
                edgecolors='black', linewidths=2, zorder=5, label='S_t center (t=T)')
    
    # Ball B
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(rB*np.cos(th), rB*np.sin(th), linestyle="--", linewidth=2.5, color='blue', label=f'Ball B (r={rB})')
    
    # Main trajectory
    plt.plot(X[:,0], X[:,1], linewidth=2.5, color='green', label='Trajectory')
    plt.scatter([X[0,0]],[X[0,1]], s=100, marker="o", color='green', edgecolors='black', linewidths=2, label='Trajectory start', zorder=5)
    plt.scatter([X[-1,0]],[X[-1,1]], s=100, marker="x", color='red', linewidths=3, label='Trajectory end', zorder=5)
    
    # Origin trajectory (shows origin is not an equilibrium) - only if enabled
    if X_origin is not None:
        plt.plot(X_origin[:,0], X_origin[:,1], linewidth=2.5, color='orange', linestyle=':', label='Trajectory from origin')
        plt.scatter([0.0],[0.0], s=120, marker="s", color='orange', edgecolors='black', linewidths=2, label='Origin (not equilibrium)', zorder=5)
    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.title("Final state with obstacle (solid) and S_T (dashed)", fontsize=16)
    plt.legend(loc="upper right", fontsize=11)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import sys
    
    print("[Loaded] trajectory_data.npy")
    print(f"[Info] Parameters: x0={params['x0']}, T={params['T']}, dt={params['dt']}")
    print(f"[Info] ENABLE_SAFETY={params['ENABLE_SAFETY']}, ENABLE_REACHABILITY={params['ENABLE_REACHABILITY']}")
    print(f"[Info] PLOT_ORIGIN_TRAJECTORY={params['PLOT_ORIGIN_TRAJECTORY']}")
    
    # Check if user wants to skip GIF generation
    skip_gif = "--no-gif" in sys.argv
    
    if skip_gif:
        print("[Info] Skipping GIF generation...")
        png = final_plot(tgrid, X, X_origin)
        print("[Saved]", png)
    else:
        gif = make_gif(tgrid, X, X_origin)
        png = final_plot(tgrid, X, X_origin)
        print("[Saved]", gif)
        print("[Saved]", png)
