"""
Load vanishing CBF-QP trajectory data and generate plots.
Supports single or multiple trajectories. Saves figures to ./Figures/.
 
Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from barriers import make_hU, make_hB

# ========== Setup ==========

figures_dir = Path("Figures")
figures_dir.mkdir(exist_ok=True)

data        = np.load('trajectory_data.npy', allow_pickle=True).item()
params      = data['params']
rB          = params['rB']
all_results = data.get('all_results', None)

# Support single-trajectory legacy format
if all_results is None:
    all_results = [{
        'x0':       params['x0'],
        'tgrid':    data['tgrid'],
        'X':        data['X'],
        'us':       data['us'],
        'h_B_vals': data['h_B_vals'],
        'h_U_vals': data['h_U_vals'],
        'l_t_vals': data['l_t_vals'],
        'u_max':    data['u_max'],
        'u_norm_type': data['u_norm_type'],
    }]

n_traj = len(all_results)
colors = plt.cm.viridis(np.linspace(0.3, 0.7, n_traj))

x0_center = data['x0_center']
x0_radius = data['x0_radius']

u_max = data['u_max']
u_norm_type = data['u_norm_type']
if u_norm_type == "inf":
    u_norm_ord = np.inf
    u_norm_label = r'\infty'
elif u_norm_type == "1":
    u_norm_ord = 1
    u_norm_label = '1'
else:
    raise ValueError(f"Unknown norm type: {u_norm_type}")

# ========== Plot style parameters ==========

style = {
    'traj_lw':        1.2,
    'traj_alpha':     1,
    'ball_lw':        1.5,
    'ball_ls':        '--',
    'ic_ball_lw':     1.2,
    'ic_ball_ls':     '--',
    'l_t_lw':         1.5,
    'l_t_ls':         '--',
    'legend_fs':      20,
    'axis_fs'  :      20,
    'tick_fs':        20,
    'unsafe_lw':      1.0,
    'unsafe_alpha':   0.25,
    'marker_end_sz':  4,
    'origin_sz':      10,
    'log_scale':      True,
}

zoom = {
    'center':     x0_center + [-6,3],   # Set after loading data
    'width':      x0_radius * 5,        # Set after loading data
    'height':     x0_radius * 5,        # Set after loading data
    'angle':      0,                    # Degrees: 0=right, 90=top, 180=left, 270=bottom (axes fraction from center)
    'distance':   0.4,                  # Distance from axes center to inset (axes fraction)
    'inset_size': 0.45,                 # Inset width/height as fraction of axes
}

# ========== Barrier field ==========

# REPLACE entire barrier field section with:
x_scale  = data['hU']['x_scale']
y_scale  = data['hU']['y_scale']
x_shift  = data['hU']['x_shift']
y_shift  = data['hU']['y_shift']
sigma    = data['hU'].get('sigma', None)

_, _, h_U_grid_func = make_hU(x_scale=x_scale, y_scale=y_scale,
                               x_shift=x_shift, y_shift=y_shift, sigma=sigma)
_, _, h_B_grid_func = make_hB(rB)

grid_n     = 1000
all_X      = np.vstack([r_['X'] for r_ in all_results])
extent     = max(float(np.abs(all_X).max()), rB) + 5.0
xlim, ylim = (-extent, extent), (-extent, extent)
xs_g       = np.linspace(*xlim, grid_n)
ys_g       = np.linspace(*ylim, grid_n)
XX, YY     = np.meshgrid(xs_g, ys_g)
h_U_grid   = h_U_grid_func(XX, YY)
th         = np.linspace(0, 2*np.pi, 400)
ic_th      = np.linspace(0, 2*np.pi, 400)

# ========== Plot 1: State-space trajectories ==========

fig1, ax1 = plt.subplots(figsize=(8, 8))

ax1.contourf(XX, YY, h_U_grid, levels=[-1e10, 0.0],
             colors=['lightcoral'], alpha=style['unsafe_alpha'])
ax1.contour(XX, YY, h_U_grid, levels=[0.0],
            colors='red', linewidths=style['unsafe_lw'])
ax1.plot(rB*np.cos(th), rB*np.sin(th),
         style['ball_ls'], linewidth=style['ball_lw'],
         color='red', label=r'$B_{30}(0)$')

for i, r_ in enumerate(all_results):
    X = r_['X']
    ax1.plot(X[:, 0], X[:, 1],
             linewidth=style['traj_lw'], color=colors[i], alpha=style['traj_alpha'])
    ax1.plot(X[-1, 0], X[-1, 1], '^',
             markersize=style['marker_end_sz'], color=colors[i])

ax1.plot(x0_center[0] + x0_radius*np.cos(ic_th),
         x0_center[1] + x0_radius*np.sin(ic_th),
         style['ic_ball_ls'], linewidth=style['ic_ball_lw'],
         color='green', label=r'$\mathcal{X}_0$')

l_t = all_results[0]['l_t_vals']
ax1.plot(l_t[:, 0], l_t[:, 1],
         style['l_t_ls'], linewidth=style['l_t_lw'],
         color='black', label='$l(t)$')

ax1.plot(0, 0, '*', markersize=style['origin_sz'], color='red')

ax1.set_xlim(xlim); ax1.set_ylim(ylim)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('$x_1$', fontsize=style['axis_fs']) 
ax1.set_ylabel('$x_2$', fontsize=style['axis_fs'])
ax1.tick_params(axis='both', labelsize=style['tick_fs'])
ax1.legend(fontsize=style['legend_fs'], loc='upper right')

# Zoom inset
angle_rad = np.deg2rad(zoom['angle'])
ix = 0.5 + zoom['distance'] * np.cos(angle_rad) - zoom['inset_size'] / 2
iy = 0.5 + zoom['distance'] * np.sin(angle_rad) - zoom['inset_size'] / 2
ix = np.clip(ix, 0.01, 1 - zoom['inset_size'] - 0.01)
iy = np.clip(iy, 0.01, 1 - zoom['inset_size'] - 0.01)

axins = ax1.inset_axes([ix, iy, zoom['inset_size'], zoom['inset_size']])
axins.contourf(XX, YY, h_U_grid, levels=[-1e10, 0.0],
               colors=['lightcoral'], alpha=style['unsafe_alpha'])
axins.contour(XX, YY, h_U_grid, levels=[0.0],
              colors='red', linewidths=style['unsafe_lw'])
axins.plot(x0_center[0] + x0_radius*np.cos(ic_th),
           x0_center[1] + x0_radius*np.sin(ic_th),
           style['ic_ball_ls'], linewidth=style['ic_ball_lw'], color='green')
for i, r_ in enumerate(all_results):
    X = r_['X']
    axins.plot(X[:, 0], X[:, 1],
               linewidth=style['traj_lw'], color=colors[i], alpha=style['traj_alpha'])

zc, zw, zh = zoom['center'], zoom['width'], zoom['height']
axins.set_xlim(zc[0] - zw/2, zc[0] + zw/2)
axins.set_ylim(zc[1] - zh/2, zc[1] + zh/2)
axins.set_xticklabels([]); axins.set_yticklabels([])
axins.set_xticks([]); axins.set_yticks([])
ax1.indicate_inset_zoom(axins, edgecolor='black', linestyle=':', linewidth=2)

fig1.tight_layout()
fig1.savefig(figures_dir / 'trajectories.png', dpi=600, bbox_inches='tight')
plt.close(fig1)
print("[Saved] Figures/trajectories.png")

# ========== Plot 2: h_U level sets ==========

fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.contourf(XX, YY, h_U_grid, levels=[-1e10, 0.0], colors=['#d4edda'], alpha=style['unsafe_alpha'])
if style['log_scale']:
    plot_grid = np.log1p(np.where(h_U_grid > 0, h_U_grid, np.nan))
    cbar_label = '$\\log(1 + h_U)$'
else:
    plot_grid = np.where(h_U_grid > 0, h_U_grid, np.nan)
    cbar_label = '$h_U$'
cf = ax2.contourf(XX, YY, plot_grid, levels=60, cmap='YlOrRd')
ax2.contour(XX, YY, h_U_grid, levels=[0.0], colors='black', linewidths=style['unsafe_lw'])
cbar = plt.colorbar(cf, ax=ax2, label=cbar_label, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=style['tick_fs'])
cbar.set_label(cbar_label, fontsize=style['axis_fs'])

# Trajectories:
for i, r_ in enumerate(all_results):
    X = r_['X']
    ax2.plot(X[:, 0], X[:, 1], linewidth=style['traj_lw'], color=colors[i], alpha=style['traj_alpha'])
    ax2.plot(X[-1, 0], X[-1, 1], '^', markersize=style['marker_end_sz'], color=colors[i])  # end
    
ax2.set_xlim(xlim); ax2.set_ylim(ylim)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('$x_1$', fontsize=style['axis_fs'])
ax2.set_ylabel('$x_2$', fontsize=style['axis_fs'])
ax2.tick_params(axis='both', labelsize=style['tick_fs'])
fig2.tight_layout()
fig2.savefig(figures_dir / 'levelsets_hU.png', dpi=600, bbox_inches='tight')
plt.close(fig2)
print("[Saved] Figures/levelsets_hU.png")

# ========== Plot 3: Ball barrier h_B(x(t)) ==========

fig3, ax3 = plt.subplots(figsize=(8, 4))
for i, r_ in enumerate(all_results):
    ax3.plot(r_['tgrid'], r_['h_B_vals'],
             color=colors[i], linewidth=style['traj_lw'], alpha=style['traj_alpha'])
ax3.axhline(0, color='r', linestyle='--', linewidth=1.5, label='$h_B(x(t)) = 0$')
ax3.set_xlabel('Time $t$ (s)', fontsize=style['axis_fs'])
ax3.set_ylabel('$h_B$', fontsize=style['axis_fs'])
ax3.tick_params(axis='both', labelsize=style['tick_fs'])
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=style['legend_fs'])
fig3.tight_layout()
fig3.savefig(figures_dir / 'barrier_ball.png', dpi=600, bbox_inches='tight')
plt.close(fig3)
print("[Saved] Figures/barrier_ball.png")

# ========== Plot 4: Safety barrier h_U(x(t)) ==========

fig4, ax4 = plt.subplots(figsize=(8, 4))
for i, r_ in enumerate(all_results):
    ax4.plot(r_['tgrid'], r_['h_U_vals'],
             color=colors[i], linewidth=style['traj_lw'], alpha=style['traj_alpha'])
ax4.axhline(0, color='r', linestyle='--', linewidth=1.5, label='h_U(x(t)) = 0')
ax4.set_xlabel('Time $t$ (s)', fontsize=style['axis_fs'])
ax4.set_ylabel('$h_U$', fontsize=style['axis_fs'])
ax4.tick_params(axis='both', labelsize=style['tick_fs'])
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=style['legend_fs'])
fig4.tight_layout()
fig4.savefig(figures_dir / 'barrier_safety.png', dpi=600, bbox_inches='tight')
plt.close(fig4)
print("[Saved] Figures/barrier_safety.png")

# ========== Plot 5: Control input ==========

fig5, ax5 = plt.subplots(figsize=(8, 4))
for i, r_ in enumerate(all_results):
    us    = r_['us']
    tgrid = r_['tgrid']
    t_us  = np.linspace(0, tgrid[-1], len(us))
    u_norm = np.linalg.norm(us, ord=u_norm_ord, axis=1)
    ax5.plot(t_us, u_norm, color=colors[i], linewidth=style['traj_lw']*1.5,
             alpha=style['traj_alpha'])
ax5.axhline(u_max, color='r', linestyle='--', linewidth=1.5, label='$u_{max}$')
ax5.set_xlabel('Time $t$ (s)', fontsize=style['axis_fs'])
ax5.set_ylabel(fr'$\|u\|_{{{u_norm_label}}}$', fontsize=style['axis_fs']) 
ax5.tick_params(axis='both', labelsize=style['tick_fs'])
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(figures_dir / 'control.png', dpi=600, bbox_inches='tight')
plt.close(fig5)
print("[Saved] Figures/control.png")

# ========== Summary ==========

print("\n" + "="*60)
print(f"Trajectories: {n_traj}, Algorithm: {params.get('algorithm','VanishingCBFQP')}")
for r_ in all_results:
    viol_B = np.sum(r_['h_B_vals'] < -1e-6)
    viol_U = np.sum(r_['h_U_vals'] < 1e-6)
    u_norm = np.linalg.norm(r_['us'], ord=u_norm_ord, axis=1)
    print(f"  x0={np.round(r_['x0'],2)}  h_B_viol={viol_B}  h_U_viol={viol_U}"
          f"  max||u||={u_norm.max():.3f}  final||u||={u_norm[-1]:.4f}")
print("="*60 + "\n")