"""
Load vanishing CBF-QP trajectory data and generate plots.
Supports single or multiple trajectories. Saves figures to ./Figures/.
 
Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path

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
    }]

n_traj = len(all_results)
colors = plt.cm.viridis(np.linspace(0.3, 0.7, n_traj))

x0_center = data['x0_center']
x0_radius = data['x0_radius']

# ========== Plot style parameters ==========

style = {
    'traj_lw':        0.8,
    'traj_alpha':     0.7,
    'ball_lw':        1.2,
    'ball_ls':        '--',
    'ic_ball_lw':     1.2,
    'ic_ball_ls':     '--',
    'l_t_lw':         1.5,
    'l_t_ls':         ':',
    'unsafe_lw':      1.0,
    'unsafe_alpha':   0.25,
    'marker_end_sz':  4,
    'origin_sz':      8,
}

zoom = {
    'center':     x0_center,        # Set after loading data
    'width':      x0_radius * 5,    # Set after loading data
    'height':     x0_radius * 5,    # Set after loading data
    'angle':      0,                # Degrees: 0=right, 90=top, 180=left, 270=bottom (axes fraction from center)
    'distance':   0.3,             # Distance from axes center to inset (axes fraction)
    'inset_size': 0.4,              # Inset width/height as fraction of axes
}

# ========== Barrier field ==========

x_scale, y_scale = data['obstacle']['x_scale'], data['obstacle']['y_scale']
x_shift, y_shift = data['obstacle']['x_shift'], data['obstacle']['y_shift']

x_sym, y_sym = sp.symbols("x y", real=True)
# Scale/Shift
x_s = x_sym / x_scale - x_shift
y_s = y_sym / y_scale - y_shift
# Unsafe set barrier (from vanishing-controller.py)
hU_sym = (4*x_s**4 - 20*x_s**2*y_s - 13*x_s**2
          + 25*y_s**2 + 35*y_s - 2)
hU     = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")

grid_n       = 1000
all_X        = np.vstack([r_['X'] for r_ in all_results])
extent       = max(float(np.abs(all_X).max()), rB) + 5.0
xlim, ylim   = (-extent, extent), (-extent, extent)
xs_g         = np.linspace(*xlim, grid_n)
ys_g         = np.linspace(*ylim, grid_n)
XX, YY       = np.meshgrid(xs_g, ys_g)
HU_grid      = hU(XX, YY)
th           = np.linspace(0, 2*np.pi, 400)
ic_th        = np.linspace(0, 2*np.pi, 400)

# ========== Plot 1: State-space trajectories ==========

fig1, ax1 = plt.subplots(figsize=(8, 8))

ax1.contourf(XX, YY, HU_grid, levels=[-1e10, 0.0],
             colors=['lightcoral'], alpha=style['unsafe_alpha'])
ax1.contour(XX, YY, HU_grid, levels=[0.0],
            colors='red', linewidths=style['unsafe_lw'])
ax1.plot(rB*np.cos(th), rB*np.sin(th),
         style['ball_ls'], linewidth=style['ball_lw'],
         color='blue', label=f'Ball $B_r(0)$, $r={rB}$')

for i, r_ in enumerate(all_results):
    X = r_['X']
    ax1.plot(X[:, 0], X[:, 1],
             linewidth=style['traj_lw'], color=colors[i], alpha=style['traj_alpha'])
    ax1.plot(X[-1, 0], X[-1, 1], '^',
             markersize=style['marker_end_sz'], color=colors[i])

ax1.plot(x0_center[0] + x0_radius*np.cos(ic_th),
         x0_center[1] + x0_radius*np.sin(ic_th),
         style['ic_ball_ls'], linewidth=style['ic_ball_lw'],
         color='green', label=f'IC ball ($r={x0_radius}$)')

l_t = all_results[0]['l_t_vals']
ax1.plot(l_t[:, 0], l_t[:, 1],
         style['l_t_ls'], linewidth=style['l_t_lw'],
         color='black', label='$l(t)$')

ax1.plot(0, 0, '*', markersize=style['origin_sz'],
         color='red', zorder=5, label='Nominal equilibrium')

ax1.set_xlim(xlim); ax1.set_ylim(ylim)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('$x_1$', fontsize=12); ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title(f'Vanishing CBF-QP: State Trajectories ({n_traj})', fontsize=12)
ax1.legend(fontsize=10, loc='upper right')

# Zoom inset
angle_rad = np.deg2rad(zoom['angle'])
ix = 0.5 + zoom['distance'] * np.cos(angle_rad) - zoom['inset_size'] / 2
iy = 0.5 + zoom['distance'] * np.sin(angle_rad) - zoom['inset_size'] / 2
ix = np.clip(ix, 0.01, 1 - zoom['inset_size'] - 0.01)
iy = np.clip(iy, 0.01, 1 - zoom['inset_size'] - 0.01)

axins = ax1.inset_axes([ix, iy, zoom['inset_size'], zoom['inset_size']])
axins.contourf(XX, YY, HU_grid, levels=[-1e10, 0.0],
               colors=['lightcoral'], alpha=style['unsafe_alpha'])
axins.contour(XX, YY, HU_grid, levels=[0.0],
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
ax1.indicate_inset_zoom(axins, edgecolor='black', linestyle=':', linewidth=1.0)

fig1.tight_layout()
fig1.savefig(figures_dir / 'trajectories.png', dpi=600, bbox_inches='tight')
plt.close(fig1)
print("[Saved] Figures/trajectories.png")

# ========== Plot 2: h_U level sets ==========

kappa_U_grid = np.where(HU_grid > 0, HU_grid**4, np.nan)
kappa_log    = np.log1p(kappa_U_grid)
fig2, ax2 = plt.subplots(figsize=(8, 8))
# Safe region background
ax2.contourf(XX, YY, HU_grid, levels=[-1e10, 0.0], colors=['#d4edda'], alpha=1.0)
# kappa_U levels only inside unsafe set
cf = ax2.contourf(XX, YY, kappa_log, levels=60, cmap='YlOrRd')
ax2.contour(XX, YY, HU_grid, levels=[0.0], colors='black', linewidths=1.5)
plt.colorbar(cf, ax=ax2, label='$\\log(1 + \\kappa_U(h_U))$', fraction=0.046, pad=0.04)
ax2.set_xlim(xlim); ax2.set_ylim(ylim)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('$x_1$', fontsize=12); ax2.set_ylabel('$x_2$', fontsize=12)
ax2.set_title('$h_U$ level sets (red = boundary)', fontsize=12)
fig2.tight_layout()
fig2.savefig(figures_dir / 'levelsets_hU.png', dpi=600, bbox_inches='tight')
plt.close(fig2)
print("[Saved] Figures/levelsets_hU.png")

# ========== Plot 3: Ball barrier h_B(x(t)) ==========

fig3, ax3 = plt.subplots(figsize=(8, 4))
for i, r_ in enumerate(all_results):
    ax3.plot(r_['tgrid'], r_['h_B_vals'],
             color=colors[i], linewidth=style['traj_lw'], alpha=style['traj_alpha'])
ax3.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Constraint boundary')
ax3.set_xlabel('Time $t$ (s)', fontsize=11); ax3.set_ylabel('$h_B$', fontsize=11)
ax3.set_title('Ball Barrier $h_B(x(t)) \\geq 0$', fontsize=12)
ax3.grid(True, alpha=0.3); ax3.legend(fontsize=10)
fig3.tight_layout()
fig3.savefig(figures_dir / 'barrier_ball.png', dpi=600, bbox_inches='tight')
plt.close(fig3)
print("[Saved] Figures/barrier_ball.png")

# ========== Plot 4: Safety barrier h_U(x(t)) ==========

fig4, ax4 = plt.subplots(figsize=(8, 4))
for i, r_ in enumerate(all_results):
    ax4.plot(r_['tgrid'], r_['h_U_vals'],
             color=colors[i], linewidth=style['traj_lw'], alpha=style['traj_alpha'])
ax4.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Constraint boundary')
ax4.set_xlabel('Time $t$ (s)', fontsize=11); ax4.set_ylabel('$h_U$', fontsize=11)
ax4.set_title('Safety Barrier $h_U(x(t)) \\leq 0$ must hold', fontsize=12)
ax4.grid(True, alpha=0.3); ax4.legend(fontsize=10)
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
    u_norm = np.linalg.norm(us, ord=np.inf, axis=1)
    ax5.plot(t_us, us[:, 0], color=colors[i], linewidth=style['traj_lw'],
             alpha=style['traj_alpha'])
    ax5.plot(t_us, us[:, 1], color=colors[i], linewidth=style['traj_lw'],
             alpha=style['traj_alpha'], linestyle='--')
    ax5.plot(t_us, u_norm, color=colors[i], linewidth=style['traj_lw']*1.5,
             alpha=style['traj_alpha'])
ax5.set_xlabel('Time $t$ (s)', fontsize=11); ax5.set_ylabel('Control input', fontsize=11)
ax5.set_title('Applied Control: Vanishing Feedback $\\|u(t)\\|_\\infty$', fontsize=12)
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
    u_norm = np.linalg.norm(r_['us'], axis=1)
    print(f"  x0={np.round(r_['x0'],2)}  h_B_viol={viol_B}  h_U_viol={viol_U}"
          f"  max||u||={u_norm.max():.3f}  final||u||={u_norm[-1]:.4f}")
print("="*60 + "\n")