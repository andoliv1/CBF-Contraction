# Vanishing CBF-QP Safety Filter

Reference implementation for:

> A. Oliveira, A. Honarpisheh, M. Bozdag, and M. Sznaier,
> "Time-Varying Perturbations of Contractive Systems With an Application to Safe Stabilization,"
> *IEEE Control Systems Letters*, vol. 10, pp. 673–678, 2026.
> [doi:10.1109/LCSYS.2026.3701872](https://doi.org/10.1109/LCSYS.2026.3701872)

## What this does

The filter steers a nominally contractive system to a desired equilibrium while keeping
the state inside a prescribed safe set

```
S = B_r(0) \ U,        U = { x : h_U(x) > 0 }
```

rendering `S` forward invariant. Safety and convergence are handled together by
combining three ingredients:

1. **Smooth barriers** for the ball `B_r(0)` and the unsafe set `U`.
2. **A time-varying equilibrium shift** `l(t) = l_0 / s(t)`, driven by a class-K∞
   schedule `alpha(t)`, so the target moves to the true equilibrium as `t → ∞`.
3. **Vanishing feedback** `u(t) = k*/s'(t)`, where `k*` solves a QP in the unscaled
   decision variable subject to CBF constraints on both barriers.

Because the applied input vanishes as the equilibrium shift does, the closed loop
recovers the nominal contractive behaviour asymptotically rather than fighting it.

## Layout

| Path | Contents |
| --- | --- |
| `utils/vanish_filter_2.py` | `VanishingFilter` class and `run_simulation` — the algorithm |
| `utils/barriers.py` | `make_hU`, `make_hB` — symbolic barriers and their gradients |
| `utils/animate.py` | Trajectory animation |
| `linear/` | Linear example: `x_dot = -x`, `g(x,t) = I` |
| `nonlinear/` | Nonlinear example: `x1_dot = -2x1 + x2 - sin(x1)`, `x2_dot = -2x2 - x1 - cos(x2) - 1` |
| `plot_l_g_h.py` | Level sets of `h_U` and its gradient |
| `Figures/` | Generated figures |

## Requirements

Python 3 with `numpy`, `scipy`, `sympy`, `matplotlib`, and `gurobipy`.

The QP is solved with Gurobi, which needs a license — the free academic license is
sufficient to reproduce every example here.

```bash
pip install numpy scipy sympy matplotlib gurobipy
```

## Running

Each mode runs the example, generates the plots, and writes the animation:

```bash
python run_pipeline.py --mode linear
python run_pipeline.py --mode nonlinear
```

Obstacle geometry (`x_scale`, `y_scale`, `y_shift`, smoothing `mode`), ball radius `r`,
and the initial equilibrium shift `l_0` are set at the top of `linear/example.py` and
`nonlinear/example_nonlinear.py`.

## Contact

Andreas Oliveira — andreasoliveira792@gmail.com
