"""
Barrier function definitions for the Vanishing CBF-QP Safety Filter.

Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
import sympy as sp


def make_hU(x_scale=1.0, y_scale=1.0, x_shift=0.0, y_shift=15.0, mode='raw', sigma=None, delta=None):
    """
    Build the unsafe set barrier h_U and its gradient.
    Modes for h_U:
        'raw'   : h_U(x) = h_U_func(x) (no smoothing)
        'tanh'  : h_U(x) = tanh(h_U_func(x) / sigma) (Tanh smoothing, sigma controls steepness)
        'log'   : h_smooth(h) = h - delta*log(h/delta + 1) (Log smoothing near boundary, raw elsewhere)
    Returns:
        h_U          : Callable (x: np.ndarray) -> float
        h_U_grad     : Callable (x: np.ndarray) -> np.ndarray
        h_U_grid_func: Callable (XX, YY) -> np.ndarray  (for plotting)
    """
    x_sym, y_sym = sp.symbols("x y", real=True)
    x_s = x_sym / x_scale - x_shift
    y_s = y_sym / y_scale - y_shift
    # Crescent-shaped barrier centered on the origin when unshifted
    hU_sym = (4*x_s**4 - 20*x_s**2*y_s - 13*x_s**2 + 25*y_s**2 + 35*y_s - 2)
    dhdx_sym     = sp.diff(hU_sym, x_sym)
    dhdy_sym     = sp.diff(hU_sym, y_sym)
    h_U_func     = sp.lambdify((x_sym, y_sym), hU_sym, "numpy")
    grad_hU_func = sp.lambdify((x_sym, y_sym), (dhdx_sym, dhdy_sym), "numpy")
    
    if mode=='raw':
        def h_U(x): return float(h_U_func(x[0], x[1]))

        def h_U_grad(x):
            gv = grad_hU_func(x[0], x[1])
            return np.array([float(gv[0]), float(gv[1])])

        def h_U_grid_func(XX, YY): return h_U_func(XX, YY)

    elif mode=='tanh':
        assert sigma is not None, "tanh mode requires sigma"
        
        def h_U(x): return float(np.tanh(h_U_func(x[0], x[1]) / sigma))

        def h_U_grad(x):
            gv   = grad_hU_func(x[0], x[1])
            grad = np.array([float(gv[0]), float(gv[1])])
            h    = h_U_func(x[0], x[1])
            return (1.0 / sigma) * (1.0 - np.tanh(h / sigma)**2) * grad

        def h_U_grid_func(XX, YY): return np.tanh(h_U_func(XX, YY) / sigma)
        
    elif mode == 'log':
        assert delta is not None, "log mode requires delta"

        def h_U(x):
            h = float(h_U_func(x[0], x[1]))
            return h - delta * np.log(h / delta + 1) if h > 0 else h

        def h_U_grad(x):
            h    = float(h_U_func(x[0], x[1]))
            gv   = grad_hU_func(x[0], x[1])
            grad = np.array([float(gv[0]), float(gv[1])])
            scale = h / (h + delta) if h > 0 else 1.0
            return scale * grad

        def h_U_grid_func(XX, YY):
            h = h_U_func(XX, YY)
            return np.where(h > 0, h - delta * np.log(h / delta + 1), h)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return h_U, h_U_grad, h_U_grid_func


def make_hB(r, Q=None):
    """
    Build the ball barrier h_B and its gradient.
    h_B(x) = r - ||x||_{Q}  (positive inside B_r(0), zero on boundary)
    Returns:
        h_B          : Callable (x: np.ndarray) -> float
        h_B_grad     : Callable (x: np.ndarray) -> np.ndarray
        h_B_grid_func: Callable (XX, YY) -> np.ndarray  (for plotting, assumes Q=I)
    """
    def h_B(x): return float(r - np.sqrt(x @ Q @ x))

    def h_B_grad(x):
        norm_x = np.sqrt(float(x @ Q @ x))
        if norm_x < 1e-10:
            return np.zeros(len(x))
        return -(Q @ x) / norm_x

    def h_B_grid_func(XX, YY):
        """ 
        Q weighted norm: ||x||_Q = sqrt(x^T Q x)
        For 2D grid: x = [XX, YY], so x^T Q x = Q[0,0]*XX^2 + 2*Q[0,1]*XX*YY + Q[1,1]*YY^2 
        """
        Q_ = np.eye(2) if Q is None else Q
        return r - np.sqrt(Q_[0,0]*XX**2 + 2*Q_[0,1]*XX*YY + Q_[1,1]*YY**2)

    return h_B, h_B_grad, h_B_grid_func