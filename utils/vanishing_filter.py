"""
Vanishing CBF-QP Safety Filter Algorithm

Implements a safety filter that:
1. Defines smooth barriers for safe set S = B_r(0) \ U
2. Uses equilibrium shift l(t) → 0
3. Solves a QP with CBF constraints in unscaled decision variable k
4. Applies vanishing feedback u(t) = k*/s'(t)

Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional

class VanishingFilter:
    """
    Vanishing CBF-QP Safety Filter on S = B_r(0) \ U with equilibrium shift.
    Parameters:
        f :         (Callable)      Autonomous dynamics: x_dot = f(x)
        g :         (Callable)      Control input matrix: appears as g(x,t) * u
        l_0 :       (np.ndarray)    Initial disturbance that shifts equilibrium (when unscaled)
        h_U :       (Callable)      Barrier for unsafe set: U = {x : h_U(x) < 0}
        h_U_grad :  (Callable)      Gradient of h_U
        r :         (float)         Radius of ball B_r(0)
        Q :         (np.ndarray)    Weight matrix for norm ||x||_{p,Q}
        kappa_B :   (Callable)      Class-K function for ball barrier
        kappa_U :   (Callable)      Class-K function for unsafe set barrier
        alpha :     (Callable)      Class-K_infinity schedule for equilibrium shift
        s_t :       (Callable)      Scaling factor for vanishing feedback (default s(t) = 1 + alpha(t))
        l_t :       (Callable)      Time-varying equilibrium shift (default l(t) = l_0 / s(t))
    """
    
    def __init__(
        self,
        f: Callable,
        g: Callable,
        l_0: np.ndarray,
        h_B: Callable,
        h_U: Callable,
        h_B_grad: Callable,
        h_U_grad: Callable,
        r: float,
        Q: Optional[np.ndarray] = None,
        kappa_B: Optional[Callable] = None,
        kappa_U: Optional[Callable] = None,
        alpha: Optional[Callable] = None,
        s_fun: Optional[Callable] = None,
        l_fun: Optional[Callable] = None,
        u_norm_type: Optional[str] = None,
        u_max: Optional[float] = None,
    ):
        self.f = f
        self.g = g
        self.l_0 = l_0
        self.h_B = h_B
        self.h_U = h_U
        self.h_B_grad = h_B_grad
        self.h_U_grad = h_U_grad
        self.n = len(l_0)
        
        self.Q = Q if Q is not None else np.eye(self.n)
        self.kappa_B = kappa_B if kappa_B is not None else (lambda h: 2.0 * h)
        self.kappa_U = kappa_U if kappa_U is not None else (lambda h: 2.0 * h)
        self.alpha = alpha if alpha is not None else (lambda t: t)
        self.s_fun = s_fun if s_fun is not None else (lambda t: 1.0 + self.alpha(t))
        self.l_fun = l_fun if l_fun is not None else (lambda t: self.l_0 / self.s_fun(t))
        
        self.u_norm_type = u_norm_type if u_norm_type is not None else 'inf'
        self.u_max = u_max if u_max is not None else 10.0
    
    # Lie derivatives for CBF constraints
    def L_f_h_B(self, x: np.ndarray) -> float: return float(self.h_B_grad(x) @ self.f(x))
    def L_g_h_B(self, x: np.ndarray, t: float) -> np.ndarray: return self.h_B_grad(x) @ self.g(x, t)
    def L_f_h_U(self, x: np.ndarray) -> float: return float(self.h_U_grad(x) @ self.f(x))
    def L_g_h_U(self, x: np.ndarray, t: float) -> np.ndarray: return self.h_U_grad(x) @ self.g(x, t)
    
    def solve_qp(
        self,
        x: np.ndarray,
        t: float,
        u_nom: Optional[np.ndarray] = None,
        eps: float = 1e-9,
    ) -> np.ndarray:
        """
        Solve the vanishing CBF-QP:
            min 0.5 ||k||_Q^2
            s.t.
              L_f h_B + L_g h_B (k/s') + ∇h_B^T l >= -κ_B(h_B)
              L_f h_U + L_g h_U (k/s') + ∇h_U^T l >= -κ_U(h_U)
        
        Returns k* (unscaled control), which is then applied as u = k* / s'(t).
        """
        # Control dimension
        m = self.g(x, t).shape[1]
            
        # Equilibrium shift, barrier values and derivatives
        s_t = self.s_fun(t)
        l_t = self.l_fun(t)
        h_B = self.h_B(x)
        h_U = self.h_U(x)
        grad_h_B = self.h_B_grad(x)
        grad_h_U = self.h_U_grad(x)
        L_f_h_B = self.L_f_h_B(x)
        L_g_h_B = self.L_g_h_B(x, t)
        L_f_h_U = self.L_f_h_U(x)
        L_g_h_U = self.L_g_h_U(x, t)
        
        # CBF constraints: A k <= b
        epsilon = 0.1  # Tightening margin, tuneable
        A1 = -L_g_h_B / s_t  # coefficient for k, negated for <= form
        b1 = L_f_h_B + float(grad_h_B @ l_t) + self.kappa_B(h_B) - epsilon
        A2 = -L_g_h_U / s_t
        b2 = L_f_h_U + float(grad_h_U @ l_t) + self.kappa_U(h_U) - epsilon
        # Stack constraints
        A = np.vstack([A1, A2])
        b = np.array([b1, b2])
        
        # Initial guess: nominal control or zero
        if u_nom is not None:
            k0 = u_nom / (s_t)
        else:
            k0 = np.zeros(m)
        
        # Constraints for scipy
        constraints = []
        for i in range(A.shape[0]):
            constraints.append({
                'type': 'ineq',
                'fun': lambda k, i=i: b[i] - A[i, :] @ k,
                'jac': lambda k, i=i: -A[i, :],
            })
        
        # Control bounds
        if self.u_norm_type == 'inf':
            for i in range(m):
                constraints.append({'type': 'ineq', 'fun': lambda k, i=i: self.u_max - k[i]})
                constraints.append({'type': 'ineq', 'fun': lambda k, i=i: self.u_max + k[i]})
        elif self.u_norm_type == 'l1':
            constraints.append({'type': 'ineq', 'fun': lambda k: self.u_max - np.sum(np.abs(k)),
                                 'jac': lambda k: -np.sign(k)})
        
        # Objective — swap by commenting/uncommenting
        # def objective(k):      return 0.5 * float(k @ self.Q @ k)
        # def objective_grad(k): return self.Q @ k
        g_xt = self.g(x, t)
        drift = self.f(x) + l_t
        def objective(k):      return -float(drift @ (g_xt @ k)) # + 1e-5 * float(k @ k)
        def objective_grad(k): return -(g_xt.T @ drift) # + 2e-5 * k
        
        # Solve with SLSQP
        result = minimize(
            objective,
            k0,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            options={'ftol': eps, 'maxiter': 100},
        )
 
        if not result.success:
            print(f"Warning: Did not converge at t={t:.4f}. Message: {result.message}")
            # print(f"  b1={b1:.4f}, b2={b2:.4f}, h_B={h_B:.4f}, h_U={h_U:.4f}")
            # print(f"  L_g_h_U={L_g_h_U}, u_max={self.u_max}")
        
        return result.x
    
    def compute_control(self, x: np.ndarray, t: float, u_nom: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
        """Compute the vanishing control u(t)."""
        # Solve QP to get k*
        k_star = self.solve_qp(x, t, u_nom=u_nom)        
        # Apply vanishing feedback
        s_t = self.s_fun(t)
        u_t = k_star / s_t         
        
        return u_t
    
    def _dynamics(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """ Compute state derivative: x_dot = f(x) + g(x,t) u + l(t)."""
        return self.f(x) + self.g(x, t) @ u + self.l_fun(t)
    
    def step(self, x: np.ndarray, t: float, dt: float, u_nom=None) -> Tuple[np.ndarray, np.ndarray]:
        """"Compute control and integrate dynamics for one time step."""
        u = self.compute_control(x, t, u_nom)
        sol = solve_ivp(
            fun=lambda t_, x_: self._dynamics(x_, u, t_),
            t_span=(t, t + dt),
            y0=x,
            method='RK45',
            dense_output=False,
            rtol=1e-8,
            atol=1e-12,
        )
        return sol.y[:, -1], u

def run_simulation(
    filter_obj: VanishingFilter,
    x0: np.ndarray,
    t_sim: float,
    dt: float,
    u_nom_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run simulation with the vanishing CBF-QP filter.
    Returns:
        ts :        (np.ndarray)    Time points
        xs :        (np.ndarray)    State trajectory
        us :        (np.ndarray)    Control trajectory
        h_B_vals :  (np.ndarray)    Ball barrier values
    """
    N = int(t_sim / dt)
    ts = np.linspace(0, t_sim, N + 1)
    
    xs = np.zeros((N + 1, x0.shape[0]))
    us = np.zeros((N, len(filter_obj.l_0)))  # m = dim of control
    
    x = x0.copy()
    xs[0] = x
    
    for k in range(N):
        t = ts[k]
        u_nom = u_nom_func(x, t) if u_nom_func else None
        x, u = filter_obj.step(x, t, dt, u_nom=u_nom)
        xs[k + 1] = x
        us[k] = u
        
    return ts, xs, us
