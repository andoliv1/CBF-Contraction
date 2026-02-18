"""
Vanishing CBF-QP Safety Filter Algorithm (Algorithm: Vanishing CBF-QP safety filter)

Implements a safety filter that:
1. Defines smooth barriers for safe set S = B_r(0) \ U
2. Uses equilibrium shift l(t) → 0
3. Solves a QP with CBF constraints in unscaled decision variable k
4. Applies vanishing feedback u(t) = k*/s'(t)

The algorithm enforces:
- State constraint within ball B_r(0): h_B(x) = r - ||x||_{p,Q} >= 0
- Safety constraint outside unsafe set U: h_bar_U(x) = -h_U(x) >= 0
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional


class VanishingCBFQPFilter:
    """
    Vanishing CBF-QP Safety Filter on S = B_r(0) \ U with equilibrium shift.
    
    Parameters:
    -----------
    f : Callable
        Autonomous dynamics: x_dot = f(x)
    g : Callable
        Control input matrix: appears as g(x,t) * u
    l_0 : np.ndarray
        Initial disturbance that shifts equilibrium (when unscaled)
    h_U : Callable
        Barrier for unsafe set: U = {x : h_U(x) > 0}
    h_U_grad : Callable
        Gradient of h_U
    r : float
        Radius of ball B_r(0)
    Q : np.ndarray
        Weight matrix for norm ||x||_{p,Q}
    kappa_B : Callable
        Class-K function for ball barrier
    kappa_U : Callable
        Class-K function for unsafe set barrier
    alpha : Callable
        Class-K_infinity schedule for equilibrium shift
    alpha_prime : Callable
        Class-K_infinity schedule for control scaling
    """
    
    def __init__(
        self,
        f: Callable,
        g: Callable,
        l_0: np.ndarray,
        h_U: Callable,
        h_U_grad: Callable,
        r: float,
        Q: Optional[np.ndarray] = None,
        kappa_B: Optional[Callable] = None,
        kappa_U: Optional[Callable] = None,
        alpha: Optional[Callable] = None,
        alpha_prime: Optional[Callable] = None,
    ):
        self.f = f
        self.g = g
        self.l_0 = l_0
        self.h_U = h_U
        self.h_U_grad = h_U_grad
        self.r = r
        self.n = len(l_0)  # state dimension
        
        # Weight matrix (identity by default)
        if Q is None:
            Q = np.eye(self.n)
        self.Q = Q
        
        # Default class-K functions (exponential)
        if kappa_B is None:
            kappa_B = lambda h: 2.0 * h
        self.kappa_B = kappa_B
        
        if kappa_U is None:
            kappa_U = lambda h: 2.0 * h
        self.kappa_U = kappa_U
        
        # Default schedules (set to 1 for no vanishing)
        if alpha is None:
            alpha = lambda t: np.exp(-t)  # decays to 0
        self.alpha = alpha
        
        if alpha_prime is None:
            alpha_prime = lambda t: np.exp(-t)  # decays to 0
        self.alpha_prime = alpha_prime
    
    def h_B(self, x: np.ndarray) -> float:
        """Ball barrier: h_B(x) = r - ||x||_{Q}."""
        norm_x = np.sqrt(float(x @ self.Q @ x))
        return self.r - norm_x
    
    def grad_h_B(self, x: np.ndarray) -> np.ndarray:
        """Gradient of h_B."""
        norm_x = np.sqrt(float(x @ self.Q @ x))
        if norm_x < 1e-10:
            return np.zeros(self.n)
        return -(self.Q @ x) / norm_x
    
    def h_bar_U(self, x: np.ndarray) -> float:
        """Complement barrier: h_bar_U(x) = -h_U(x) (safe region outside U)."""
        return -self.h_U(x)
    
    def grad_h_bar_U(self, x: np.ndarray) -> np.ndarray:
        """Gradient of h_bar_U."""
        return -self.h_U_grad(x)
    
    def L_f_h_B(self, x: np.ndarray) -> float:
        """Lie derivative L_f h_B(x) = ∇h_B · f(x)."""
        return float(self.grad_h_B(x) @ self.f(x))
    
    def L_g_h_B(self, x: np.ndarray, t: float) -> np.ndarray:
        """Lie derivative L_g h_B(x,t) = ∇h_B · g(x,t)."""
        return self.grad_h_B(x) @ self.g(x, t)
    
    def L_f_h_bar_U(self, x: np.ndarray) -> float:
        """Lie derivative L_f h_bar_U(x) = ∇h_bar_U · f(x)."""
        return float(self.grad_h_bar_U(x) @ self.f(x))
    
    def L_g_h_bar_U(self, x: np.ndarray, t: float) -> np.ndarray:
        """Lie derivative L_g h_bar_U(x,t) = ∇h_bar_U · g(x,t)."""
        return self.grad_h_bar_U(x) @ self.g(x, t)
    
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
              L_f h_bar_U + L_g h_bar_U (k/s') + ∇h_bar_U^T l >= -κ_U(h_bar_U)
        
        Returns k* (unscaled control), which is then applied as u = k* / s'(t).
        """
        m = self.g(x, t).shape[1]  # control dimension
        
        # Compute time-varying parameters
        s_t = 1.0 + self.alpha(t)
        s_prime_t = 1.0 + self.alpha_prime(t)
        l_t = self.l_0 / s_t
        
        # Barrier values and derivatives
        h_B = self.h_B(x)
        grad_h_B = self.grad_h_B(x)
        L_f_h_B = self.L_f_h_B(x)
        L_g_h_B = self.L_g_h_B(x, t)
        
        h_bar_U = self.h_bar_U(x)
        grad_h_bar_U = self.grad_h_bar_U(x)
        L_f_h_bar_U = self.L_f_h_bar_U(x)
        L_g_h_bar_U = self.L_g_h_bar_U(x, t)
        
        # Build QP: min 0.5 k^T Q k
        # Cost matrix
        cost_matrix = self.Q
        
        # CBF constraints: A k <= b
        # Constraint 1: L_f h_B + L_g h_B (k/s') + ∇h_B^T l >= -κ_B(h_B)
        #              => -L_g h_B (k/s') <= L_f h_B + ∇h_B^T l + κ_B(h_B)
        #              => L_g h_B (k/s') >= -L_f h_B - ∇h_B^T l - κ_B(h_B)
        #              => L_g h_B k >= -s' (L_f h_B + ∇h_B^T l + κ_B(h_B))
        
        b1_rhs = L_f_h_B + float(grad_h_B @ l_t) + self.kappa_B(h_B)
        A1 = -L_g_h_B / s_prime_t  # coefficient for k, negated for <= form
        b1 = -b1_rhs  # RHS for A1 k <= b1 form
        
        # Constraint 2: L_f h_bar_U + L_g h_bar_U (k/s') + ∇h_bar_U^T l >= -κ_U(h_bar_U)
        b2_rhs = L_f_h_bar_U + float(grad_h_bar_U @ l_t) + self.kappa_U(h_bar_U)
        A2 = -L_g_h_bar_U / s_prime_t
        b2 = -b2_rhs
        
        # Stack constraints
        A = np.vstack([A1, A2])
        b = np.array([b1, b2])
        
        # Solve QP using scipy
        def objective(k):
            return 0.5 * float(k @ cost_matrix @ k)
        
        def objective_grad(k):
            return cost_matrix @ k
        
        # Initial guess: nominal control or zero
        if u_nom is not None:
            k0 = u_nom / (s_prime_t)
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
        
        # Solve
        result = minimize(
            objective,
            k0,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            options={'ftol': eps, 'maxiter': 100},
        )
        
        if not result.success:
            print(f"Warning: QP did not converge at t={t:.4f}. Message: {result.message}")
        
        return result.x
    
    def compute_control(
        self,
        x: np.ndarray,
        t: float,
        u_nom: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the vanishing control u(t).
        
        Returns:
        --------
        u : np.ndarray
            Applied control u(t) = k*(t) / s'(t)
        s_prime_t : float
            The scaling factor s'(t) for reference
        """
        # Solve QP to get k*
        k_star = self.solve_qp(x, t, u_nom=u_nom)
        
        # Apply vanishing feedback
        s_prime_t = 1.0 + self.alpha_prime(t)
        u_t = k_star / s_prime_t
        
        return u_t, s_prime_t
    
    def _dynamics(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Compute state derivative: x_dot = f(x) + g(x,t) u + l(t).
        """
        s_t = 1.0 + self.alpha(t)
        l_t = self.l_0 / s_t
        return self.f(x) + self.g(x, t) @ u + l_t
    
    def step_rk4(
        self,
        x: np.ndarray,
        t: float,
        dt: float,
        u_nom: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 integration step with vanishing CBF-QP safety filter.
        
        Returns:
        --------
        x_next : np.ndarray
            State at t + dt
        u_avg : np.ndarray
            Average control input applied
        """
        # RK4 stages
        # Stage 1
        u1, _ = self.compute_control(x, t, u_nom=u_nom)
        k1 = self._dynamics(x, u1, t)
        
        # Stage 2
        x2 = x + 0.5 * dt * k1
        u2, _ = self.compute_control(x2, t + 0.5 * dt, u_nom=u_nom)
        k2 = self._dynamics(x2, u2, t + 0.5 * dt)
        
        # Stage 3
        x3 = x + 0.5 * dt * k2
        u3, _ = self.compute_control(x3, t + 0.5 * dt, u_nom=u_nom)
        k3 = self._dynamics(x3, u3, t + 0.5 * dt)
        
        # Stage 4
        x4 = x + dt * k3
        u4, _ = self.compute_control(x4, t + dt, u_nom=u_nom)
        k4 = self._dynamics(x4, u4, t + dt)
        
        # Combine stages
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_avg = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
        
        return x_next, u_avg
    
    def step(
        self,
        x: np.ndarray,
        t: float,
        dt: float,
        u_nom: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single integration step with safety filter (uses RK4).
        
        Returns:
        --------
        x_next : np.ndarray
            State at t + dt
        u_applied : np.ndarray
            Control input applied
        """
        return self.step_rk4(x, t, dt, u_nom=u_nom)


def run_simulation(
    filter_obj: VanishingCBFQPFilter,
    x0: np.ndarray,
    t_sim: float,
    dt: float,
    u_nom_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run simulation with the vanishing CBF-QP filter.
    
    Returns:
    --------
    ts : np.ndarray
        Time points
    xs : np.ndarray
        State trajectory
    us : np.ndarray
        Control trajectory
    h_B_vals : np.ndarray
        Ball barrier values
    """
    N = int(t_sim / dt)
    ts = np.linspace(0, t_sim, N + 1)
    
    xs = np.zeros((N + 1, x0.shape[0]))
    us = np.zeros((N, len(filter_obj.l_0)))  # m = dim of control
    h_B_vals = np.zeros(N + 1)
    
    x = x0.copy()
    xs[0] = x
    h_B_vals[0] = filter_obj.h_B(x)
    
    for k in range(N):
        t = ts[k]
        u_nom = u_nom_func(x, t) if u_nom_func else None
        x, u = filter_obj.step(x, t, dt, u_nom=u_nom)
        xs[k + 1] = x
        us[k] = u
        h_B_vals[k + 1] = filter_obj.h_B(xs[k + 1])
    
    return ts, xs, us, h_B_vals
