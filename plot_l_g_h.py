import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Symbolic definitions
# ------------------------------------------------------------------
x1, x2 = sp.symbols('x1 x2', real=True)

# Define G = [1, 1]^T
G = sp.Matrix([1, 1])

# Define h_U(x1, x2)
h_U = (
    4*(x1/1.5)**4
    - 20*(x1/1.5)**2*(x2 - 15)
    - 13*(x1/1.5)**2
    + 25*(x2 - 15)**2
    + 35*(x2 - 15)
    - 2
)

# Compute gradient of h_U
grad_h_U = sp.Matrix([
    sp.diff(h_U, x1),
    sp.diff(h_U, x2)
])

# Define phi(x1,x2) = grad(h_U)^T G
phi = sp.expand((grad_h_U.T * G)[0])

# Print symbolic expressions
print("h_U(x1,x2) =")
sp.pprint(h_U)
print("\nGradient of h_U:")
sp.pprint(grad_h_U)
print("\nphi(x1,x2) = grad(h_U)^T G =")
sp.pprint(phi)

# ------------------------------------------------------------------
# Numerical function for plotting
# ------------------------------------------------------------------
phi_fun = sp.lambdify((x1, x2), phi, 'numpy')

# Grid for plotting
x1_vals = np.linspace(-40, 40, 1200)
x2_vals = np.linspace(-40, 40, 1200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

Z = phi_fun(X1, X2)

# ------------------------------------------------------------------
# Plot zero level set phi(x1,x2) = 0
# ------------------------------------------------------------------
plt.figure(figsize=(8, 8))
cs = plt.contour(X1, X2, Z, levels=[0], linewidths=2)
plt.clabel(cs, fmt={0: r'$\nabla h_U^\top G = 0$'}, inline=True, fontsize=10)

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlim([-40, 40])
plt.ylim([-40, 40])

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Zero level set of $\nabla h_U(x_1,x_2)^\top G$ with $G=[2,0]^\top$')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()