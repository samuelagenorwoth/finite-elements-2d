
# This is a code for the error vs mesh size on a unit square domain

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define boundary condition
u_D = Expression('0', degree=1)

def boundary(x, on_boundary):
    return on_boundary

# Define variational problem
def solve_pde(N):
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'P', 1)
    bc = DirichletBC(V, u_D, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)
    a = (u*v + dot(grad(u), grad(v)))*dx
    L = f*v*dx
    u = Function(V)
    solve(a == L, u, bc)
    return u, mesh

# Compute error in L2 norm and record mesh size
mesh_sizes = [8, 16, 32, 64, 128]
errors_L2 = []
h = []

for N in mesh_sizes:
    u, mesh = solve_pde(N)
    u_exact = interpolate(u_D, V)
    error_L2 = errornorm(u_exact, u, 'L2')
    errors_L2.append(error_L2)
    h.append(mesh.hmax())

# Plotting the error vs. mesh size
plt.loglog(h, errors_L2, '-o')
plt.xlabel('Mesh size (h)')
plt.ylabel('L2 error')
plt.title('L2 error vs. Mesh size')
plt.grid(True)

plt.savefig("error_solution")

plt.show()
