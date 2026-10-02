# This is the code for the unit square domain

from fenics import *
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1)
a = (u*v + dot(grad(u), grad(v)))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
import matplotlib.pyplot as plt
plot(u)

plt.savefig("unit square_solution")

plt.show()

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Print errors
print('error_L2 =', error_L2)

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print maximum error
print('error_max =', error_max)
