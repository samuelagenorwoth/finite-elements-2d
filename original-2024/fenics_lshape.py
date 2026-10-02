# FEniCS code for the L-shaped domain

from fenics import *
import matplotlib.pyplot as plt

# Creating a mesh for the unit square
mesh = UnitSquareMesh(32, 32)

# Defining a subdomain for the L-shape
class LShape(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5 or x[1] <= 0.5

# Marking all cells, 1 for those inside the L-shape, 0 otherwise
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)
lshape = LShape()
lshape.mark(domains, 1)

# Creating a new mesh only with the cells marked as 1
lshape_mesh = SubMesh(mesh, domains, 1)

# Defining function space on the L-shaped mesh
V = FunctionSpace(lshape_mesh, 'P', 1)

# Defining boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0), boundary)

# Defining variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1)
a = (u*v + dot(grad(u), grad(v)))*dx
L = f*v*dx

# Computing solution
u = Function(V)
solve(a == L, u, bc)

# Saving solution to file (optional)
vtkfile = File('lshape_solution.pvd')
vtkfile << u

# Plotting the solution
plot(u)
plt.title('Solution on L-shaped Domain')

plt.savefig('lshape_solution.png')
plt.show()
