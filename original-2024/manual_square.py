import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy.linalg as scp
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show

## Transformation to standard traingle
# Function to produce the affine transformation from a given triangle $P_i(xi,yi),~i=1,2,3$,
# the standard triangle $A(0,0), B(1,0), C(0,1)$.
def Btransf(p1, p2, p3):
    A = np.zeros((6, 6))
    Lef = np.array([0, 0, 1, 0, 0, 1])
    A[:, 2] = np.array([1, 0, 1, 0, 1, 0])
    A[:, 5] = np.array([0, 1, 0, 1, 0, 1])
    A[:, 0] = np.array([p1[0], 0, p2[0], 0, p3[0], 0])
    A[:, 1] = np.array([p1[1], 0, p2[1], 0, p3[1], 0])
    A[:, 3] = np.array([0, p1[0], 0, p2[0], 0, p3[0]])
    A[:, 4] = np.array([0, p1[1], 0, p2[1], 0, p3[1]])
    B_standard = scp.solve(A, Lef)
    # The matrix or linear part
    B_tr = np.array([[B_standard[0], B_standard[1]], [B_standard[3], B_standard[4]]])
    # The intercepts
    vect = np.array([B_standard[2], B_standard[5]])
    return B_tr, vect
# Collecting the matrix
def proj1Btransf(p1, p2, p3):
    return Btransf(p1, p2, p3)[0]
# Collecting the vector
def proj2Btransf(p1, p2, p3):
    return Btransf(p1, p2, p3)[1]
# Bijection from ${1,2,...,N^2}$ to ${(i,j):1\leq i,j\leq N}$
# This function helps to locate the hat function for the grid points
# so that one can know the nodes to use and the elements to consider.
def i_k(k, N):
    ik = min([i for i in range(1, N+1) if k <= i*N])
    jk = k - (ik-1)*N
    ik = ik - 1
    jk = jk - 1
    return np.array([ik, jk])
## The standard hat function
def standHat(x, y, i):
    if i == 1:
        z = 1 - x - y
    elif i == 2:
        z = x
    else:
        z = y
    return z
## The hat function built upon nodes of the grid points
def HatFunck(x, y, k, N):
    # x,y are point coordinates
    # k is the hat function's number
    # the numbers of internal nodes considered
    h = 1./(N+1)  # To get N internal nodes
    ind1 = i_k(k, N)
    ik = ind1[0]+1
    jk = ind1[1]+1
    # $(i_k,i_k)$ represent the position of $\vfiz_k$ in the grid points
    ik0, jk0 = ik, jk-1
    ik2, jk2 = ik, jk+1
    ik3, jk3 = ik-1, jk
    ik4, jk4 = ik+1, jk
    if x == ik*h and y == jk*h:
        z = 1
        grad = 0
    elif x > ik3*h and x <= ik*h:
        if y > -x+(ik0+jk0)*h and y <= jk*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik0*h, jk0*h])
            P3 = np.array([ik3*h, jk3*h])
            transf = Btransf(P2, P3, P1)
            transfM = proj1Btransf(P2, P3, P1)
            transfV = proj2Btransf(P2, P3, P1)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 3)
            grad = transfM[1]
        elif y <= -x+(ik+jk)*h and y > jk*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik3*h, jk3*h])
            P3 = np.array([ik3*h, jk2*h])
            transfM = proj1Btransf(P2, P1, P3)
            transfV = proj2Btransf(P2, P1, P3)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 2)
            grad = transfM[0]
        elif y > -x+(ik+jk)*h and y < jk2*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik*h, jk2*h])
            P3 = np.array([ik3*h, jk2*h])
            transfM = proj1Btransf(P1, P3, P2)
            transfV = proj2Btransf(P1, P3, P2)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 1)
            grad = -transfM[0] - transfM[1]
        else:
            z = 0
            grad = np.zeros(2)
    elif x > ik*h and x < ik4*h:
        if y < -x+(ik4+jk4)*h and y >= jk*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik2*h, jk2*h])
            P3 = np.array([ik4*h, jk4*h])
            transf = Btransf(P2, P3, P1)
            transfM = proj1Btransf(P1, P3, P2)
            transfV = proj2Btransf(P1, P3, P2)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 1)
            grad = -transfM[1] - transfM[0]
        elif y >= -x+(ik4+jk0)*h and y < jk*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik4*h, jk0*h])
            P3 = np.array([ik4*h, jk4*h])
            transfM = proj1Btransf(P2, P1, P3)
            transfV = proj2Btransf(P2, P1, P3)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 2)
            grad = transfM[0]
        elif y < -x+(ik4+jk0)*h and y > jk0*h:
            P1 = np.array([ik*h, jk*h])
            P2 = np.array([ik4*h, jk0*h])
            P3 = np.array([ik0*h, jk0*h])
            transfM = proj1Btransf(P3, P2, P1)
            transfV = proj2Btransf(P3, P2, P1)
            vec = np.array([x, y])
            Xtra = np.dot(transfM, vec) + transfV
            z = standHat(Xtra[0], Xtra[1], 3)
            grad = transfM[1]
        else:
            z = 0
            grad = np.zeros(2)
    else:
        z = 0
        grad = np.array([0, 0])
    return z, grad  # we also return the gradient of the $\vfi_k$
def proj1Hat(x, y, k, N):  # To get the hat function
    return HatFunck(x, y, k, N)[0]
def proj2Hat(x, y, k, N):  # To get the gradiant of the hat function
    return HatFunck(x, y, k, N)[1]
## Compute the norm squared
def norm_2(a):
    return np.linalg.norm(a)**2
## Computation of the scalar product of the gradient of hat function
# $\nabla\vfi_k\cdot\nabla\vfi_l$ to used in $\int_{\Om}\nabla\vfi_k\cdot\nabla\vfi_l$
def gradcomp(k, l, N):
    h = 1./(N+1)
    ind = i_k(k, N)
    ik = ind[0]+1
    jk = ind[1]+1
    # building points for computing successfully the gradient
    if l != k:
        if l == k-1:
            x1, y1 = ik*h+(h/10), (jk-1)*h+(h/10)
            x2, y2 = ik*h-(h/10), jk*h-(h/10)
        elif l == k-N:
            x1, y1 = ik*h-(h/10), jk*h-(h/10)
            x2, y2 = (ik-1)*h+(h/10), jk*h+(h/10)
        elif l == k-N+1:
            x1, y1 = (ik-1)*h+(h/10), jk*h+(h/10)
            x2, y2 = ik*h-(h/10), (jk+1)*h-(h/10)
        elif l == k+1:
            x1, y1 = ik*h-(h/10), (jk+1)*h-(h/10)
            x2, y2 = ik*h+(h/10), jk*h+(h/10)
        elif l == k+N:
            x1, y1 = ik*h+(h/10), jk*h+(h/10)
            x2, y2 = (ik+1)*h-(h/10), jk*h-(h/10)
        elif l == k+N-1:
            x1, y1 = (ik+1)*h-(h/10), jk*h-(h/10)
            x2, y2 = ik*h+(h/10), (jk-1)*h+(h/10)
        else:
            x1, y1 = (ik+2)*h, (jk+2)*h
            x2, y2 = (ik-2)*h, (jk-2)*h
        s1 = np.dot(proj2Hat(x1, y1, k, N), proj2Hat(x1, y1, l, N))
        s2 = np.dot(proj2Hat(x2, y2, k, N), proj2Hat(x2, y2, l, N))
        s3 = 0
        s4 = 0
        s5 = 0
        s6 = 0
    else:
        x1, y1 = ik*h+(h/10), (jk-1)*h+(h/10)
        x2, y2 = ik*h-(h/10), jk*h-(h/10)
        x3, y3 = (ik-1)*h+(h/10), jk*h+(h/10)
        x4, y4 = ik*h-(h/10), (jk+1)*h-(h/10)
        x5, y5 = ik*h+(h/10), jk*h+(h/10)
        x6, y6 = (ik+1)*h-(h/10), jk*h-(h/10)
        s1 = norm_2(proj2Hat(x1, y1, k, N))
        s2 = norm_2(proj2Hat(x2, y2, k, N))
        s3 = norm_2(proj2Hat(x3, y3, k, N))
        s4 = norm_2(proj2Hat(x4, y4, k, N))
        s5 = norm_2(proj2Hat(x5, y5, k, N))
        s6 = norm_2(proj2Hat(x6, y6, k, N))
    sum1 = s1+s2+s3+s4+s5+s6
    sum1 = s1+s2+s3+s4+s5+s6
    return sum1
## A set of multiples of $N$ and integer that can be written as $iN+1$
def Mult(N):
    mult1 = np.array([N+1])
    mult2 = np.array([N])
    for i in range(2, N):
        mult1 = np.concatenate((mult1, np.array([i*N+1])), axis=0)
        mult2 = np.concatenate((mult2, np.array([i*N])), axis=0)
    return mult1, mult2
def proj1Mult(N):
    return Mult(N)[0]
def proj2Mult(N):
    return Mult(N)[1]
## Evaluating the component k of the solution
def LocalSol(coef, f, x, y, k, N):
    z = coef[k-1]*f(x, y, k, N)
    return z
# The code initializes a numerical grid and defines a Position function
# compute indices for a sparse matrix, crucial in solving grid-based problems
## The indices varies with grid size and point location, indicating a complex grid
# structure or boundary conditions
x0 = 0
xfin = 1
N = 4
h0 = 1./(N+1)
dim0 = N**2
if N == 1:
    dim = 1
elif N == 2:
    dim = 14
else:
    dim = 6+20*(N-2)+8+7*((N-2)**2)
I = np.zeros(dim)
J = np.zeros(dim)
S = np.zeros(dim)
B = np.zeros(dim0)
def Position(k, N):
    if N == 1:
        positions = np.array([1])
    elif N == 2:
        if k == 1:
            positions = np.array([1, 2, 3])
        elif k == 2:
            positions = np.array([4, 5, 6, 7])
        elif k == 3:
            positions = np.array([8, 9, 10, 11])
        else:
            positions = np.array([12, 13, 14])
    else:
        if k == 1:
            positions = np.array([1, 2, 3])
            return positions
        elif k == 2:
            positions = np.array([4, 5, 6, 7, 8])
        elif (k > 2) and (k <= N-1):
            ik = (k-2)*5+4
            positions = np.array([ik])
            for i in range(1, 5):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif k == N:
            ik = Position(k-1, N)[4]+1
            positions = np.array([ik])
            for i in range(1, 4):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif (k in proj1Mult(N)):
            check = int((k-1)/N)
            if check == 1:
                ik = Position(k-1, N)[3]+1
            else:
                ik = Position(k-1, N)[4]+1
            positions = np.array([ik])
            if check == N-1:
                for i in range(1, 4):
                    positions = np.concatenate((positions, np.array([ik+i])), axis=0)
            else:
                for i in range(1, 5):
                    positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif (k in proj2Mult(N)) and (k > N) and k < N**2:
            ik = Position(k-1, N)[6]+1
            positions = np.array([ik])
            for i in range(1, 5):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif k == N**2:
            ik = Position(k-1, N)[4]+1
            positions = np.array([ik])
            for i in range(1, 3):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif k > (N**2-N+1) and k < N**2:
            if k == N**2-N+2:
                ik = Position(k-1, N)[3]+1
            else:
                ik = Position(k-1, N)[4]+1
            positions = np.array([ik])
            for i in range(1, 5):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        elif k-1 in proj1Mult(N):
            ik = Position(k-1, N)[4]+1
            positions = np.array([ik])
            for i in range(1, 7):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
        else:
            ik = Position(k-1, N)[6]+1
            positions = np.array([ik])
            for i in range(1, 7):
                positions = np.concatenate((positions, np.array([ik+i])), axis=0)
    return positions

for k in range(1, N**2+1):
    pos = Position(k, N)
    for i in pos:
        I[i-1] = k-1
if N == 1:
    J[0] = 0
elif N >= 2:
    J[0], J[1], J[2] = 0, 1, N
    J[dim-3], J[dim-2], J[dim-1] = dim0-N-1, dim0-2, dim0-1
    j = dim0-N+1
    pos = Position(j, N)[0]
    pos = int(pos)
    J[pos-1], J[pos], J[pos+1], J[pos+2] = j-1-N, j-N, j-1, j
    pos2 = Position(N, N)[0]
    pos2 = int(pos2)
    J[pos2-1], J[pos2], J[pos2+1], J[pos2+2] = N-2, N-1, 2*N-2, 2*N-1
    if N == 2:
        pass
    else:
        for k in range(1, dim0+1):
            pos3 = Position(k, N)
            if len(pos3) != 3 or len(pos3) != 4:
                pos30 = pos3[0]
                pos30 = int(pos30)
                if len(pos3) == 5:
                    if k > 1 and k < N:
                        J[pos30-1], J[pos30], J[pos30+1], J[pos30+2], J[pos30+3] = k-2, k-1, k, k+N-2, k+N-1
                    if k > dim0-N+1 and k < dim0:
                        J[pos30-1], J[pos30], J[pos30+1], J[pos30+2], J[pos30+3] = k-N-1, k-N, k-2, k-1, k
                    if k in proj2Mult(N) and k > N and k < dim0:
                        J[pos30-1], J[pos30], J[pos30+1], J[pos30+2], J[pos30+3] = k-N-1, k-2, k-1, k+N-2, k+N-1
                    if k in proj1Mult(N) and k < j:
                        J[pos30-1], J[pos30], J[pos30+1], J[pos30+2], J[pos30+3] = k-N-1, k-N, k-1, k, k+N-1
                if len(pos3) == 7:
                    J[pos30-1], J[pos30], J[pos30+1], J[pos30+2], J[pos30+3] = k-N-1, k-N, k-2, k-1, k
                    J[pos30+4], J[pos30+5] = k+N-2, k+N-1
for i in range(dim):
    i1 = int(I[i])+1
    i2 = int(J[i])+1
    if i1 == i2:
        S[i] = ((h0**2)/2)+((h0**2)/2)*gradcomp(i1, i2, N)
    else:
        S[i] = ((h0**2)/12)+((h0**2)/2)*gradcomp(i1, i2, N)
A = csr_matrix((S, (I, J)), shape=(dim0, dim0)).toarray()
for i in range(dim0):
    B[i] = h0**2
U_sol = scp.solve(A, B)
def Solution0(x, y):
    Summ = 0
    for k in range(1, dim0+1):
        Summ = Summ + LocalSol(U_sol, proj1Hat, x, y, k, N)
    return Summ
Sol = np.vectorize(HatFunck)
Solution = np.vectorize(Solution0)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X0, Y0 = np.meshgrid(x, y)
Z0 = Solution(X0, Y0)
im = imshow(Z0, cmap=cm.RdBu, extent=[0, 1, 0, 1], origin='lower')  # drawing the function
colorbar(im)  # adding the colobar on the rigth
# latex fashion title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
title('Solution of $u-\\Delta u=1$ for $N=4$')
plt.savefig('SolutionPlotSquare1.png')
plt.show()
