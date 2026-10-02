"""Linear (P1) finite elements for  u - Laplace(u) = f  with u = 0 on the boundary.

The method, step by step:
  1. Mesh:      split the domain into triangles (mesh_square, mesh_lshape).
  2. Assembly:  loop over the triangles, compute the small 3x3 element matrices
                and add them into the large sparse matrices M (mass) and K
                (stiffness). This is the standard approach in every FEM code.
  3. Solve:     (M + K) u = b for the values of u at the interior nodes;
                the boundary nodes are fixed at u = 0.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------------
# 1. Meshes
# ----------------------------------------------------------------------------
def mesh_square(n):
    """Unit square, n x n cells, each cut into two triangles.

    The cut runs from the top-left to the bottom-right corner of each cell,
    like the 2024 report, so every interior node touches six triangles.
    Returns nodes (P x 2 array of coordinates) and triangles (T x 3 array of
    node numbers, counter-clockwise).
    """
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)   # idx[i, j] = node at (x_i, y_j)
    sw, se = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    nw, ne = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    lower = np.column_stack([sw, se, nw])                  # below the cut
    upper = np.column_stack([se, ne, nw])                  # above the cut
    return nodes, np.vstack([lower, upper])


def mesh_lshape(n):
    """L-shaped domain: the unit square without the top-right quarter [0.5, 1]^2.

    n must be even, so that x = 0.5 and y = 0.5 are grid lines and the
    re-entrant corner (0.5, 0.5) is a node of the mesh.
    """
    if n % 2:
        raise ValueError("n must be even for the L-shaped domain")
    nodes, tri = mesh_square(n)
    centre = nodes[tri].mean(axis=1)
    keep = ~((centre[:, 0] > 0.5) & (centre[:, 1] > 0.5))
    tri = tri[keep]
    used = np.unique(tri)                                  # drop nodes no triangle uses
    renumber = -np.ones(len(nodes), dtype=int)
    renumber[used] = np.arange(len(used))
    return nodes[used], renumber[tri]


def boundary_nodes(nodes, tri):
    """Nodes on the boundary: the ends of edges that belong to only one triangle."""
    edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[count == 1])


# ----------------------------------------------------------------------------
# 2. Assembly
# ----------------------------------------------------------------------------
def assemble(nodes, tri):
    """Mass matrix M, stiffness matrix K and load vector b (for f = 1).

    For one triangle with area |T| and the three P1 basis functions:
      mass:      int phi_i phi_j      = |T|/12 * (2 if i == j else 1)
      stiffness: int grad phi_i . grad phi_j = |T| * g_i . g_j
                 (the gradients g_i are constant on the triangle)
      load:      int phi_i            = |T|/3
    """
    p = nodes[tri]                                   # T x 3 x 2 corner coordinates
    d1 = p[:, 1] - p[:, 0]
    d2 = p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

    # Gradients of the three basis functions on each triangle:
    # g_i = (y_j - y_k, x_k - x_j) / (2|T|) for (i, j, k) in cyclic order.
    x, y = p[:, :, 0], p[:, :, 1]
    g = np.empty((len(tri), 3, 2))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        g[:, i, 0] = (y[:, j] - y[:, k]) / (2 * area)
        g[:, i, 1] = (x[:, k] - x[:, j]) / (2 * area)

    Ke = area[:, None, None] * np.einsum("tid,tjd->tij", g, g)
    Me = area[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))

    rows = np.repeat(tri, 3, axis=1).ravel()         # (i, j) index pairs for the 3x3 blocks
    cols = np.tile(tri, (1, 3)).ravel()
    P = len(nodes)
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(P, P))   # duplicates are added
    M = sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(P, P))
    b = np.bincount(tri.ravel(), weights=np.repeat(area / 3, 3), minlength=P)
    return M, K, b


# ----------------------------------------------------------------------------
# 3. Solve
# ----------------------------------------------------------------------------
def solve(nodes, tri):
    """Solve u - Laplace(u) = 1, u = 0 on the boundary. Returns u at all nodes."""
    M, K, b = assemble(nodes, tri)
    A = (M + K).tocsr()
    free = np.setdiff1d(np.arange(len(nodes)), boundary_nodes(nodes, tri))
    u = np.zeros(len(nodes))
    u[free] = spsolve(A[free][:, free].tocsc(), b[free])
    return u
