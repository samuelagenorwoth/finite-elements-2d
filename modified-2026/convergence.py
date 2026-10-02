"""Convergence study: how fast does the finite element error shrink as h -> 0?

Square:   the error is measured against the exact series solution.
L-shape:  there is no formula for the exact solution, so the error is measured
          against a solution on a much finer mesh (n = 1024).

Both errors are computed on the fine mesh: the coarse solution is evaluated at
the fine nodes (exactly, since P1 functions are linear on each triangle), and
the difference e gives
    L2 error    = sqrt(e^T M e)     (size of the error)
    H1 error    = sqrt(e^T K e)     (size of the error in the gradient)
with M and K the fine-mesh mass and stiffness matrices.
"""
from pathlib import Path
import numpy as np
from fem import mesh_square, mesh_lshape, assemble, solve
from exact import exact_square

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_COARSE = [4, 8, 16, 32, 64, 128]
N_REF = {"square": 512, "lshape": 1024}


def to_grid(nodes, u, n):
    """Put nodal values on the full (n+1) x (n+1) grid (zero where there is no node)."""
    G = np.zeros((n + 1, n + 1))
    ij = np.rint(nodes * n).astype(int)
    G[ij[:, 0], ij[:, 1]] = u
    return G


def evaluate(G, n, pts):
    """Evaluate the P1 function with grid values G (mesh_square(n) layout) at points."""
    s = np.clip(pts[:, 0] * n, 0, n - 1e-12)
    t = np.clip(pts[:, 1] * n, 0, n - 1e-12)
    i, j = np.floor(s).astype(int), np.floor(t).astype(int)
    s, t = s - i, t - j                                   # local coordinates in the cell
    sw, se, nw, ne = G[i, j], G[i + 1, j], G[i, j + 1], G[i + 1, j + 1]
    lower = s + t <= 1                                    # triangle (sw, se, nw)
    return np.where(lower,
                    sw * (1 - s - t) + se * s + nw * t,
                    se * (1 - t) + ne * (s + t - 1) + nw * (1 - s))


def study(domain):
    mesh = mesh_square if domain == "square" else mesh_lshape
    n_ref = N_REF[domain]
    fine_nodes, fine_tri = mesh(n_ref)
    M, K, _ = assemble(fine_nodes, fine_tri)
    if domain == "square":
        x = np.linspace(0, 1, n_ref + 1)
        G = exact_square(x, x)
        u_ref = G[np.rint(fine_nodes[:, 0] * n_ref).astype(int),
                  np.rint(fine_nodes[:, 1] * n_ref).astype(int)]
    else:
        u_ref = solve(fine_nodes, fine_tri)

    h, eL2, eH1 = [], [], []
    for n in N_COARSE:
        nodes, tri = mesh(n)
        u = solve(nodes, tri)
        e = u_ref - evaluate(to_grid(nodes, u, n), n, fine_nodes)
        h.append(1 / n)
        eL2.append(np.sqrt(e @ (M @ e)))
        eH1.append(np.sqrt(e @ (K @ e)))
    return np.array(h), np.array(eL2), np.array(eH1)


def rates(e):
    """Convergence rate for each halving of h: log2(e(h) / e(h/2))."""
    return np.log2(e[:-1] / e[1:])


def main():
    RESULTS.mkdir(exist_ok=True)
    data = {}
    for domain in ["square", "lshape"]:
        h, eL2, eH1 = study(domain)
        data.update({f"{domain}_h": h, f"{domain}_L2": eL2, f"{domain}_H1": eH1})
        print(f"\n{domain}:")
        rL2, rH1 = rates(eL2), rates(eH1)
        print("     n     L2 error  rate     H1 error  rate")
        for k, (hh, a, b) in enumerate(zip(h, eL2, eH1)):
            ra = f"{rL2[k-1]:5.2f}" if k else "     "
            rb = f"{rH1[k-1]:5.2f}" if k else "     "
            print(f"  {round(1 / hh):4d}   {a:.3e} {ra}    {b:.3e} {rb}")
    np.savez(RESULTS / "convergence.npz", **data)


if __name__ == "__main__":
    main()
