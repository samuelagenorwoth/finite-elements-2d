"""Compare the 2026 solver with the 2024 hand-built assembly.

Runs the matrix assembly and solve of the original scripts in ../original-2024
(without their slow plotting part) and compares the nodal values.
  * Unit square: the two agree to machine precision.
  * L-shape: the 2024 code keeps the nodes of the removed corner as unknowns
    ("ghost" nodes). They are coupled to their neighbours through the mass
    matrix and pull the solution off by a few percent. Removing them from the
    2024 matrix gives the 2026 result exactly.
Run from this folder: python3 compare_2024.py   (takes about a minute)
"""
from pathlib import Path
import numpy as np
from fem import mesh_square, mesh_lshape, solve

ORIGINAL = Path(__file__).resolve().parent.parent / "original-2024"


def run_2024(script, N):
    """Assemble and solve with the 2024 code; return its matrix and solution."""
    src = (ORIGINAL / script).read_text()
    src = src.split("def Solution0")[0]                  # skip the plotting part
    for old in ("\nN = 4\n", "\nN = 16\n"):
        src = src.replace(old, f"\nN = {N}\n")
    g = {"__name__": "run_2024"}
    exec(src, g)
    return g["A"], g["U_sol"].reshape(N, N)              # values at [x index, y index]


def interior_2026(mesh, N):
    n = N + 1
    nodes, tri = mesh(n)
    G = np.zeros((n + 1, n + 1))
    ij = np.rint(nodes * n).astype(int)
    G[ij[:, 0], ij[:, 1]] = solve(nodes, tri)
    return G[1:-1, 1:-1]


def main():
    print("Unit square (2024 Algorithm 1):")
    for N in [3, 7, 15]:
        _, old = run_2024("manual_square.py", N)
        new = interior_2026(mesh_square, N)
        print(f"  N = {N:2d}: largest difference {abs(new - old).max():.1e}")

    print("L-shaped domain (2024 Algorithm 2):")
    for N in [7, 15]:                                   # N odd: the corner is exactly at 0.5
        A, old = run_2024("manual_lshape.py", N)
        new = interior_2026(mesh_lshape, N)
        i = np.arange(1, N + 1)
        kept = ~((i[:, None] >= (N + 1) / 2) & (i[None, :] >= (N + 1) / 2))
        diff = abs(new[kept] - old[kept]).max()
        k = kept.ravel()
        fixed = np.linalg.solve(A[np.ix_(k, k)], np.full(k.sum(), 1 / (N + 1) ** 2))
        print(f"  N = {N:2d}: largest difference {diff:.1e} "
              f"({100 * diff / new.max():.0f}% of max u); "
              f"after removing the ghost nodes: {abs(fixed - new[kept]).max():.1e}")


if __name__ == "__main__":
    main()
