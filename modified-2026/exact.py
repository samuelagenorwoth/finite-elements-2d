"""Exact solution of  u - Laplace(u) = 1  on the unit square with u = 0 on the boundary.

Expanding in the eigenfunctions sin(m pi x) sin(n pi y) of the Laplacian gives

    u(x, y) = sum over odd m, n of
              16 / (pi^2 m n (1 + pi^2 (m^2 + n^2))) * sin(m pi x) sin(n pi y).

The coefficients fall off like 1/(m n (m^2 + n^2)), so a few hundred terms in
each direction are enough for errors far below those of the finite elements.
"""
import numpy as np


def exact_square(x, y, terms=400):
    """Evaluate the series on the grid x (length P) by y (length Q). Returns P x Q."""
    k = np.arange(1, 2 * terms, 2)                       # odd mode numbers
    c = 16 / (np.pi**2 * np.outer(k, k) * (1 + np.pi**2 * (k[:, None]**2 + k[None, :]**2)))
    Sx = np.sin(np.pi * np.outer(x, k))                  # P x terms
    Sy = np.sin(np.pi * np.outer(y, k))                  # Q x terms
    return Sx @ c @ Sy.T
