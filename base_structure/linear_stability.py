"""
Jacobian, community matrix, and eigendecomposition at a fixed point
of the generalized Lotka-Volterra equations.
"""

import numpy as np
from collections import namedtuple

EigenResult = namedtuple("EigenResult", [
    "eigenvalues", "eigenvectors", "max_real_part"
])


def jacobian(r, K, alpha, N):
    """Compute the Jacobian of the LV system at abundances N.

    For dN_i/dt = (r_i/K_i) N_i (K_i - N_i - sum_j alpha_ij N_j):

        J_ii = (r_i/K_i) (K_i - 2 N_i - (alpha @ N)_i)
        J_ij = -(r_i/K_i) N_i alpha_ij   (j != i)

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
        Interaction matrix (zero diagonal).
    N : ndarray (S,)
        Abundances at which to evaluate.

    Returns
    -------
    J : ndarray (S, S)
    """
    rK = r / K
    interaction = alpha @ N

    # Off-diagonal: J_ij = -(r_i/K_i) * N_i * alpha_ij
    J = -(rK * N)[:, np.newaxis] * alpha

    # Diagonal: J_ii = (r_i/K_i) * (K_i - 2*N_i - (alpha @ N)_i)
    np.fill_diagonal(J, rK * (K - 2.0 * N - interaction))

    return J


def community_matrix(r, K, alpha, fp):
    """Compute the community matrix M* for surviving species.

    M is defined so that J_reduced = -M at the fixed point.
    For surviving species (where K_i - N_i - alpha @ N = 0):

        M_ij = (r_i N_i / K_i) * alpha_tilde_ij

    where alpha_tilde_ij = alpha_ij for j != i, and alpha_tilde_ii = 1
    (accounting for self-regulation).

    M positive definite <=> fixed point is stable.

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
    fp : FixedPoint

    Returns
    -------
    M_star : ndarray (S*, S*) where S* = number of surviving species
    """
    s = fp.surviving
    r_s, K_s, N_s = r[s], K[s], fp.N_star[s]
    alpha_s = alpha[np.ix_(s, s)].copy()

    # Include self-regulation on diagonal
    alpha_tilde = alpha_s.copy()
    np.fill_diagonal(alpha_tilde, 1.0)

    # M_ij = (r_i * N_i / K_i) * alpha_tilde_ij
    M = (r_s * N_s / K_s)[:, np.newaxis] * alpha_tilde

    return M


def eigendecomposition(r, K, alpha, fp):
    """Eigenvalues and eigenvectors of the Jacobian at the fixed point.

    Restricted to surviving species. Uses np.linalg.eig since the
    Jacobian is generally non-symmetric (gamma != 1).

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
    fp : FixedPoint

    Returns
    -------
    EigenResult
        .eigenvalues : ndarray (S*,) complex
        .eigenvectors : ndarray (S*, S*) columns are eigenvectors
        .max_real_part : float
    """
    s = fp.surviving
    J_full = jacobian(r, K, alpha, fp.N_star)
    J_reduced = J_full[np.ix_(s, s)]

    eigenvalues, eigenvectors = np.linalg.eig(J_reduced)

    # Sort by descending real part
    order = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        max_real_part=eigenvalues[0].real
    )
