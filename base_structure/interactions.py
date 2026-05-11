"""
Interaction matrix generation with correlated reciprocal interactions.

Implements the parameterization from Bunin (2017), Eq. (2):
    alpha_ij = mu/S + sigma * a_ij

where a_ij has mean 0, variance 1/S, and corr(a_ij, a_ji) = gamma.
"""

import numpy as np


def generate_interaction_matrix(S, mu, sigma, gamma, rng):
    """Generate the interaction matrix alpha with correlated pairs.

    For each pair (i,j) with i<j, draws (z1, z2) from a bivariate
    normal with correlation gamma, then sets a_ij = z1/sqrt(S),
    a_ji = z2/sqrt(S). The diagonal is zero.

    Parameters
    ----------
    S : int
        Species pool size.
    mu : float
        Scaled mean interaction (= S * mean(alpha_ij)).
    sigma : float
        Scaled interaction heterogeneity (= sqrt(S * var(alpha_ij))).
    gamma : float
        Correlation between alpha_ij and alpha_ji, in [-1, 1].
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    alpha : ndarray of shape (S, S)
        Interaction matrix with zero diagonal.
    """
    n_pairs = S * (S - 1) // 2

    x1 = rng.standard_normal(n_pairs)
    x2 = rng.standard_normal(n_pairs)

    # Bivariate normal with correlation gamma
    z1 = x1
    z2 = gamma * x1 + np.sqrt(max(1.0 - gamma**2, 0.0)) * x2

    # Build the fluctuation matrix a_ij with var = 1/S
    a = np.zeros((S, S))
    iu, ju = np.triu_indices(S, k=1)
    a[iu, ju] = z1 / np.sqrt(S)
    a[ju, iu] = z2 / np.sqrt(S)

    # Full interaction matrix
    alpha = mu / S + sigma * a
    np.fill_diagonal(alpha, 0.0)

    return alpha
