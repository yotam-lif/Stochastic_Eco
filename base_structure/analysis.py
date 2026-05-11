"""
PCA of stochastic trajectories and Linear Noise Approximation (LNA).

The LNA predicts the covariance structure of fluctuations around the
fixed point by solving the continuous Lyapunov equation:
    J C + C J^T + D = 0
where J is the Jacobian and D is the diffusion matrix.
"""

import numpy as np
from collections import namedtuple
from scipy.linalg import solve_continuous_lyapunov

PCAResult = namedtuple("PCAResult", [
    "eigenvalues", "eigenvectors", "explained_ratio", "projections", "mean"
])

LNAResult = namedtuple("LNAResult", [
    "covariance", "eigenvalues", "eigenvectors", "explained_ratio"
])


def pca_trajectory(N_trajectory, surviving=None, n_pcs=None):
    """PCA of fluctuations in a stochastic trajectory.

    Computes delta_N(t) = N(t) - <N>_t, then eigendecomposes the
    covariance matrix.

    Parameters
    ----------
    N_trajectory : ndarray (S, n_times)
        Trajectory from SDE integration.
    surviving : ndarray of int or None
        If given, restrict to these species indices.
    n_pcs : int or None
        Number of principal components to return projections for.

    Returns
    -------
    PCAResult
        .eigenvalues : ndarray (S*,) sorted descending
        .eigenvectors : ndarray (S*, S*) columns are PCs
        .explained_ratio : ndarray (S*,) fraction of variance per PC
        .projections : ndarray (n_pcs, n_times) trajectory in PC space
        .mean : ndarray (S*,) temporal mean abundances
    """
    if surviving is not None:
        N = N_trajectory[surviving, :]
    else:
        N = N_trajectory

    S_eff, n_times = N.shape

    # Temporal mean and fluctuations
    mean_N = N.mean(axis=1)
    delta_N = N - mean_N[:, np.newaxis]

    # Covariance matrix
    cov = (delta_N @ delta_N.T) / (n_times - 1)

    # Eigendecomposition (symmetric positive semi-definite)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Explained variance ratio
    total_var = eigenvalues.sum()
    if total_var > 0:
        explained_ratio = eigenvalues / total_var
    else:
        explained_ratio = np.zeros_like(eigenvalues)

    # Project trajectory onto PCs
    if n_pcs is None:
        n_pcs = S_eff
    n_pcs = min(n_pcs, S_eff)
    projections = eigenvectors[:, :n_pcs].T @ delta_N

    return PCAResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_ratio=explained_ratio,
        projections=projections,
        mean=mean_N
    )


def build_diffusion_matrix(N_star, D, noise_type="demographic"):
    """Construct the diffusion matrix for the LNA.

    Parameters
    ----------
    N_star : ndarray (S*,)
        Fixed point abundances (surviving species only).
    D : float
        Noise amplitude.
    noise_type : str
        'demographic': D_ii = 2 D N*_i
        'additive': D_ii = D^2

    Returns
    -------
    D_matrix : ndarray (S*, S*) diagonal matrix
    """
    S = len(N_star)
    if noise_type == "demographic":
        return np.diag(2.0 * D * N_star)
    elif noise_type == "additive":
        return np.diag(np.full(S, D**2))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")


def linear_noise_approximation(J, D_matrix):
    """Solve the Lyapunov equation for the LNA covariance.

    The steady-state covariance of fluctuations around the fixed point
    satisfies:
        J C + C J^T + D = 0

    where J must have all eigenvalues with negative real part.

    Parameters
    ----------
    J : ndarray (S*, S*)
        Jacobian at the fixed point (restricted to surviving species).
    D_matrix : ndarray (S*, S*)
        Diffusion matrix (typically diagonal).

    Returns
    -------
    LNAResult
        .covariance : ndarray (S*, S*)
        .eigenvalues : ndarray (S*,) of the covariance, sorted descending
        .eigenvectors : ndarray (S*, S*)
        .explained_ratio : ndarray (S*,)
    """
    # Check stability
    eig_J = np.linalg.eigvals(J)
    max_real = np.max(eig_J.real)
    if max_real >= 0:
        import warnings
        warnings.warn(
            f"Jacobian has eigenvalue with Re(lambda)={max_real:.4e} >= 0. "
            "LNA requires a stable fixed point."
        )

    # solve_continuous_lyapunov solves A X + X A^H = Q
    # We need J C + C J^T = -D, so A=J, Q=-D
    C = solve_continuous_lyapunov(J, -D_matrix)

    # Symmetrize (numerical)
    C = 0.5 * (C + C.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_var = eigenvalues.sum()
    if total_var > 0:
        explained_ratio = eigenvalues / total_var
    else:
        explained_ratio = np.zeros_like(eigenvalues)

    return LNAResult(
        covariance=C,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_ratio=explained_ratio
    )


def project_trajectory(N_trajectory, eigenvectors, mean_N, pc_indices=None):
    """Project a trajectory onto specific principal components.

    Parameters
    ----------
    N_trajectory : ndarray (S*, n_times)
        Trajectory (should be restricted to surviving species).
    eigenvectors : ndarray (S*, S*)
        Columns are PCs (from PCA or LNA).
    mean_N : ndarray (S*,)
        Mean abundances to subtract.
    pc_indices : list of int or None
        Which PCs to project onto. Default: [0, 1].

    Returns
    -------
    projections : ndarray (n_pcs, n_times)
    """
    if pc_indices is None:
        pc_indices = [0, 1]

    delta_N = N_trajectory - mean_N[:, np.newaxis]
    V = eigenvectors[:, pc_indices]
    return V.T @ delta_N
