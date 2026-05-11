"""
Analytical cavity method solver for the generalized Lotka-Volterra model.

Implements the self-consistent equations from Bunin (2017), Eqs. (10)-(12).
"""

import numpy as np
from collections import namedtuple
from scipy.stats import norm
from scipy.optimize import brentq, root_scalar

CavitySolution = namedtuple("CavitySolution", [
    "phi", "q", "v", "h", "delta",
    "mean_N", "var_N", "sigma_lambda_sq",
    "converged"
])


def _w0(delta):
    """w_0(Delta) = Phi(Delta), the standard normal CDF."""
    return norm.cdf(delta)


def _w1(delta):
    """w_1(Delta) = phi(Delta) + Delta * Phi(Delta)."""
    return norm.pdf(delta) + delta * norm.cdf(delta)


def _w2(delta):
    """w_2(Delta) = (1 + Delta^2) Phi(Delta) + Delta phi(Delta)."""
    return (1.0 + delta**2) * norm.cdf(delta) + delta * norm.pdf(delta)


def _solve_for_delta(u_target, gamma, sigma_lambda_sq=0.0,
                     delta_range=(-10.0, 50.0)):
    """Find Delta such that u(Delta) = u_target.

    Given Delta, the self-consistent equations yield:
        q = w2 / w1^2
        u_hat = sqrt(w2 + sigma_lambda^2 * w1^2)
        v = w0 / u_hat
        u = u_hat + gamma * v

    u(Delta) is monotonically increasing, so Brent's method works.

    Returns (delta, q, v, u_hat) or raises if no solution.
    """
    def u_of_delta(delta):
        w0 = _w0(delta)
        w1 = _w1(delta)
        w2 = _w2(delta)

        if w1 <= 0:
            return -np.inf

        u_hat = np.sqrt(w2 + sigma_lambda_sq * w1**2)
        v = w0 / u_hat
        return u_hat + gamma * v

    # Check bracket validity
    u_lo = u_of_delta(delta_range[0])
    u_hi = u_of_delta(delta_range[1])

    if u_target < u_lo or u_target > u_hi:
        raise ValueError(
            f"u_target={u_target:.4f} outside achievable range "
            f"[{u_lo:.4f}, {u_hi:.4f}] for gamma={gamma}, "
            f"sigma_lambda^2={sigma_lambda_sq}"
        )

    delta_sol = brentq(lambda d: u_of_delta(d) - u_target,
                       delta_range[0], delta_range[1],
                       xtol=1e-12, rtol=1e-12)
    return delta_sol


def solve_cavity(mu, sigma, gamma, sigma_K=0.0, S=None,
                 max_iter=100, tol=1e-10):
    """Solve the self-consistent cavity equations.

    Parameters
    ----------
    mu : float
        Scaled mean interaction.
    sigma : float
        Scaled interaction heterogeneity.
    gamma : float
        Interaction symmetry in [-1, 1].
    sigma_K : float
        Std of carrying capacities.
    S : int or None
        If provided, uses u = (1 - mu/S)/sigma. Otherwise u = 1/sigma.
    max_iter : int
        Max iterations for the sigma_lambda self-consistency loop
        (only needed when sigma_K > 0).
    tol : float
        Convergence tolerance for the self-consistency loop.

    Returns
    -------
    CavitySolution
    """
    if S is not None:
        u_target = (1.0 - mu / S) / sigma
    else:
        u_target = 1.0 / sigma

    if sigma_K == 0.0:
        # No self-consistency needed
        return _solve_given_sigma_lambda(u_target, gamma, mu, sigma,
                                         sigma_lambda_sq=0.0)
    else:
        # Iterate on sigma_lambda
        sigma_lambda_sq = 0.0
        converged = False

        for iteration in range(max_iter):
            sol = _solve_given_sigma_lambda(u_target, gamma, mu, sigma,
                                            sigma_lambda_sq)
            # Update: sigma_lambda^2 = sigma_K^2 / (sigma <N>)^2
            mean_N = sol.mean_N
            if mean_N <= 0:
                break
            new_sigma_lambda_sq = sigma_K**2 / (sigma * mean_N)**2

            if abs(new_sigma_lambda_sq - sigma_lambda_sq) < tol:
                converged = True
                sigma_lambda_sq = new_sigma_lambda_sq
                break

            sigma_lambda_sq = new_sigma_lambda_sq

        sol = _solve_given_sigma_lambda(u_target, gamma, mu, sigma,
                                        sigma_lambda_sq)
        return sol._replace(converged=converged,
                            sigma_lambda_sq=sigma_lambda_sq)


def _solve_given_sigma_lambda(u_target, gamma, mu, sigma, sigma_lambda_sq):
    """Solve cavity equations for fixed sigma_lambda^2."""
    try:
        delta = _solve_for_delta(u_target, gamma, sigma_lambda_sq)
    except ValueError:
        return CavitySolution(
            phi=np.nan, q=np.nan, v=np.nan, h=np.nan,
            delta=np.nan, mean_N=np.nan, var_N=np.nan,
            sigma_lambda_sq=sigma_lambda_sq, converged=False
        )

    w0 = _w0(delta)
    w1 = _w1(delta)
    w2 = _w2(delta)

    phi = w0
    q = w2 / w1**2
    u_hat = np.sqrt(w2 + sigma_lambda_sq * w1**2)
    v = w0 / u_hat

    # h from Eq. (12): h = q * [u_hat/q - v*(1 + gamma + sigma_lambda^2/q) / q]
    # Simplified: h = u_hat - v*(1 + gamma + sigma_lambda^2/q)
    # Wait, Eq 12: h = q[u - v(1 + gamma + sigma_lambda^2/q)]
    # where u here is u_target, not u_hat. Let me use the formula directly.
    # Actually from the paper: h = q[u - v(1+gamma+sigma_lambda^2/q)]
    h = q * (u_target - v * (1.0 + gamma + sigma_lambda_sq / q))

    # Physical quantities
    # <N_i> = 1 / (sigma * h + mu)
    denom = sigma * h + mu
    if denom <= 0:
        mean_N = np.inf
    else:
        mean_N = 1.0 / denom

    # <N_i^2> = q * <N_i>^2
    var_N = (q - 1.0) * mean_N**2 if mean_N < np.inf else np.inf

    return CavitySolution(
        phi=phi, q=q, v=v, h=h, delta=delta,
        mean_N=mean_N, var_N=var_N,
        sigma_lambda_sq=sigma_lambda_sq,
        converged=True
    )


def abundance_distribution(cavity_sol, N_values=None, n_points=200):
    """Predicted abundance distribution P(N) from the cavity solution.

    The distribution is a truncated Gaussian (truncated at N=0).

    Parameters
    ----------
    cavity_sol : CavitySolution
    N_values : ndarray or None
        N values at which to evaluate. If None, auto-generated.
    n_points : int
        Number of points if N_values is None.

    Returns
    -------
    N_values : ndarray (n_points,)
    P_N : ndarray (n_points,)
    """
    mean_N = cavity_sol.mean_N
    phi = cavity_sol.phi
    q = cavity_sol.q

    # Std of the untruncated Gaussian in N space
    # From Eq. 9: n = max(0, (h + sqrt(q + sigma_lambda^2) z) / u_hat)
    # Converting to N = <N> * n: N is truncated Gaussian
    std_N = np.sqrt(q) * mean_N

    if N_values is None:
        N_max = mean_N + 4 * std_N
        N_values = np.linspace(0, max(N_max, 1e-10), n_points)

    # The truncated Gaussian: P(N) = phi * normal_pdf(N; mean_N, std_N) / Phi(mean_N/std_N)
    # where Phi(mean_N/std_N) accounts for the truncation
    z_scores = (N_values - mean_N) / std_N
    truncation_factor = norm.cdf(mean_N / std_N)
    P_N = phi * norm.pdf(z_scores) / (std_N * truncation_factor)

    # Zero out negative N
    P_N[N_values < 0] = 0.0
    # Add a delta function weight at N=0 for extinct species
    # (not included here — the user sees phi as the survival fraction)

    return N_values, P_N


def phase_boundary_ufp_ma(gamma, sigma_K=0.0, mu_range=(0.1, 10.0),
                          n_points=100):
    """Trace the UFP-MA phase boundary.

    The transition occurs when phi = (u - gamma*v)^2.
    For sigma_K = 0: sigma_c = sqrt(2) / (1 + gamma).

    Parameters
    ----------
    gamma : float
    sigma_K : float
    mu_range : tuple
    n_points : int

    Returns
    -------
    mu_values : ndarray
    sigma_critical : ndarray
    """
    if sigma_K == 0.0:
        # Exact result: sigma_c = sqrt(2/(1+gamma)) for all mu > 0
        sigma_c = np.sqrt(2.0 / (1.0 + gamma))
        mu_values = np.linspace(mu_range[0], mu_range[1], n_points)
        return mu_values, np.full(n_points, sigma_c)

    # For sigma_K > 0, need to find sigma_c(mu) numerically
    mu_values = np.linspace(mu_range[0], mu_range[1], n_points)
    sigma_critical = np.empty(n_points)

    for i, mu in enumerate(mu_values):
        def stability_criterion(sigma):
            if sigma <= 0:
                return -1.0
            try:
                sol = solve_cavity(mu, sigma, gamma, sigma_K)
                u_hat = sol.q**0.5 * _w1(sol.delta) / _w0(sol.delta) \
                    if sol.phi > 0 else 1e10
                # Actually u_hat = u - gamma*v
                u = 1.0 / sigma
                u_hat_val = u - gamma * sol.v
                return u_hat_val**2 - sol.phi
            except (ValueError, RuntimeError):
                return 1.0

        # Search for sigma where criterion = 0
        try:
            from scipy.optimize import brentq
            sigma_critical[i] = brentq(stability_criterion, 0.1, 10.0,
                                       xtol=1e-6)
        except ValueError:
            sigma_critical[i] = np.nan

    return mu_values, sigma_critical
