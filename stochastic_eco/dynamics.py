"""
Deterministic integration and fixed-point finding for the generalized
Lotka-Volterra equations.

    dN_i/dt = (r_i / K_i) * N_i * (K_i - N_i - sum_{j!=i} alpha_ij N_j)
"""

import numpy as np
from collections import namedtuple
from scipy.integrate import solve_ivp

IntegrationResult = namedtuple("IntegrationResult", ["t", "N"])
FixedPoint = namedtuple("FixedPoint", ["N_star", "surviving", "phi", "converged"])


def lv_rhs(N, r, K, alpha):
    """Compute the RHS of the generalized Lotka-Volterra equations.

    Parameters
    ----------
    N : ndarray (S,)
        Current abundances (will be clamped to >= 0).
    r : ndarray (S,)
        Intrinsic growth rates.
    K : ndarray (S,)
        Carrying capacities.
    alpha : ndarray (S, S)
        Interaction matrix (zero diagonal).

    Returns
    -------
    dNdt : ndarray (S,)
    """
    N = np.maximum(N, 0.0)
    interaction = alpha @ N
    return (r / K) * N * (K - N - interaction)


def integrate(r, K, alpha, N0, t_span, t_eval=None,
              extinction_threshold=1e-6, method="BDF", **solve_ivp_kwargs):
    """Integrate the deterministic LV dynamics.

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
    N0 : ndarray (S,)
        Initial abundances.
    t_span : tuple (t0, tf)
    t_eval : ndarray or None
        Times at which to store the solution.
    extinction_threshold : float
        Species below this are clamped to 0 in the RHS.
    method : str
        Integration method for solve_ivp. Default 'BDF' for stiff systems.
    **solve_ivp_kwargs
        Additional arguments passed to solve_ivp (rtol, atol, etc.).

    Returns
    -------
    IntegrationResult
        .t : ndarray (n_times,)
        .N : ndarray (S, n_times)
    """
    S = len(N0)

    def rhs(t, N_flat):
        N_clamped = np.maximum(N_flat, 0.0)
        N_clamped[N_clamped < extinction_threshold] = 0.0
        interaction = alpha @ N_clamped
        return (r / K) * N_clamped * (K - N_clamped - interaction)

    kwargs = dict(rtol=1e-8, atol=1e-10)
    kwargs.update(solve_ivp_kwargs)

    sol = solve_ivp(rhs, t_span, N0, method=method,
                    t_eval=t_eval, **kwargs)

    if not sol.success:
        import warnings
        warnings.warn(f"Integration warning: {sol.message}")

    return IntegrationResult(t=sol.t, N=sol.y)


def find_fixed_point(r, K, alpha, N0=None, t_max=2000,
                     extinction_threshold=1e-6, convergence_tol=1e-8,
                     rng=None, method="BDF"):
    """Run dynamics to convergence and return the fixed point.

    Integrates in chunks and checks convergence after each chunk.

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
    N0 : ndarray (S,) or None
        Initial abundances. If None, samples uniformly on [0, 1].
    t_max : float
        Maximum integration time.
    extinction_threshold : float
        Species below this threshold are considered extinct.
    convergence_tol : float
        Convergence criterion: max(|dN/dt|) < tol.
    rng : np.random.Generator or None
        Used if N0 is None.
    method : str
        Integration method.

    Returns
    -------
    FixedPoint
        .N_star : ndarray (S,)
        .surviving : ndarray of int indices
        .phi : float (fraction surviving)
        .converged : bool
    """
    S = len(r)
    if N0 is None:
        if rng is not None:
            N0 = rng.uniform(0.1, 1.0, size=S)
        else:
            N0 = np.full(S, 0.5)

    chunk_size = min(500.0, t_max)
    t_current = 0.0
    N_current = N0.copy()
    converged = False

    while t_current < t_max:
        t_end = min(t_current + chunk_size, t_max)
        result = integrate(r, K, alpha, N_current,
                           t_span=(t_current, t_end),
                           extinction_threshold=extinction_threshold,
                           method=method)

        N_current = result.N[:, -1].copy()
        N_current[N_current < extinction_threshold] = 0.0

        # Check convergence
        dNdt = lv_rhs(N_current, r, K, alpha)
        if np.max(np.abs(dNdt)) < convergence_tol:
            converged = True
            break

        t_current = t_end

    N_star = N_current
    N_star[N_star < extinction_threshold] = 0.0
    surviving = np.where(N_star > 0)[0]
    phi = len(surviving) / S

    return FixedPoint(N_star=N_star, surviving=surviving,
                      phi=phi, converged=converged)
