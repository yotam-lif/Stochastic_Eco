"""Fixed-point finding for GLV: deterministic flow-to-convergence + root polish.

Analogous to GARD's flow-relax → root-solve pipeline:
  1. flow_to_fixed_point  — integrate ODE until max|dN/dt| < tol (no noise)
  2. polish_fixed_point   — refine with scipy.optimize.root in the
                            surviving-species subspace
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root


def flow_to_fixed_point(
    model,
    N0: np.ndarray | None = None,
    t_max: float = 5000.0,
    convergence_tol: float = 1e-8,
    extinction_threshold: float = 1e-6,
):
    """Integrate deterministic LV dynamics to convergence (no noise).

    Thin wrapper around ``model.find_fixed_point`` with defaults tuned
    for the SFP regime.

    Parameters
    ----------
    model : GLVModel
    N0 : ndarray (S,) or None
        Initial abundances. If None, model draws Uniform(0.1, 1).
    t_max : float
        Maximum integration time. Default 5000 is longer than the
        model default (2000) for robustness near the UFP-MA boundary.
    convergence_tol : float
        Stop when max|dN/dt| < tol.
    extinction_threshold : float
        Species below this are zeroed (extinct).

    Returns
    -------
    FixedPoint
        .N_star      : ndarray (S,)
        .surviving   : ndarray of int indices
        .phi         : float  fraction surviving
        .converged   : bool
    """
    return model.find_fixed_point(
        N0=N0,
        t_max=t_max,
        extinction_threshold=extinction_threshold,
        convergence_tol=convergence_tol,
    )


def polish_fixed_point(
    model,
    fp,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Newton-polish an ODE-converged fixed point via scipy root-finding.

    Works in the surviving-species subspace to avoid the degeneracy of
    extinct species (where the Jacobian is rank-deficient). This gives
    machine-precision accuracy for downstream eigenvalue analysis.

    Parameters
    ----------
    model : GLVModel
    fp : FixedPoint
        Output of flow_to_fixed_point.
    tol : float
        Root-finding tolerance passed to scipy.optimize.root.

    Returns
    -------
    N_polished : ndarray (S,)
        Refined N_star; extinct species remain at 0.
    success : bool
        Whether root-finding converged.
    """
    s = fp.surviving
    N_init = fp.N_star[s].copy()

    def rhs_surviving(N_s: np.ndarray) -> np.ndarray:
        N_full = np.zeros(model.S)
        N_full[s] = N_s
        return model.lotka_volterra_rhs(N_full)[s]

    result = root(rhs_surviving, N_init, method="hybr", tol=tol)

    N_polished = fp.N_star.copy()
    if result.success:
        N_polished[s] = np.maximum(result.x, 0.0)

    return N_polished, bool(result.success)
