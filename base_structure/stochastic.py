"""
Stochastic integration of the generalized Lotka-Volterra equations
via Euler-Maruyama.

Supports demographic (multiplicative) and additive Gaussian noise.
"""

import numpy as np
from collections import namedtuple

SDEResult = namedtuple("SDEResult", ["t", "N"])


def _lv_drift(N, r, K, alpha):
    """Deterministic drift f_i(N) for the LV equations."""
    interaction = alpha @ N
    return (r / K) * N * (K - N - interaction)


def integrate_sde(r, K, alpha, N0, t_span, dt, D,
                  noise_type="demographic", boundary="reflecting",
                  extinction_threshold=1e-6, save_every=1, rng=None):
    """Euler-Maruyama integration of the stochastic LV system.

    Demographic noise (multiplicative, Ito):
        dN_i = f_i(N) dt + sqrt(2 D N_i) dW_i

    Additive noise:
        dN_i = f_i(N) dt + D dW_i

    Parameters
    ----------
    r, K : ndarray (S,)
    alpha : ndarray (S, S)
    N0 : ndarray (S,)
        Initial abundances.
    t_span : tuple (t0, tf)
    dt : float
        Euler-Maruyama timestep.
    D : float
        Noise amplitude. For demographic noise, this is the diffusion
        coefficient. For additive noise, this is the noise standard
        deviation.
    noise_type : str
        'demographic' or 'additive'.
    boundary : str
        'reflecting': clamp N >= 0 after each step.
        'absorbing': once N_i <= 0, species is permanently extinct.
    extinction_threshold : float
        Species below this are set to 0.
    save_every : int
        Save state every k-th timestep.
    rng : np.random.Generator or None

    Returns
    -------
    SDEResult
        .t : ndarray (n_saved,)
        .N : ndarray (S, n_saved)
    """
    if rng is None:
        rng = np.random.default_rng()

    t0, tf = t_span
    n_steps = int(round((tf - t0) / dt))
    S = len(N0)

    # Preallocate storage
    n_save = n_steps // save_every + 1
    N_history = np.empty((S, n_save))
    t_history = np.empty(n_save)

    N = N0.copy()
    N = np.maximum(N, 0.0)
    N_history[:, 0] = N
    t_history[0] = t0
    save_idx = 1

    extinct = np.zeros(S, dtype=bool)
    sqrt_dt = np.sqrt(dt)

    for step in range(1, n_steps + 1):
        # Drift
        drift = _lv_drift(N, r, K, alpha)

        # Noise
        xi = rng.standard_normal(S)
        if noise_type == "demographic":
            noise = np.sqrt(2.0 * D * np.maximum(N, 0.0)) * sqrt_dt * xi
        elif noise_type == "additive":
            noise = D * sqrt_dt * xi
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # Euler step
        N = N + drift * dt + noise

        # Boundary conditions
        if boundary == "reflecting":
            N = np.maximum(N, 0.0)
        elif boundary == "absorbing":
            newly_extinct = N <= 0
            extinct |= newly_extinct
            N[extinct] = 0.0
        else:
            raise ValueError(f"Unknown boundary: {boundary!r}")

        # Extinction threshold
        below = N < extinction_threshold
        N[below] = 0.0
        if boundary == "absorbing":
            extinct |= below

        # Save
        if step % save_every == 0:
            N_history[:, save_idx] = N
            t_history[save_idx] = t0 + step * dt
            save_idx += 1

    return SDEResult(t=t_history[:save_idx], N=N_history[:, :save_idx])
