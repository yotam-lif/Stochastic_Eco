"""
GLVModel: facade class for the generalized Lotka-Volterra model.

Holds all model parameters and delegates computation to specialized modules.
"""

import numpy as np
from . import interactions, dynamics, stochastic, linear_stability, cavity, analysis


class GLVModel:
    """Generalized Lotka-Volterra model following Bunin (2017).

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
    sigma_K : float
        Std of carrying capacities K_i (mean = 1, Gaussian).
    r_mean : float
        Mean intrinsic growth rate.
    r_std : float
        Std of intrinsic growth rates. 0 means all equal to r_mean.
    r_distribution : str
        Distribution for r_i: 'constant', 'normal', 'uniform', 'lognormal'.
    seed : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    alpha : ndarray (S, S)
        Interaction matrix.
    K : ndarray (S,)
        Carrying capacities.
    r : ndarray (S,)
        Intrinsic growth rates.
    rng : np.random.Generator
    """

    def __init__(self, S, mu, sigma, gamma=0.0, sigma_K=0.0,
                 r_mean=1.0, r_std=0.0, r_distribution="constant",
                 seed=None):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.sigma_K = sigma_K
        self.r_mean = r_mean
        self.r_std = r_std
        self.r_distribution = r_distribution

        self.rng = np.random.default_rng(seed)

        self._generate_all()

    def _generate_all(self):
        self._generate_alpha()
        self._generate_K()
        self._generate_r()

    def _generate_alpha(self):
        self.alpha = interactions.generate_interaction_matrix(
            self.S, self.mu, self.sigma, self.gamma, self.rng
        )

    def _generate_K(self):
        if self.sigma_K == 0.0:
            self.K = np.ones(self.S)
        else:
            self.K = self.rng.normal(1.0, self.sigma_K, size=self.S)

    def _generate_r(self):
        if self.r_distribution == "constant" or self.r_std == 0.0:
            self.r = np.full(self.S, self.r_mean)
        elif self.r_distribution == "normal":
            self.r = self.rng.normal(self.r_mean, self.r_std, size=self.S)
            self.r = np.maximum(self.r, 1e-10)
        elif self.r_distribution == "uniform":
            half = self.r_std * np.sqrt(3)
            self.r = self.rng.uniform(self.r_mean - half,
                                      self.r_mean + half, size=self.S)
            self.r = np.maximum(self.r, 1e-10)
        elif self.r_distribution == "lognormal":
            # Parameterize so that mean and std match
            var = self.r_std**2
            mu_ln = np.log(self.r_mean**2 / np.sqrt(var + self.r_mean**2))
            sigma_ln = np.sqrt(np.log(1 + var / self.r_mean**2))
            self.r = self.rng.lognormal(mu_ln, sigma_ln, size=self.S)
        else:
            raise ValueError(f"Unknown r_distribution: {self.r_distribution!r}")

    def resample(self, what="all", seed=None):
        """Regenerate specified parameters.

        Parameters
        ----------
        what : str or list of str
            'all', 'alpha', 'K', 'r', or a list like ['alpha', 'K'].
        seed : int or None
            If provided, reseeds the RNG first.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if isinstance(what, str):
            what = [what]

        if "all" in what:
            self._generate_all()
        else:
            if "alpha" in what:
                self._generate_alpha()
            if "K" in what:
                self._generate_K()
            if "r" in what:
                self._generate_r()

    @property
    def u(self):
        """Derived parameter u = (1 - mu/S) / sigma."""
        return (1.0 - self.mu / self.S) / self.sigma

    # ---- Deterministic dynamics ----

    def lotka_volterra_rhs(self, N):
        """Compute dN/dt for given abundances."""
        return dynamics.lv_rhs(N, self.r, self.K, self.alpha)

    def integrate(self, N0=None, t_span=(0, 500), t_eval=None,
                  extinction_threshold=1e-6, **kwargs):
        """Integrate deterministic LV dynamics.

        Parameters
        ----------
        N0 : ndarray (S,) or None
            Initial abundances. If None, uses uniform [0.1, 1].
        t_span : tuple (t0, tf)
        t_eval : ndarray or None
        extinction_threshold : float
        **kwargs
            Passed to scipy.integrate.solve_ivp.

        Returns
        -------
        IntegrationResult
            .t : ndarray (n_times,)
            .N : ndarray (S, n_times)
        """
        if N0 is None:
            N0 = self.rng.uniform(0.1, 1.0, size=self.S)
        return dynamics.integrate(self.r, self.K, self.alpha, N0,
                                  t_span, t_eval, extinction_threshold,
                                  **kwargs)

    def find_fixed_point(self, N0=None, t_max=2000,
                         extinction_threshold=1e-6,
                         convergence_tol=1e-8, **kwargs):
        """Run dynamics to convergence and return the fixed point.

        Returns
        -------
        FixedPoint
            .N_star : ndarray (S,)
            .surviving : ndarray of int
            .phi : float
            .converged : bool
        """
        return dynamics.find_fixed_point(
            self.r, self.K, self.alpha, N0, t_max,
            extinction_threshold, convergence_tol, rng=self.rng,
            **kwargs
        )

    # ---- Linear stability ----

    def jacobian(self, N):
        """Jacobian of the LV system at abundances N."""
        return linear_stability.jacobian(self.r, self.K, self.alpha, N)

    def community_matrix(self, fp):
        """Community matrix M* for surviving species."""
        return linear_stability.community_matrix(
            self.r, self.K, self.alpha, fp
        )

    def eigendecomposition(self, fp):
        """Eigenvalues and eigenvectors at the fixed point.

        Returns
        -------
        EigenResult
            .eigenvalues : ndarray (S*,) complex
            .eigenvectors : ndarray (S*, S*)
            .max_real_part : float
        """
        return linear_stability.eigendecomposition(
            self.r, self.K, self.alpha, fp
        )

    # ---- Stochastic dynamics ----

    def integrate_sde(self, N0=None, t_span=(0, 500), dt=0.01,
                      D=0.01, noise_type="demographic",
                      boundary="reflecting", extinction_threshold=1e-6,
                      save_every=1):
        """Integrate stochastic LV dynamics.

        Parameters
        ----------
        N0 : ndarray (S,) or None
            Initial abundances. Typically set to fp.N_star.
        t_span : tuple (t0, tf)
        dt : float
            Euler-Maruyama timestep.
        D : float
            Noise amplitude.
        noise_type : str
            'demographic' or 'additive'.
        boundary : str
            'reflecting' or 'absorbing'.
        extinction_threshold : float
        save_every : int
            Store every k-th timestep.

        Returns
        -------
        SDEResult
            .t : ndarray (n_saved,)
            .N : ndarray (S, n_saved)
        """
        if N0 is None:
            N0 = self.rng.uniform(0.1, 1.0, size=self.S)
        return stochastic.integrate_sde(
            self.r, self.K, self.alpha, N0, t_span, dt, D,
            noise_type, boundary, extinction_threshold, save_every,
            rng=self.rng
        )

    # ---- Cavity method ----

    def cavity_solve(self):
        """Solve the analytical cavity equations.

        Returns
        -------
        CavitySolution
            .phi, .q, .v, .h, .delta, .mean_N, .var_N, .converged
        """
        return cavity.solve_cavity(
            self.mu, self.sigma, self.gamma, self.sigma_K, S=self.S
        )

    # ---- Analysis helpers ----

    def pca(self, sde_result, fp=None, n_pcs=None):
        """PCA of a stochastic trajectory.

        Parameters
        ----------
        sde_result : SDEResult
        fp : FixedPoint or None
            If given, restricts PCA to surviving species.
        n_pcs : int or None

        Returns
        -------
        PCAResult
        """
        surviving = fp.surviving if fp is not None else None
        return analysis.pca_trajectory(sde_result.N, surviving, n_pcs)

    def lna(self, fp, D, noise_type="demographic"):
        """Linear Noise Approximation at a fixed point.

        Parameters
        ----------
        fp : FixedPoint
        D : float
            Noise amplitude (same as in integrate_sde).
        noise_type : str

        Returns
        -------
        LNAResult
        """
        s = fp.surviving
        J = self.jacobian(fp.N_star)[np.ix_(s, s)]
        D_mat = analysis.build_diffusion_matrix(
            fp.N_star[s], D, noise_type
        )
        return analysis.linear_noise_approximation(J, D_mat)
