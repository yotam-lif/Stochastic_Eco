# Stochastic_Eco

Probing generalized Lotka-Volterra (GLV) models with demographic noise in the stable fixed-point (SFP) regime. The deterministic core follows Bunin (2017, PRE 95, 042414); the stochastic layer adds Itô noise and analyzes the resulting fluctuations analytically (Linear Noise Approximation) and numerically (PCA).

---

## The GLV model

Each of the S species has abundance N_i(t) evolving under:

```
dN_i/dt = (r_i / K_i) * N_i * (K_i - N_i - Σ_{j≠i} α_ij N_j)
```

- **r_i** — intrinsic growth rate. Without interactions, species i grows logistically toward K_i.
- **K_i** — carrying capacity (mean = 1). Heterogeneity in K is controlled by `sigma_K`.
- **α_ij** — per-capita effect of species j on species i. Positive α_ij means competition; negative means facilitation. The diagonal is zero (self-regulation enters through the K_i - N_i term).

At a fixed point N* the LV equation reduces to `K_i - N*_i - (α N*)_i = 0` for every surviving species.

*Implemented in `dynamics.py`: `lv_rhs`, `integrate`, `find_fixed_point`.*

---

## Random interaction matrices

For large S the interaction matrix is drawn from a random ensemble (Bunin Eq. 2):

```
α_ij = μ/S  +  σ · a_ij
```

where a_ij has mean 0 and variance 1/S. The dimensionless parameters are:

- **μ** = S · ⟨α_ij⟩ — scaled mean interaction. Positive μ means, on average, species compete.
- **σ** = √(S · Var(α_ij)) — scaled interaction heterogeneity. Controls how much interactions vary across pairs.
- **γ = Corr(α_ij, α_ji)** — symmetry of the interaction matrix. γ = +1 → fully symmetric (reciprocal interactions); γ = −1 → fully antisymmetric (predator-prey-like); γ = 0 → no correlation.

Pairs (a_ij, a_ji) are drawn from a bivariate normal with correlation γ. The 1/S scaling ensures that the total competitive pressure on any one species, Σ_j α_ij N_j, stays O(1) as S → ∞.

*Implemented in `interactions.py`: `generate_interaction_matrix`.*

---

## Phase diagram

In the thermodynamic limit (S → ∞) the model has three phases determined by (μ, σ, γ):

| Phase | Description |
|-------|-------------|
| **UFP** — Unique Fixed Point | All random initial conditions converge to the same fixed point. The SFP regime lives here. |
| **MA** — Multiple Attractors | The steady state depends on initial conditions; exponentially many fixed points coexist. |
| **Unbounded** | No stable fixed point; abundances diverge. |

The UFP–MA boundary (at σ_K = 0) is:

```
σ_c = √(2 / (1 + γ))
```

This is exact and independent of μ. For γ = 0 (uncorrelated), σ_c = √2. More symmetric matrices (larger γ) destabilize the unique fixed point at smaller σ.

*Implemented in `cavity.py`: `phase_boundary_ufp_ma`.*

---

## Fixed points and linear stability

### Fixed-point finding

The deterministic ODE is integrated with `scipy.solve_ivp` (BDF method, good for stiff systems) until `max|dN/dt| < tol`. Species below an extinction threshold are set to zero; the surviving set S* and their abundances N* are returned.

*Implemented in `dynamics.py`: `find_fixed_point`.*

### Jacobian

Linearizing around any point N gives the Jacobian:

```
J_ii = (r_i/K_i)(K_i - 2N_i - (αN)_i)
J_ij = -(r_i/K_i) N_i α_ij   (j ≠ i)
```

At a fixed point, K_i - N*_i - (αN*)_i = 0 for survivors, so J_ii = −(r_i/K_i) N*_i. The diagonal is always negative — each species is locally self-regulating.

*Implemented in `linear_stability.py`: `jacobian`.*

### Community matrix

The community matrix M* is defined so that J_reduced = −M* at the fixed point (restricted to S* surviving species):

```
M_ij = (r_i N*_i / K_i) · α̃_ij
```

where α̃ equals α off-diagonal and 1 on the diagonal (explicit self-regulation). Stability of the fixed point is equivalent to M* being positive definite (all eigenvalues > 0).

*Implemented in `linear_stability.py`: `community_matrix`, `eigendecomposition`.*

---

## Cavity method (analytical)

The cavity method provides exact predictions for macroscopic observables in the S → ∞ limit. The key insight is that species i experiences all other species as an effective Gaussian field, whose statistics can be solved self-consistently.

### Self-consistent equations

The cavity solution is parameterized by a single scalar Δ ("distance from the extinction boundary in units of the effective field fluctuations"). Three integral moments of the truncated Gaussian appear:

```
w_0(Δ) = Φ(Δ)                                      [survival probability φ]
w_1(Δ) = φ_gauss(Δ) + Δ Φ(Δ)                       [mean abundance]
w_2(Δ) = (1+Δ²) Φ(Δ) + Δ φ_gauss(Δ)               [second moment]
```

where Φ is the standard normal CDF and φ_gauss is its PDF. The self-consistent equations are:

```
q   = w_2 / w_1²           [ratio of 2nd to 1st moment squared]
û   = √(w_2 + σ_λ² w_1²)   [effective field std, σ_λ captures K heterogeneity]
v   = w_0 / û              [susceptibility]
u   = û + γ v               [must equal u_target = (1 − μ/S)/σ]
```

Δ is found numerically (Brent's method) so that u(Δ) = u_target. When σ_K > 0, an additional self-consistency loop on σ_λ² = σ_K² / (σ ⟨N⟩)² is required.

### Predicted observables

| Observable | Formula |
|------------|---------|
| Survival fraction φ | φ = w_0(Δ) |
| Mean abundance ⟨N⟩ | 1 / (σh + μ) |
| Variance of abundance Var(N) | (q−1)·⟨N⟩² |
| Abundance distribution P(N) | Truncated Gaussian, mean ⟨N⟩, std √q ·⟨N⟩ |

The truncated Gaussian shape for P(N) is a core prediction: it results from the central-limit theorem applied to the effective field each species experiences, with the truncation at N = 0 representing extinction.

*Implemented in `cavity.py`: `solve_cavity`, `abundance_distribution`, `phase_boundary_ufp_ma`.*

---

## Stochastic dynamics

In the SFP regime the deterministic dynamics has a stable fixed point N*. Demographic noise drives persistent fluctuations around it.

### Demographic noise (Itô SDE)

Population size fluctuations arising from birth/death stochasticity scale as √N (shot noise):

```
dN_i = f_i(N) dt + √(2D N_i) dW_i
```

where f_i is the deterministic LV drift, D is the diffusion coefficient, and dW_i are independent Wiener increments. This is the Itô convention; the √N scaling means larger populations fluctuate more in absolute terms but less relative to their size. D sets the overall noise amplitude.

### Additive noise

An alternative where each species receives an equal noise kick regardless of abundance:

```
dN_i = f_i(N) dt + D dW_i
```

This is less physical for population dynamics but useful as a reference.

### Integration

Both variants are integrated with the Euler-Maruyama scheme (first-order strong scheme for Itô SDEs). Boundary conditions are either reflecting (clamp N ≥ 0) or absorbing (species that hit zero stay extinct). A `save_every` parameter subsamples the trajectory to control memory use.

*Implemented in `stochastic.py`: `integrate_sde`.*

---

## Linear Noise Approximation (LNA)

The LNA gives an exact (in the limit of small noise) analytical prediction for the covariance of fluctuations around the fixed point.

### Lyapunov equation

Write N_i(t) = N*_i + δN_i(t). Linearizing the SDE around N* gives:

```
d(δN) = J · δN dt + B(N*) · dW
```

where J is the Jacobian at N* and B encodes the noise amplitude. In steady state the covariance matrix C = ⟨δN δNᵀ⟩ satisfies the continuous Lyapunov equation:

```
J C + C Jᵀ + D_mat = 0
```

where D_mat is the diffusion matrix (diagonal, D_ii = 2D N*_i for demographic noise). This equation has a unique solution when all eigenvalues of J have negative real part — precisely the SFP condition.

### What the LNA predicts

- **Covariance structure** C among species: which species fluctuate together and which anti-correlate.
- **Principal modes** of fluctuation: the eigenvectors of C give the collective modes, their eigenvalues give the variance in each mode.
- **Comparison with PCA**: the LNA eigenvectors should match the PCA modes extracted from the actual SDE trajectory. Agreement validates both the linearization and the numerics.

The Lyapunov equation is solved with `scipy.linalg.solve_continuous_lyapunov`.

*Implemented in `analysis.py`: `linear_noise_approximation`, `build_diffusion_matrix`.*

---

## PCA of trajectories

After running the SDE, PCA decomposes the empirical fluctuations:

1. Compute δN(t) = N(t) − ⟨N⟩_t (temporal mean subtracted).
2. Form the S* × S* covariance matrix C_emp = δN δNᵀ / (T−1).
3. Eigendecompose C_emp; eigenvectors are the empirical principal modes, eigenvalues give mode variances.
4. Project the trajectory onto the top PCs for visualization.

Comparing C_emp with the LNA prediction C tests whether fluctuations are well described by linear theory. In the SFP regime far from the MA boundary the LNA is highly accurate; near the transition fluctuations grow and the linear approximation breaks down.

*Implemented in `analysis.py`: `pca_trajectory`, `project_trajectory`.*

---

## Key dimensionless control parameter

The quantity:

```
u = (1 − μ/S) / σ  ≈  1/σ  (for S → ∞)
```

controls the phase behavior. It is roughly the inverse of the interaction heterogeneity. Large u (small σ) → UFP regime; decreasing u toward the critical value signals the approach to the MA transition. The cavity Δ is set by inverting u(Δ) = u_target.