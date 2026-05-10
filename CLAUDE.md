# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local virtual environment at `.venv/` with Python 3.14, numpy, and scipy pre-installed. Always use `.venv/bin/python` to run scripts — the system `python3` does not have the dependencies.

```bash
# Run a script
.venv/bin/python my_script.py

# Interactive session
.venv/bin/python -c "from stochastic_eco import GLVModel; ..."

# Install new packages
.venv/bin/pip install <package>
```

## Architecture

The package (`stochastic_eco/`) implements the generalized Lotka-Volterra model from Bunin (2017, PRE 95, 042414). The core equation is:

```
dN_i/dt = (r_i/K_i) * N_i * (K_i - N_i - Σ_{j≠i} α_ij N_j)
```

**Module responsibilities:**

- `model.py` — `GLVModel` class: the main interface. Holds `alpha`, `K`, `r` arrays and delegates to all other modules. Users import this and call methods on it.
- `interactions.py` — generates the interaction matrix `α_ij = μ/S + σ·a_ij` with correlated pairs (a_ij, a_ji) having correlation γ.
- `dynamics.py` — deterministic ODE integration (`scipy.integrate.solve_ivp` with BDF) and fixed-point finding. Exports `IntegrationResult` and `FixedPoint` namedtuples.
- `linear_stability.py` — Jacobian, community matrix M*, and eigendecomposition at a fixed point. Exports `EigenResult`.
- `stochastic.py` — Euler-Maruyama SDE integration with demographic (multiplicative √N) or additive noise. Exports `SDEResult`.
- `cavity.py` — analytical cavity method (Bunin Eqs. 10-12): self-consistent equations for φ, q, v, h. Exports `CavitySolution`.
- `analysis.py` — PCA of trajectories and Linear Noise Approximation (Lyapunov equation). Exports `PCAResult`, `LNAResult`.

**Key parameters (set at `GLVModel.__init__`):**
- `S`: species pool size
- `mu`: scaled mean interaction = S·⟨α_ij⟩
- `sigma`: scaled heterogeneity = √(S·var(α_ij))
- `gamma`: corr(α_ij, α_ji) ∈ [-1, 1]
- `sigma_K`: std of carrying capacities (mean=1)
- `r_mean`, `r_std`, `r_distribution`: growth rate distribution

**Derived quantity:** `model.u = (1 - mu/S) / sigma ≈ 1/sigma` (controls phase behavior).

**Phase diagram** (Bunin Fig. 2): UFP (stable unique fixed point) vs MA (multiple attractors) vs unbounded growth. The UFP-MA boundary is at σ = √(2/(1+γ)) for σ_K=0.

## Typical workflow for a figure script

```python
from stochastic_eco import GLVModel
from stochastic_eco.analysis import pca_trajectory, linear_noise_approximation, build_diffusion_matrix
from stochastic_eco.cavity import abundance_distribution, phase_boundary_ufp_ma
import numpy as np

# 1. Set up model (in SFP regime: small sigma, moderate mu)
model = GLVModel(S=200, mu=4.0, sigma=1.0, gamma=0.0, sigma_K=0.0, seed=42)

# 2. Find deterministic fixed point
fp = model.find_fixed_point(t_max=3000)   # fp.surviving gives survivor indices

# 3. Linearize around FP
eig = model.eigendecomposition(fp)        # eig.eigenvalues, eig.max_real_part
M_star = model.community_matrix(fp)

# 4. Cavity prediction (analytical)
cav = model.cavity_solve()               # cav.phi, cav.mean_N, cav.var_N

# 5. Stochastic dynamics
sde = model.integrate_sde(N0=fp.N_star, t_span=(0, 1000), dt=0.005,
                            D=0.01, noise_type="demographic", save_every=20)
# sde.N shape: (S, n_saved); sde.t shape: (n_saved,)

# 6. PCA of fluctuations
pca = model.pca(sde, fp=fp, n_pcs=5)
# pca.projections: (n_pcs, n_times) in PC space

# 7. LNA (analytical covariance prediction)
lna = model.lna(fp, D=0.01)
# lna.covariance: (S*, S*) predicted covariance; lna.eigenvectors: PCA modes
```

## Result types (namedtuples — all fields accessible by name or positional unpack)

| Type | Fields |
|------|--------|
| `IntegrationResult` | `t`, `N` |
| `FixedPoint` | `N_star`, `surviving`, `phi`, `converged` |
| `SDEResult` | `t`, `N` |
| `EigenResult` | `eigenvalues`, `eigenvectors`, `max_real_part` |
| `CavitySolution` | `phi`, `q`, `v`, `h`, `delta`, `mean_N`, `var_N`, `sigma_lambda_sq`, `converged` |
| `PCAResult` | `eigenvalues`, `eigenvectors`, `explained_ratio`, `projections`, `mean` |
| `LNAResult` | `covariance`, `eigenvalues`, `eigenvectors`, `explained_ratio` |

All trajectory arrays use shape `(S, n_times)` — species first.

## Key physics notes

- **Stochastic noise**: demographic noise is `dN_i = f_i dt + √(2D N_i) dW_i` (Ito). `D` is the diffusion coefficient. Use `noise_type="additive"` for `D dW_i`.
- **LNA**: solves Lyapunov equation `JC + CJ^T + D_mat = 0` via `scipy.linalg.solve_continuous_lyapunov`. Requires stable FP (all Re(λ_J) < 0).
- **Cavity solver**: parameterizes by Δ, computes q=w₂/w₁², then finds Δ via Brent's method such that u(Δ)=u_target. Self-consistency loop for σ_K > 0.
- **Community matrix** `M*`: defined with self-regulation (diagonal = 1 in α̃), so `J_reduced = -M*` at the FP.
- **Memory**: for large S or long runs, use `save_every` in `integrate_sde` to subsample. `(S=200, dt=0.005, T=2000, save_every=20)` → shape (200, 20001).
