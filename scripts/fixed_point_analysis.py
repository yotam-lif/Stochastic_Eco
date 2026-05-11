"""Fixed-point statistics across 100 independent system realisations — SFP regime.

Each run draws a fresh interaction matrix (new seed), finds the fixed point,
and records population statistics.  This tells us about the *distribution over
disorder* — how phi, survivor abundances, etc. vary as we draw different random
ecosystems from the same (mu, sigma, gamma) ensemble.

Per run:
  1. Build a new GLVModel (fresh alpha, K, r).
  2. Flow the ODE to convergence  (flow_to_fixed_point, t_max=5000).
  3. If flow didn't converge, retry with t_max=20000.
  4. Newton-polish the result    (polish_fixed_point, tol=1e-12).
  5. Record residual max|f(N*)|.

All results are saved to scripts/data/fixed_point_results.npz.
Figure → scripts/figs/fixed_point_analysis.png.

Usage (from project root):
    .venv/bin/python scripts/fixed_point_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from base_structure import GLVModel
from composome_analysis import flow_to_fixed_point, polish_fixed_point

# ── Parameters ────────────────────────────────────────────────────────────────
S       = 200
MU      = 4.0
SIGMA   = 0.8
GAMMA   = 0.0
SIGMA_K = 0.0
N_RUNS  = 100
BASE_SEED = 0   # seed for run i = BASE_SEED + i

T_MAX_FAST = 5_000.0
T_MAX_SLOW = 20_000.0
CONV_TOL   = 1e-8
EXTINCTION_THRESHOLD = 1e-6
POLISH_TOL = 1e-12
RESIDUAL_WARN = 1e-6

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR  = Path(__file__).parent / "figs"
NPZ_PATH = DATA_DIR / "fixed_point_results.npz"
FIG_PATH = FIG_DIR  / "fixed_point_analysis.png"

print(f"GLV ensemble:  S={S}, mu={MU}, sigma={SIGMA}, gamma={GAMMA}")
print(f"Running {N_RUNS} independent realisations …\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
phis           = np.zeros(N_RUNS)
N_stars        = np.zeros((N_RUNS, S))   # full N* (extinct species = 0)
flow_converged = np.zeros(N_RUNS, dtype=bool)
needed_retry   = np.zeros(N_RUNS, dtype=bool)
polish_ok      = np.zeros(N_RUNS, dtype=bool)
residuals      = np.zeros(N_RUNS)

for run in range(N_RUNS):
    if run % 10 == 0:
        print(f"  Realisation {run:3d}/{N_RUNS} …")

    model = GLVModel(S=S, mu=MU, sigma=SIGMA, gamma=GAMMA,
                     sigma_K=SIGMA_K, seed=BASE_SEED + run)

    # step 1: fast flow
    fp = flow_to_fixed_point(model, t_max=T_MAX_FAST,
                             convergence_tol=CONV_TOL,
                             extinction_threshold=EXTINCTION_THRESHOLD)

    # step 2: retry with longer integration if not converged
    if not fp.converged:
        fp = flow_to_fixed_point(model, N0=fp.N_star.copy(),
                                 t_max=T_MAX_SLOW,
                                 convergence_tol=CONV_TOL,
                                 extinction_threshold=EXTINCTION_THRESHOLD)
        needed_retry[run] = True

    flow_converged[run] = fp.converged

    # step 3: newton polish
    N_pol, ok = polish_fixed_point(model, fp, tol=POLISH_TOL)
    polish_ok[run] = ok
    if ok:
        fp = fp._replace(N_star=N_pol)

    # step 4: residual
    residuals[run] = float(np.max(np.abs(model.lotka_volterra_rhs(fp.N_star))))

    phis[run]    = fp.phi
    N_stars[run] = fp.N_star

print(f"\nDone. Summary over {N_RUNS} realisations:")
print(f"  Flow converged (t_max=5k)  : {(flow_converged & ~needed_retry).sum()}")
print(f"  Needed slow retry (t_max=20k): {needed_retry.sum()}")
print(f"  Still not converged         : {(~flow_converged).sum()}")
print(f"  Polish succeeded            : {polish_ok.sum()}")
print(f"  High-residual runs (>{RESIDUAL_WARN:.0e}): {(residuals > RESIDUAL_WARN).sum()}")
print(f"  phi  mean={phis.mean():.3f},  std={phis.std():.4f},  "
      f"min={phis.min():.3f},  max={phis.max():.3f}")

# pooled survivor abundances
all_N = np.concatenate([N_stars[r][N_stars[r] > EXTINCTION_THRESHOLD]
                        for r in range(N_RUNS)])
print(f"  N*  mean={all_N.mean():.4f},  std={all_N.std():.4f},  "
      f"median={np.median(all_N):.4f},  max={all_N.max():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    NPZ_PATH,
    phis=phis,
    N_stars=N_stars,
    flow_converged=flow_converged,
    needed_retry=needed_retry,
    polish_ok=polish_ok,
    residuals=residuals,
    params=np.array([S, MU, SIGMA, GAMMA, SIGMA_K, BASE_SEED, N_RUNS]),
)
print(f"\nData saved → {NPZ_PATH}")

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (0,0) — distribution of phi across realisations
ax = axes[0, 0]
ax.hist(phis, bins=20, color="#1976d2", alpha=0.8, edgecolor="white", lw=0.5)
ax.axvline(phis.mean(), color="#d32f2f", lw=1.5, ls="--",
           label=rf"mean $\phi$ = {phis.mean():.3f} ± {phis.std():.3f}")
ax.set_xlabel(r"Survivor fraction $\phi$", fontsize=12)
ax.set_ylabel("Count (realisations)", fontsize=12)
ax.set_title(r"Distribution of $\phi$ across 100 realisations", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.2, axis="y")

# (0,1) — pooled N* distribution of surviving species (linear scale)
ax = axes[0, 1]
ax.hist(all_N, bins=60, color="#388e3c", alpha=0.8, edgecolor="white", lw=0.3)
ax.axvline(all_N.mean(), color="#d32f2f", lw=1.5, ls="--",
           label=rf"mean = {all_N.mean():.3f}")
ax.axvline(np.median(all_N), color="#f57c00", lw=1.5, ls=":",
           label=rf"median = {np.median(all_N):.3f}")
ax.set_xlabel(r"Survivor abundance $N^*_i$", fontsize=12)
ax.set_ylabel("Count (species × realisations)", fontsize=12)
ax.set_title(
    rf"Pooled $N^*$ distribution  ({len(all_N):,} survivors, {N_RUNS} realisations)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.2, axis="y")

# (1,0) — log10(N*) distribution
ax = axes[1, 0]
log_N = np.log10(all_N)
ax.hist(log_N, bins=60, color="#7b1fa2", alpha=0.8, edgecolor="white", lw=0.3)
ax.axvline(log_N.mean(), color="#d32f2f", lw=1.5, ls="--",
           label=rf"mean $\log_{{10}}N^*$ = {log_N.mean():.3f}")
ax.set_xlabel(r"$\log_{10}(N^*_i)$", fontsize=12)
ax.set_ylabel("Count (species × realisations)", fontsize=12)
ax.set_title(r"Log-scale abundance distribution of survivors", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.2, axis="y")

# (1,1) — residual quality (convergence confidence)
ax = axes[1, 1]
log_res = np.log10(residuals + 1e-20)
ax.hist(log_res, bins=30, color="#f57c00", alpha=0.8, edgecolor="white", lw=0.4)
ax.axvline(np.log10(RESIDUAL_WARN), color="#d32f2f", lw=1.5, ls="--",
           label=rf"warn threshold ({RESIDUAL_WARN:.0e})")
ax.set_xlabel(r"$\log_{10}(\max|f(N^*)|)$", fontsize=12)
ax.set_ylabel("Count (realisations)", fontsize=12)
ax.set_title(
    rf"Fixed-point residual quality  (polish ok: {polish_ok.sum()}/{N_RUNS})",
    fontsize=12,
)
ax.legend(fontsize=9)
ax.grid(alpha=0.2, axis="y")

fig.suptitle(
    rf"Fixed-point population statistics  $S$={S}, $\mu$={MU}, $\sigma$={SIGMA}, "
    rf"$\gamma$={GAMMA}  ({N_RUNS} independent realisations)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved → {FIG_PATH}")