"""Eigenvalue distribution at the GLV fixed point — SFP regime.

Computes the Jacobian eigenspectrum at the deterministic fixed point,
compares it with the cavity-method spectral prediction, and saves a
two-panel figure (complex-plane scatter + Re(λ) histogram).

Usage (from project root):
    .venv/bin/python scripts/eigenvalue_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from base_structure import GLVModel
from composome_analysis import (
    flow_to_fixed_point,
    polish_fixed_point,
    plot_eigenvalues_complex,
    plot_eigenvalue_histogram,
)

# ── Model parameters ──────────────────────────────────────────────────────────
# SFP regime: sigma=0.8 is well below the UFP-MA boundary sqrt(2/(1+gamma)) ≈ 1.41
S       = 200
MU      = 4.0
SIGMA   = 0.8
GAMMA   = 0.0
SIGMA_K = 0.0
SEED    = 42

FIG_DIR  = Path(__file__).parent / "figs"
OUT_PATH = FIG_DIR / "eigenvalue_analysis.png"

# ── Build model ───────────────────────────────────────────────────────────────
print(f"GLVModel  S={S}, mu={MU}, sigma={SIGMA}, gamma={GAMMA}, seed={SEED}")
model = GLVModel(S=S, mu=MU, sigma=SIGMA, gamma=GAMMA, sigma_K=SIGMA_K, seed=SEED)
print(f"  u = {model.u:.4f}  (u >> 1 → deep SFP)")

# ── Fixed point: ODE flow → root polish ──────────────────────────────────────
print("Finding fixed point (ODE flow, no noise)...")
fp = flow_to_fixed_point(model, t_max=5000)
print(f"  converged={fp.converged},  phi={fp.phi:.3f}  ({len(fp.surviving)}/{S} survivors)")

print("Polishing fixed point (scipy root-solve)...")
N_polished, polish_ok = polish_fixed_point(model, fp)
if polish_ok:
    fp = fp._replace(N_star=N_polished)
    print("  polish succeeded")
else:
    print("  polish did not converge — using ODE result")

# ── Eigendecomposition ────────────────────────────────────────────────────────
print("Computing Jacobian eigendecomposition...")
eig = model.eigendecomposition(fp)
print(f"  max Re(lambda) = {eig.max_real_part:.6e}  (< 0 → stable FP)")

# ── Cavity prediction ─────────────────────────────────────────────────────────
print("Solving cavity equations...")
cav = model.cavity_solve()
print(f"  cavity converged={cav.converged},  phi_cav={cav.phi:.3f}")

sigma_lambda: float | None = None
if cav.converged and cav.sigma_lambda_sq is not None:
    sigma_lambda = float(np.sqrt(max(cav.sigma_lambda_sq, 0.0)))
    print(f"  sigma_lambda (predicted spectral width) = {sigma_lambda:.4f}")

# ── Print eigenvalue table ────────────────────────────────────────────────────
print(f"\nTop-20 eigenvalues of J restricted to {len(fp.surviving)} survivors")
print(f"  (sorted by descending Re(λ))")
print(f"{'#':>4}  {'Re(λ)':>14}  {'Im(λ)':>14}  {'|λ|':>12}")
print("-" * 52)
for i, lam in enumerate(eig.eigenvalues[:20]):
    sign = "+" if lam.imag >= 0 else "-"
    print(
        f"{i+1:4d}  {lam.real:14.6e}  {sign}{abs(lam.imag):13.6e}  {abs(lam):12.6e}"
    )

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

plot_eigenvalues_complex(
    eig.eigenvalues,
    axes[0],
    title=rf"Complex plane  ($S$={S}, $\mu$={MU}, $\sigma$={SIGMA})",
    sigma_lambda=sigma_lambda,
)

# ── Panel 2: histogram of log10(−Re(λ)) ──────────────────────────────────────
re = np.real(eig.eigenvalues)
re_neg = re[re < 0]   # guard: skip any zero/positive (shouldn't occur at stable FP)
n_re_skipped = len(re) - len(re_neg)
log_neg_re = np.log10(-re_neg)
axes[1].hist(log_neg_re, bins=30, color="#1976d2", alpha=0.7,
             edgecolor="white", lw=0.4)
if sigma_lambda is not None and sigma_lambda > 0:
    axes[1].axvline(
        np.log10(sigma_lambda), color="#d32f2f", lw=1.5, ls="--",
        label=rf"$\log_{{10}}\sigma_\lambda$",
    )
    axes[1].legend(fontsize=9)
axes[1].set_xlabel(r"$\log_{10}(-\mathrm{Re}(\lambda))$", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
skip_note = f"  ({n_re_skipped} skipped: Re≥0)" if n_re_skipped else ""
axes[1].set_title(rf"$\log_{{10}}(-\mathrm{{Re}}(\lambda))$ distribution{skip_note}",
                  fontsize=11)
axes[1].grid(alpha=0.2, axis="y")

# ── Panel 3: histogram of log10(|Im(λ)|) ─────────────────────────────────────
IM_THRESHOLD = 1e-10
im = np.imag(eig.eigenvalues)
im_nonzero = im[np.abs(im) > IM_THRESHOLD]
n_real_only = len(im) - len(im_nonzero)
if len(im_nonzero) > 0:
    log_abs_im = np.log10(np.abs(im_nonzero))
    axes[2].hist(log_abs_im, bins=30, color="#388e3c", alpha=0.7,
                 edgecolor="white", lw=0.4)
axes[2].set_xlabel(r"$\log_{10}(|\mathrm{Im}(\lambda)|)$", fontsize=11)
axes[2].set_ylabel("Count", fontsize=11)
axes[2].set_title(
    rf"$\log_{{10}}(|\mathrm{{Im}}(\lambda)|)$ distribution  "
    rf"({n_real_only} purely real excluded)",
    fontsize=11,
)
axes[2].grid(alpha=0.2, axis="y")

fig.suptitle(
    f"GLV fixed-point eigenspectrum   "
    rf"$\phi$={fp.phi:.2f}   max Re($\lambda$)={eig.max_real_part:.2e}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {OUT_PATH}")
