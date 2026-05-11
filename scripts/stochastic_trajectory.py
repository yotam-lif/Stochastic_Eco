"""Stochastic trajectory near the GLV fixed point — SFP regime.

1. Find the deterministic fixed point (no noise).
2. Integrate the SDE (demographic noise) starting at the fixed point.
3. 2D PCA of species-abundance fluctuations around the FP.
4. k-means clustering in PC space (k selected by silhouette sweep).
5. Save a 4-panel figure to scripts/figs/stochastic_trajectory.png.

Usage (from project root):
    .venv/bin/python scripts/stochastic_trajectory.py
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
from composome_analysis import (
    flow_to_fixed_point,
    GLVClusterer,
    plot_pca_trajectory,
    plot_species_traces,
)

# ── Model parameters ──────────────────────────────────────────────────────────
# SFP regime: sigma=0.8 << UFP-MA boundary sqrt(2) ≈ 1.41
S       = 200
MU      = 4.0
SIGMA   = 0.8
GAMMA   = 0.0
SIGMA_K = 0.0
SEED    = 42

# SDE integration
D          = 0.01
T_MAX      = 2000.0
DT         = 0.005
SAVE_EVERY = 20        # store every 20th step → 20 001 saved points
NOISE_TYPE = "demographic"

FIG_DIR  = Path(__file__).parent / "figs"
OUT_PATH = FIG_DIR / "stochastic_trajectory.png"

# ── Build model ───────────────────────────────────────────────────────────────
print(f"GLVModel  S={S}, mu={MU}, sigma={SIGMA}, gamma={GAMMA}, seed={SEED}")
model = GLVModel(S=S, mu=MU, sigma=SIGMA, gamma=GAMMA, sigma_K=SIGMA_K, seed=SEED)
print(f"  u = {model.u:.4f}")

# ── Deterministic fixed point (no noise) ──────────────────────────────────────
print("Finding fixed point (ODE, no noise)...")
fp = flow_to_fixed_point(model, t_max=5000)
print(f"  converged={fp.converged},  phi={fp.phi:.3f}  ({len(fp.surviving)} survivors)")

# ── Stochastic integration ────────────────────────────────────────────────────
print(f"Integrating SDE  T={T_MAX}, dt={DT}, D={D}, noise={NOISE_TYPE}...")
sde = model.integrate_sde(
    N0=fp.N_star,
    t_span=(0.0, T_MAX),
    dt=DT,
    D=D,
    noise_type=NOISE_TYPE,
    save_every=SAVE_EVERY,
)
print(f"  trajectory shape: {sde.N.shape}  (S × n_saved)")

# ── PCA of fluctuations ───────────────────────────────────────────────────────
print("Computing PCA of fluctuations (5 PCs)...")
pca = model.pca(sde, fp=fp, n_pcs=5)
print("  Explained variance per PC:")
for i, ratio in enumerate(pca.explained_ratio[:5]):
    print(f"    PC{i+1}: {100*ratio:.1f}%")

proj_2d = pca.projections[:2, :]   # shape (2, n_times)

# ── k-means clustering in PC space ────────────────────────────────────────────
print("Running k-means sweep in 2D PC space...")
clusterer = GLVClusterer()
result = clusterer.fit(proj_2d.T)   # (n_times, 2)
print(f"  Selected k = {result.selected_k}")
counts = np.bincount(result.labels)
for ci, cnt in enumerate(counts):
    print(f"    Cluster {ci}: {cnt} points ({100*cnt/len(result.labels):.1f}%)")

# ── 4-panel figure ────────────────────────────────────────────────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ① Top-left: species abundance time series
plot_species_traces(
    sde.t, sde.N, axes[0, 0],
    surviving=fp.surviving,
    n_show=5,
    fp_N_star=fp.N_star,
    seed=1,
)

# ② Top-right: PC1 and PC2 vs time
axes[0, 1].plot(sde.t, proj_2d[0], lw=0.7, alpha=0.85, color="#1976d2", label="PC1")
axes[0, 1].plot(sde.t, proj_2d[1], lw=0.7, alpha=0.85, color="#d32f2f", label="PC2")
axes[0, 1].set_xlabel("Time", fontsize=11)
axes[0, 1].set_ylabel("Projection", fontsize=11)
axes[0, 1].set_title("PC1 and PC2 over time", fontsize=11)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.2)

# ③ Bottom-left: PC1 vs PC2 colored by time
plot_pca_trajectory(proj_2d, axes[1, 0], t=sde.t, alpha=0.25, s=3.0)
axes[1, 0].set_title("PC space (colored by time)", fontsize=11)

# ④ Bottom-right: PC1 vs PC2 colored by cluster
plot_pca_trajectory(
    proj_2d, axes[1, 1],
    labels=result.labels,
    centroids=result.centroids,
    alpha=0.25, s=3.0,
)
axes[1, 1].set_title(f"PC space (k={result.selected_k} clusters)", fontsize=11)

fig.suptitle(
    rf"GLV stochastic trajectory  $S$={S}, $\mu$={MU}, $\sigma$={SIGMA}, $D$={D}  "
    rf"$\phi$={fp.phi:.2f}  ({len(fp.surviving)} survivors)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {OUT_PATH}")
