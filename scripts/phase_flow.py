"""Phase-flow visualization for GLV near the fixed point — SFP regime.

Overlays the deterministic velocity field (quiver, colored by log10 speed)
onto the stochastic trajectory in 2D PC space, analogous to GARD's
pca_phase_flow.py. The slowest right singular vector of J is shown as
an arrow, indicating the direction of the slow zone.

Start: deterministic FP. Noise immediately kicks the system into the FP
neighborhood. No thermalization assumed — we observe whatever the trajectory
samples from the very beginning.

Usage (from project root):
    .venv/bin/python scripts/phase_flow.py

Tune the PRIMARY KNOBS below and re-run to experiment.
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
    analyze_speed_geometry,
    print_speed_geometry,
    plot_phase_flow,
    plot_pca_trajectory,
)

# ── Model parameters (change these later, after exploring simulation params) ──
S       = 200
MU      = 4.0
SIGMA   = 0.8
GAMMA   = 0.0
SIGMA_K = 0.0
SEED    = 42

# ── PRIMARY KNOBS — experiment here first ────────────────────────────────────
D          = 0.01      # noise amplitude        — try: 0.001, 0.01, 0.1
T_MAX      = 5000.0    # total integration time — try: 2000, 5000, 20000
DT         = 0.005     # Euler-Maruyama step    — try: 0.005, 0.01, 0.02
SAVE_EVERY = 20        # keep every k-th step   — adjust with T_MAX

NOISE_TYPE = "demographic"

# ── Velocity-field grid ───────────────────────────────────────────────────────
GRID_SIZE = 20         # 20×20 = 400 arrows
PAD_FRAC  = 0.2        # extend grid beyond trajectory bounding box

FIG_DIR  = Path(__file__).parent / "figs"
OUT_PATH = FIG_DIR / f"phase_flow_D{D}_T{int(T_MAX)}_dt{DT}.png"

# ── Build model & fixed point ─────────────────────────────────────────────────
print(f"GLVModel  S={S}, mu={MU}, sigma={SIGMA}, gamma={GAMMA}, seed={SEED}")
model = GLVModel(S=S, mu=MU, sigma=SIGMA, gamma=GAMMA, sigma_K=SIGMA_K, seed=SEED)

print("Finding fixed point (ODE, no noise)...")
fp = flow_to_fixed_point(model, t_max=5000)
print(f"  converged={fp.converged},  phi={fp.phi:.3f}  ({len(fp.surviving)} survivors)")

# ── Speed geometry: SVD of J at FP ───────────────────────────────────────────
print("\nSpeed geometry (SVD of Jacobian):")
geom = analyze_speed_geometry(model, fp, slow_threshold=0.1)
print_speed_geometry(geom, n_show=10)

# ── SDE: start at det. FP, noise kicks immediately ───────────────────────────
print(f"\nIntegrating SDE  T={T_MAX}, dt={DT}, D={D}, noise={NOISE_TYPE}...")
sde = model.integrate_sde(
    N0=fp.N_star,
    t_span=(0.0, T_MAX),
    dt=DT,
    D=D,
    noise_type=NOISE_TYPE,
    save_every=SAVE_EVERY,
)
print(f"  trajectory shape: {sde.N.shape}  (S × n_saved)")

# ── PCA on trajectory (surviving species) ────────────────────────────────────
print("Computing 2D PCA of trajectory...")
pca = model.pca(sde, fp=fp, n_pcs=2)
V   = pca.eigenvectors      # shape (S*, 2): PC directions in survivor space
proj = pca.projections      # shape (2, n_times)
print(f"  PC1: {100*pca.explained_ratio[0]:.1f}%   PC2: {100*pca.explained_ratio[1]:.1f}%")

# ── Cluster in PC space ───────────────────────────────────────────────────────
print("Clustering in PC space...")
cluster_result = GLVClusterer().fit(proj.T)
print(f"  Selected k={cluster_result.selected_k}")
counts = np.bincount(cluster_result.labels)
for ci, cnt in enumerate(counts):
    print(f"    Cluster {ci}: {cnt} ({100*cnt/len(cluster_result.labels):.1f}%)")

# ── Build velocity field on 2D grid ──────────────────────────────────────────
print(f"\nBuilding {GRID_SIZE}×{GRID_SIZE} velocity field...")
pad1 = PAD_FRAC * (proj[0].max() - proj[0].min())
pad2 = PAD_FRAC * (proj[1].max() - proj[1].min())
x_vals = np.linspace(proj[0].min() - pad1, proj[0].max() + pad1, GRID_SIZE)
y_vals = np.linspace(proj[1].min() - pad2, proj[1].max() + pad2, GRID_SIZE)
xx, yy = np.meshgrid(x_vals, y_vals)

uu = np.zeros_like(xx)
vv = np.zeros_like(yy)
s_idx = fp.surviving

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        z1, z2 = xx[i, j], yy[i, j]
        # Reconstruct full N from PC coordinates
        dN_s = z1 * V[:, 0] + z2 * V[:, 1]       # displacement in survivor subspace
        N_full = fp.N_star.copy()
        N_full[s_idx] += dN_s
        N_full = np.clip(N_full, 0.0, None)
        # Full nonlinear RHS
        dNdt = model.lotka_volterra_rhs(N_full)
        dNdt_s = dNdt[s_idx]
        # Project onto PC1, PC2
        uu[i, j] = V[:, 0] @ dNdt_s
        vv[i, j] = V[:, 1] @ dNdt_s

speed = np.sqrt(uu**2 + vv**2)
print(f"  speed range: {speed.min():.3e} – {speed.max():.3e}")

# ── FP location in PC space (should be near origin) ──────────────────────────
# The PCA centers on the trajectory mean, so the det. FP maps to a small offset
dN_fp_s = fp.N_star[s_idx] - pca.mean        # deviation of det. FP from traj. mean
fp_z1 = float(V[:, 0] @ dN_fp_s)
fp_z2 = float(V[:, 1] @ dN_fp_s)

# ── Slowest RSV projected to PC space ────────────────────────────────────────
rsv_slow = geom.right_singular_vectors[:, 0]   # slowest direction in survivor space
rsv_z1 = float(V[:, 0] @ rsv_slow)
rsv_z2 = float(V[:, 1] @ rsv_slow)

# ── 2-panel figure ────────────────────────────────────────────────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: velocity field
plot_phase_flow(
    xx, yy, uu, vv, speed,
    axes[0],
    fp_pc=(fp_z1, fp_z2),
    rsv_pc=(rsv_z1, rsv_z2),
    sigma_min=geom.sigma_min,
)
axes[0].set_facecolor("#1a1a2e")

# Right: trajectory colored by cluster
plot_pca_trajectory(
    proj, axes[1],
    labels=cluster_result.labels,
    centroids=cluster_result.centroids,
    alpha=0.25, s=3.0,
)
# Mark FP and RSV on trajectory panel too
axes[1].scatter(fp_z1, fp_z2, s=120, marker="*", color="black",
                edgecolors="white", lw=0.7, zorder=6, label="FP")
axes[1].legend(fontsize=8)
axes[1].set_title(
    f"Trajectory in PC space  (k={cluster_result.selected_k} clusters)",
    fontsize=11,
)
axes[1].set_facecolor("#f8f8f8")

fig.suptitle(
    rf"GLV phase flow  $S$={S}, $\mu$={MU}, $\sigma$={SIGMA}  |  "
    rf"$D$={D}, $T$={T_MAX:.0f}, $dt$={DT}  |  "
    rf"$\phi$={fp.phi:.2f} ({len(fp.surviving)} surv.)  |  "
    rf"anisotropy={geom.anisotropy:.1f}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {OUT_PATH}")
