"""Composome-style analysis of a GLV stochastic trajectory.

Direct analog of GARD/code/composome_history_lineage.py for the GLV system:

- SDE trajectory starting from the deterministic fixed point.
- H-similarity carpet: cosine similarity between consecutive composition snapshots.
- Non-drift/drift tagging with optional min_dwell run-length filter
  (min_dwell is the analog of GARD's drift_size — requires N consecutive
  above-threshold snapshots before calling a segment a real composome visit).
- k-sweep clustering of non-drift compositions → compotypes.
- 4-panel summary figure, 2D CLR-PCA, interactive 3D HTML.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from base_structure import GLVModel
from composome_analysis import flow_to_fixed_point, GLVClusterer, ClusterConfig


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrajectoryComposomeConfig:
    """All parameters for the composome trajectory analysis."""

    # Model
    S: int = 200
    mu: float = 4 * 10 ** 0
    sigma: float = 0.8
    gamma: float = 0.0
    sigma_K: float = 0.0
    seed: int = 41
    # SDE
    D: float = 1 * 10 ** -1
    t_max: float = 5 * 10 ** 4
    dt: float = 0.005
    save_every: int = 10
    noise_type: str = "demographic"
    # H-similarity + non-drift marking
    # h_threshold: cosine similarity threshold between consecutive snapshots
    # min_dwell:   analog of GARD drift_size — minimum run length of consecutive
    #              above-threshold snapshots required to call a segment non-drift.
    #              None = pointwise test (each snapshot judged independently).
    h_threshold: float = 0.9
    min_dwell: int | None = None
    # Clustering
    ks: tuple[int, ...] = tuple(range(1, 11))
    metric: str = "sqeuclidean"
    replicas: int = 10
    mink: int = 4
    cluster_seed: int = 0
    cluster_subsample: int = 2000   # max non-drift points used in k-means fit
    n_pca_cluster: int = 5          # CLR-PCA dims used for clustering
    clr_pseudocount: float = 0.5
    # Visualization
    h_subsample: int = 500          # downsample trajectory for H-carpet imshow
    # Output — anchored to the project root (one level above this script file)
    output: Path = Path(__file__).resolve().parent.parent / "figs" / "composome_trajectory.png"


@dataclass(frozen=True)
class ComposomeResult:
    """Output of the composome analysis pipeline."""

    tags: np.ndarray            # (n_times,) int: 0=drift, 1..k=cluster tag
    nondrift_mask: np.ndarray   # (n_times,) bool
    h_values: np.ndarray        # (n_times,) cosine similarity H(t, t+1), boundary-padded
    comps: np.ndarray           # (S, k) mean normalized composition per cluster
    selected_k: int
    silhouette_by_k: np.ndarray
    centroids: np.ndarray       # (k, n_pca_cluster) PCA-space centroids


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_trajectory(
    config: TrajectoryComposomeConfig,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Find fixed point and run SDE. Returns (t, N, fp)."""
    model = GLVModel(
        S=config.S, mu=config.mu, sigma=config.sigma,
        gamma=config.gamma, sigma_K=config.sigma_K, seed=config.seed,
    )
    print(f"  u = {model.u:.4f}")
    fp = flow_to_fixed_point(model)
    print(f"  FP: converged={fp.converged}, phi={fp.phi:.3f}, S*={len(fp.surviving)}")
    sde = model.integrate_sde(
        N0=fp.N_star,
        t_span=(0.0, config.t_max),
        dt=config.dt,
        D=config.D,
        noise_type=config.noise_type,
        save_every=config.save_every,
    )
    return sde.t, sde.N, fp


# ── Core analysis ──────────────────────────────────────────────────────────────

def _normalize_compositions(N: np.ndarray) -> np.ndarray:
    """Normalize each time snapshot to sum to 1 (place on simplex)."""
    totals = np.sum(N, axis=0, keepdims=True)
    return N / np.maximum(totals, 1e-12)


def _clr_transform(samples: np.ndarray, pseudocount: float) -> np.ndarray:
    """CLR transform applied row-wise: log(x+eps) - mean(log(x+eps))."""
    x = np.asarray(samples, dtype=float) + pseudocount
    lx = np.log(x)
    return lx - np.mean(lx, axis=1, keepdims=True)


def _compute_clr_pca(
    samples: np.ndarray, pseudocount: float, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """CLR-PCA of composition samples (rows = snapshots, cols = species).

    Returns (proj, explained_ratios), both length n_components.
    """
    x_clr = _clr_transform(samples, pseudocount)
    centered = x_clr - np.mean(x_clr, axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return (
            np.zeros((centered.shape[0], n_components), dtype=float),
            np.zeros(n_components, dtype=float),
        )
    cov = np.atleast_2d(np.cov(centered, rowvar=False, bias=False))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    total_var = float(np.sum(evals))
    m = min(n_components, evecs.shape[1])
    ratios = np.zeros(n_components, dtype=float)
    if total_var > 0.0:
        ratios[:m] = evals[:m] / total_var
    proj = centered @ evecs[:, :m]
    if m < n_components:
        proj = np.hstack((proj, np.zeros((proj.shape[0], n_components - m), dtype=float)))
    return proj, ratios


def mark_nondrift(
    N: np.ndarray,
    h_threshold: float,
    min_dwell: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark non-drift timepoints using cosine H-similarity between snapshots.

    Mirrors GARD's ComposomeAnalyzer.mark_nondrift exactly:

    - min_dwell=None: each snapshot tested pointwise via symmetric local average.
    - min_dwell=N:    run-length filter — only runs of ≥ N consecutive
                      above-threshold snapshots are called non-drift.

    Parameters
    ----------
    N : (S, n_times)
    h_threshold : float in [0, 1]
    min_dwell : int or None

    Returns
    -------
    nondrift_mask : bool (n_times,)
    h_values      : float (n_times,)  H(t, t+1), boundary-extended at t=0
    """
    comp = _normalize_compositions(N)
    col_norms = np.sqrt(np.sum(comp * comp, axis=0, keepdims=True))
    comp_unit = comp / np.maximum(col_norms, 1e-12)

    # H(t, t+1) for t = 0 .. n_times-2
    h = np.clip(np.sum(comp_unit[:, :-1] * comp_unit[:, 1:], axis=0), 0.0, 1.0)

    # Extend to length n_times: h_full[t] ≈ H(t, t+1), padded at t=0
    h_full = np.concatenate(([h[0]], h))

    if min_dwell is None:
        # Pointwise: symmetric local average (GARD drift_size=None branch)
        h_next = np.concatenate((h, [h[-1]]))
        local = 0.5 * (h_full + h_next)
        mask = local > h_threshold
    else:
        # Run-length filter (GARD drift_size=N branch)
        a = h_full > h_threshold
        b0 = np.concatenate(([False], a[:-1]))
        b2 = np.concatenate((a[1:], [False]))
        starts = np.where((~b0) & a)[0]
        ends = np.where(a & (~b2))[0]
        mask = np.zeros(len(a), dtype=bool)
        if starts.size > 0 and ends.size > 0:
            run_len = ends - starts + 1
            for idx in np.where(run_len >= min_dwell)[0]:
                mask[starts[idx] : ends[idx] + 1] = True

    return mask, h_full


def _assign_nearest_centroid(
    points: np.ndarray, centroids: np.ndarray, metric: str
) -> np.ndarray:
    """0-indexed nearest-centroid labels for every point."""
    if metric == "sqeuclidean":
        d2 = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)
    # cosine
    pn = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    cn = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return np.argmax(pn @ cn.T, axis=1).astype(np.int64)


def fit_composomes(
    N: np.ndarray,
    config: TrajectoryComposomeConfig,
) -> tuple[ComposomeResult, np.ndarray, np.ndarray]:
    """Full composome pipeline.

    Returns
    -------
    result    : ComposomeResult
    proj_all  : (n_times, n_pca) CLR-PCA projections for all snapshots
    var_ratios: (n_pca,) explained variance ratios
    """
    n_times = N.shape[1]
    # Always compute at least 3 PCA components so the 3D plot always works.
    n_pca = max(3, config.n_pca_cluster)

    # 1. Mark non-drift timepoints.
    nondrift_mask, h_values = mark_nondrift(N, config.h_threshold, config.min_dwell)
    nd_idx = np.where(nondrift_mask)[0]
    n_non = len(nd_idx)

    # 2. CLR-PCA on normalized compositions of all snapshots.
    comp_norm = _normalize_compositions(N)
    proj_all, var_ratios = _compute_clr_pca(comp_norm.T, config.clr_pseudocount, n_pca)

    tags = np.zeros(n_times, dtype=np.int64)
    comps = np.empty((N.shape[0], 0), dtype=float)
    selected_k = 1
    sil_by_k = np.full(len(config.ks), np.nan, dtype=float)
    centroids_pca = np.zeros((1, config.n_pca_cluster), dtype=float)

    if n_non > 0:
        # 3. Optionally subsample non-drift points to keep k-means tractable.
        proj_nd = proj_all[nd_idx, : config.n_pca_cluster]  # (n_non, n_pca_cluster)
        rng = np.random.default_rng(config.cluster_seed)
        if n_non > config.cluster_subsample:
            sub = np.sort(rng.choice(n_non, size=config.cluster_subsample, replace=False))
            fit_pts = proj_nd[sub]
        else:
            fit_pts = proj_nd

        # 4. k-means sweep with silhouette model selection.
        cluster_cfg = ClusterConfig(
            ks=config.ks,
            metric=config.metric,
            replicas=config.replicas,
            mink=config.mink,
            random_seed=config.cluster_seed,
        )
        cluster_result = GLVClusterer(cluster_cfg).fit(fit_pts)
        selected_k = cluster_result.selected_k
        sil_by_k = cluster_result.silhouette_by_k
        centroids_pca = cluster_result.centroids  # (k, n_pca_cluster)

        # 5. Assign ALL non-drift points by nearest centroid (not just the subsample).
        nd_labels = _assign_nearest_centroid(proj_nd, centroids_pca, config.metric)
        tags[nd_idx] = nd_labels + 1   # 1..k; 0 stays for drift

        # 6. Compotype centroid in composition space: mean normalized abundance per cluster.
        k = selected_k
        comps_out = np.zeros((N.shape[0], k), dtype=float)
        for ci in range(k):
            cluster_mask = tags == (ci + 1)
            if np.any(cluster_mask):
                comps_out[:, ci] = np.mean(comp_norm[:, cluster_mask], axis=1)
        comps = comps_out

    return (
        ComposomeResult(
            tags=tags,
            nondrift_mask=nondrift_mask,
            h_values=h_values,
            comps=comps,
            selected_k=selected_k,
            silhouette_by_k=sil_by_k,
            centroids=centroids_pca,
        ),
        proj_all,
        var_ratios,
    )


def _compute_h_matrix(N: np.ndarray, h_subsample: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise H-similarity matrix on a uniformly subsampled trajectory.

    Returns (h_mat, sub_idx).
    """
    n_times = N.shape[1]
    if h_subsample > 0 and n_times > h_subsample:
        sub_idx = np.round(np.linspace(0, n_times - 1, h_subsample)).astype(int)
    else:
        sub_idx = np.arange(n_times)
    comp = _normalize_compositions(N[:, sub_idx])
    col_norms = np.sqrt(np.sum(comp * comp, axis=0, keepdims=True))
    comp_unit = comp / np.maximum(col_norms, 1e-12)
    h_mat = np.clip(comp_unit.T @ comp_unit, 0.0, 1.0)
    return h_mat, sub_idx


# ── Plotting ───────────────────────────────────────────────────────────────────

def _relative_comps(comps: np.ndarray) -> np.ndarray:
    """Column-normalize compotype matrix for relative-composition display."""
    if comps.size == 0:
        return comps
    denom = np.maximum(np.sum(comps, axis=0, keepdims=True), 1e-12)
    return comps / denom


def _default_pca_output_path(p: Path) -> Path:
    return p.with_name(f"{p.stem}_nondrift_pca2d{p.suffix}")


def _default_pca3d_output_path(p: Path) -> Path:
    return p.with_name(f"{p.stem}_nondrift_pca3d_interactive.html")


def plot_composome_trajectory(
    t: np.ndarray,
    N: np.ndarray,
    result: ComposomeResult,
    ks: tuple[int, ...],
    config: TrajectoryComposomeConfig,
    output_path: Path,
) -> None:
    """4-panel composome summary figure.

    Panels:
    1) H-similarity carpet (subsampled time × time).
    2) Tag timeline: gray=drift, colored=non-drift cluster.
    3) k-sweep silhouette with selected-k marker.
    4) Compotype composition heatmap (active species × cluster).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    h_mat, sub_idx = _compute_h_matrix(N, config.h_subsample)
    t_sub = t[sub_idx]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # Panel 1: H-similarity carpet
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        h_mat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
        extent=[float(t_sub[0]), float(t_sub[-1]), float(t_sub[0]), float(t_sub[-1])],
    )
    ax0.set_title("Similarity Carpet (H)")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Time")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="H")

    # Panel 2: tag timeline
    ax1 = fig.add_subplot(gs[0, 1])
    nondrift = result.nondrift_mask
    drift = ~nondrift
    if np.any(drift):
        ax1.scatter(
            t[drift], np.zeros(int(np.sum(drift))),
            s=6, c="#9e9e9e", alpha=0.7, label="drift (tag=0)", rasterized=True,
        )
    if np.any(nondrift):
        ax1.scatter(
            t[nondrift], result.tags[nondrift],
            s=7, c=result.tags[nondrift], cmap="tab20", alpha=0.9,
            label="non-drift tags", rasterized=True,
        )
    ax1.set_title("Timepoint Tags")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Compotype tag")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel 3: k-sweep silhouette
    ax2 = fig.add_subplot(gs[1, 0])
    ks_arr = np.asarray(ks, dtype=int)
    sil = result.silhouette_by_k
    valid = np.isfinite(sil)
    if np.any(valid):
        ax2.plot(ks_arr[valid], sil[valid], "o-", lw=1.5, ms=4)
    ax2.axvline(
        result.selected_k, color="black", ls="--", lw=1.0, alpha=0.7,
        label=f"k={result.selected_k}",
    )
    ax2.set_title("k-Sweep Silhouette")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Mean silhouette")
    ax2.legend(frameon=False, fontsize=8)

    # Panel 4: compotype composition heatmap (active species only)
    ax3 = fig.add_subplot(gs[1, 1])
    comps_rel = _relative_comps(result.comps)
    if comps_rel.size == 0:
        ax3.text(0.5, 0.5, "No compotypes found", ha="center", va="center")
        ax3.set_axis_off()
    else:
        # Drop species that are near-zero across all clusters
        row_max = np.max(comps_rel, axis=1)
        active = row_max > 1e-4
        display = comps_rel[active, :]
        im3 = ax3.imshow(display, origin="lower", aspect="auto", cmap="magma")
        ax3.set_title("Compotypes (relative composition)")
        ax3.set_xlabel("Compotype index")
        ax3.set_ylabel(f"Species index ({int(np.sum(active))} active)")
        ax3.set_xticks(np.arange(display.shape[1]))
        ax3.set_xticklabels([str(i) for i in range(1, display.shape[1] + 1)])
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="fraction")

    drift_frac = float(np.mean(result.tags == 0))
    fig.suptitle(
        rf"GLV Composome Trajectory | $S$={config.S}, $\mu$={config.mu},"
        rf" $\sigma$={config.sigma}, $D$={config.D}"
        f"\nsnapshots={N.shape[1]} | t=[{t[0]:.3g}, {t[-1]:.3g}]"
        f" | selected k={result.selected_k} | drift frac={drift_frac:.3f}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nondrift_pca2d(
    result: ComposomeResult,
    proj_all: np.ndarray,
    var_ratios: np.ndarray,
    output_path: Path,
) -> None:
    """2D CLR-PCA scatter of non-drift timepoints, colored by cluster tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    nondrift = result.nondrift_mask
    x1 = proj_all[nondrift, 0]
    x2 = proj_all[nondrift, 1]
    tags = result.tags[nondrift]

    fig, ax = plt.subplots(figsize=(9, 7))
    if x1.size == 0:
        ax.text(0.5, 0.5, "No non-drift samples found", ha="center", va="center")
        ax.set_axis_off()
    else:
        unique_tags = np.unique(tags)
        cmap = plt.get_cmap("tab20", max(2, int(np.max(unique_tags)) + 1))
        for tag in unique_tags:
            m = tags == tag
            ax.scatter(
                x1[m], x2[m], s=10, alpha=0.7, color=cmap(int(tag)),
                label=f"cluster {int(tag)}", rasterized=True,
            )
        ax.set_xlabel("CLR-PC1")
        ax.set_ylabel("CLR-PC2")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        "Non-drift timepoints in 2D CLR-PCA space\n"
        f"PC1+PC2: {(var_ratios[0]+var_ratios[1])*100:.2f}%  "
        f"(PC1={var_ratios[0]*100:.2f}%, PC2={var_ratios[1]*100:.2f}%)",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nondrift_pca3d_interactive(
    result: ComposomeResult,
    proj_all: np.ndarray,
    var_ratios: np.ndarray,
    output_path: Path,
) -> None:
    """Interactive 3D CLR-PCA scatter as a standalone HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nondrift = result.nondrift_mask
    x1 = proj_all[nondrift, 0]
    x2 = proj_all[nondrift, 1]
    x3 = proj_all[nondrift, 2]
    tags = result.tags[nondrift]

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    traces: list[dict] = []
    if x1.size > 0:
        for i, tag in enumerate(np.unique(tags)):
            m = tags == tag
            traces.append({
                "type": "scatter3d",
                "mode": "markers",
                "name": f"cluster {int(tag)}",
                "x": x1[m].tolist(),
                "y": x2[m].tolist(),
                "z": x3[m].tolist(),
                "marker": {"size": 3, "opacity": 0.8, "color": palette[i % len(palette)]},
            })

    captured = float(np.sum(var_ratios[:3])) * 100.0
    layout: dict = {
        "title": (
            "Non-drift timepoints in interactive 3D CLR-PCA space<br>"
            f"PC1+PC2+PC3: {captured:.2f}%  "
            f"(PC1={var_ratios[0]*100:.2f}%, "
            f"PC2={var_ratios[1]*100:.2f}%, "
            f"PC3={var_ratios[2]*100:.2f}%)"
        ),
        "scene": {
            "xaxis": {"title": "CLR-PC1"},
            "yaxis": {"title": "CLR-PC2"},
            "zaxis": {"title": "CLR-PC3"},
        },
        "legend": {"orientation": "h"},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 90},
    }
    if not traces:
        layout["annotations"] = [{
            "text": "No non-drift samples found",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "font": {"size": 16},
        }]

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <title>GLV non-drift 3D CLR-PCA</title>\n"
        "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n"
        "</head>\n"
        "<body>\n"
        "  <div id=\"plot\" style=\"width: 100%; height: 95vh;\"></div>\n"
        "  <script>\n"
        f"    const data = {json.dumps(traces)};\n"
        f"    const layout = {json.dumps(layout)};\n"
        "    Plotly.newPlot('plot', data, layout, {responsive: true});\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    output_path.write_text(html, encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_ks(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("ks list is empty.")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("All ks values must be positive.")
    return vals


def build_parser(defaults: TrajectoryComposomeConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "GLV composome-style trajectory analysis. "
            "Analog of GARD composome_history_lineage.py."
        )
    )
    p.add_argument("--S", type=int, default=defaults.S)
    p.add_argument("--mu", type=float, default=defaults.mu)
    p.add_argument("--sigma", type=float, default=defaults.sigma)
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--sigma-K", type=float, default=defaults.sigma_K)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--D", type=float, default=defaults.D)
    p.add_argument("--t-max", type=float, default=defaults.t_max)
    p.add_argument("--dt", type=float, default=defaults.dt)
    p.add_argument("--save-every", type=int, default=defaults.save_every)
    p.add_argument("--noise-type", choices=["demographic", "additive"], default=defaults.noise_type)
    p.add_argument("--h-threshold", type=float, default=defaults.h_threshold)
    p.add_argument("--min-dwell", type=int, default=defaults.min_dwell)
    p.add_argument("--ks", type=str, default=",".join(str(k) for k in defaults.ks))
    p.add_argument("--metric", choices=["sqeuclidean", "cosine"], default=defaults.metric)
    p.add_argument("--replicas", type=int, default=defaults.replicas)
    p.add_argument("--mink", type=int, default=defaults.mink)
    p.add_argument("--cluster-seed", type=int, default=defaults.cluster_seed)
    p.add_argument("--cluster-subsample", type=int, default=defaults.cluster_subsample)
    p.add_argument("--n-pca-cluster", type=int, default=defaults.n_pca_cluster)
    p.add_argument("--clr-pseudocount", type=float, default=defaults.clr_pseudocount)
    p.add_argument("--h-subsample", type=int, default=defaults.h_subsample)
    p.add_argument("--output", type=Path, default=defaults.output)
    p.add_argument("--pca-output", type=Path, default=None)
    p.add_argument("--pca3d-output", type=Path, default=None)
    return p


def main() -> None:
    """Run trajectory, fit composomes, save all figures."""
    defaults = TrajectoryComposomeConfig()
    parser = build_parser(defaults)
    args = parser.parse_args()

    try:
        ks = _parse_ks(args.ks)
    except ValueError as exc:
        parser.error(f"--ks invalid: {exc}")
        return

    config = TrajectoryComposomeConfig(
        S=args.S,
        mu=args.mu,
        sigma=args.sigma,
        gamma=args.gamma,
        sigma_K=args.sigma_K,
        seed=args.seed,
        D=args.D,
        t_max=args.t_max,
        dt=args.dt,
        save_every=args.save_every,
        noise_type=args.noise_type,
        h_threshold=args.h_threshold,
        min_dwell=args.min_dwell,
        ks=ks,
        metric=args.metric,
        replicas=args.replicas,
        mink=args.mink,
        cluster_seed=args.cluster_seed,
        cluster_subsample=args.cluster_subsample,
        n_pca_cluster=args.n_pca_cluster,
        clr_pseudocount=args.clr_pseudocount,
        h_subsample=args.h_subsample,
        output=args.output,
    )

    if config.S <= 1:
        parser.error("--S must be > 1.")
    if config.t_max <= 0.0:
        parser.error("--t-max must be positive.")
    if config.dt <= 0.0:
        parser.error("--dt must be positive.")
    if config.save_every <= 0:
        parser.error("--save-every must be positive.")
    if config.D < 0.0:
        parser.error("--D must be non-negative.")
    if not 0.0 <= config.h_threshold <= 1.0:
        parser.error("--h-threshold must be in [0, 1].")
    if config.min_dwell is not None and config.min_dwell <= 0:
        parser.error("--min-dwell must be positive when provided.")
    if config.replicas <= 0:
        parser.error("--replicas must be positive.")
    if config.mink <= 0:
        parser.error("--mink must be positive.")
    if config.clr_pseudocount <= 0.0:
        parser.error("--clr-pseudocount must be positive.")
    if config.cluster_subsample <= 0:
        parser.error("--cluster-subsample must be positive.")

    print(f"GLVModel  S={config.S}, mu={config.mu}, sigma={config.sigma}, "
          f"gamma={config.gamma}, seed={config.seed}")
    t, N, fp = run_trajectory(config)
    print(f"  Trajectory shape: {N.shape}  (S × n_times)")

    print("Fitting composomes...")
    result, proj_all, var_ratios = fit_composomes(N, config)

    drift_frac = float(np.mean(result.tags == 0))
    max_tag = int(np.max(result.tags)) if result.tags.size else 0
    hist = np.bincount(result.tags, minlength=max_tag + 1).tolist()

    print(f"  Non-drift fraction: {1.0 - drift_frac:.4f}  (drift: {drift_frac:.4f})")
    print(f"  Selected k: {result.selected_k}")
    print(f"  Compotypes found: {result.comps.shape[1] if result.comps.ndim == 2 else 0}")
    print(f"  Tag histogram (0=drift, 1..k=cluster): {hist}")
    print(f"  CLR-PCA variance (first 3 PCs): "
          f"{[f'{r*100:.1f}%' for r in var_ratios[:3]]}")

    pca_out = args.pca_output or _default_pca_output_path(config.output)
    pca3d_out = args.pca3d_output or _default_pca3d_output_path(config.output)

    print("Saving figures...")
    plot_composome_trajectory(t, N, result, config.ks, config, config.output)
    plot_nondrift_pca2d(result, proj_all, var_ratios, pca_out)
    plot_nondrift_pca3d_interactive(result, proj_all, var_ratios, pca3d_out)

    print(f"  Main figure   → {config.output.resolve()}")
    print(f"  2D PCA figure → {pca_out.resolve()}")
    print(f"  3D HTML       → {pca3d_out.resolve()}")


if __name__ == "__main__":
    main()