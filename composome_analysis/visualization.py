"""Shared matplotlib visualization helpers for GLV analysis.

All functions accept a pre-created ``ax`` (matplotlib Axes) so scripts
can compose them freely into multi-panel figures.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def plot_eigenvalues_complex(
    eigenvalues: NDArray,
    ax,
    title: str = "",
    sigma_lambda: float | None = None,
) -> None:
    """Scatter Jacobian eigenvalues in the complex plane.

    Parameters
    ----------
    eigenvalues : ndarray complex (S*,)
        Eigenvalues of the Jacobian restricted to surviving species.
    ax : matplotlib Axes
    title : str
    sigma_lambda : float or None
        If given, draw a dashed vertical line at ``-sigma_lambda``
        (predicted left spectral edge from cavity ``sqrt(sigma_lambda_sq)``).
    """
    re = np.real(eigenvalues)
    im = np.imag(eigenvalues)
    ax.scatter(re, im, s=20, c="#1976d2", alpha=0.85, edgecolors="none", zorder=3)
    ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.7)
    ax.axvline(0.0, color="#666666", lw=0.8, alpha=0.7)
    if sigma_lambda is not None:
        ax.axvline(
            -sigma_lambda, color="#d32f2f", lw=1.2, ls="--", alpha=0.9,
            label=rf"$-\sigma_\lambda = {-sigma_lambda:.3f}$",
        )
        ax.legend(fontsize=9)
    ax.set_xlabel(r"Re($\lambda$)", fontsize=11)
    ax.set_ylabel(r"Im($\lambda$)", fontsize=11)
    ax.set_title(title or "Jacobian eigenvalues", fontsize=11)
    ax.grid(alpha=0.2)


def plot_eigenvalue_histogram(
    eigenvalues: NDArray,
    ax,
    sigma_lambda: float | None = None,
    n_bins: int = 30,
) -> None:
    """Histogram of Re(λ) values.

    Parameters
    ----------
    eigenvalues : ndarray complex
    ax : matplotlib Axes
    sigma_lambda : float or None
        Predicted spectral width from cavity; draws a dashed vertical line.
    n_bins : int
    """
    re = np.real(eigenvalues)
    ax.hist(re, bins=n_bins, color="#1976d2", alpha=0.7, edgecolor="white", lw=0.4)
    if sigma_lambda is not None:
        ax.axvline(
            -sigma_lambda, color="#d32f2f", lw=1.5, ls="--",
            label=rf"$-\sigma_\lambda$",
        )
        ax.legend(fontsize=9)
    ax.set_xlabel(r"Re($\lambda$)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Re(eigenvalues)", fontsize=11)
    ax.grid(alpha=0.2, axis="y")


def plot_pca_trajectory(
    projections_2d: NDArray,
    ax,
    labels: NDArray | None = None,
    t: NDArray | None = None,
    cmap: str = "tab10",
    centroids: NDArray | None = None,
    alpha: float = 0.4,
    s: float = 6.0,
) -> None:
    """Scatter PC1 vs PC2, colored by cluster label or time.

    Parameters
    ----------
    projections_2d : ndarray (2, n_times)
        First two rows of ``pca.projections``.
    ax : matplotlib Axes
    labels : ndarray (n_times,) int or None
        0-indexed cluster labels. Colors by cluster when given.
    t : ndarray (n_times,) or None
        Time array. Used for coloring when labels is None.
    cmap : str
        Colormap name for cluster coloring.
    centroids : ndarray (k, 2) or None
        Cluster centroids to overlay as stars.
    alpha, s : scatter visual kwargs.
    """
    import matplotlib.pyplot as plt

    pc1 = projections_2d[0]
    pc2 = projections_2d[1]

    if labels is not None:
        cmap_obj = plt.get_cmap(cmap)
        k = int(labels.max()) + 1
        for ci in range(k):
            mask = labels == ci
            color = cmap_obj(ci / max(k - 1, 1))
            ax.scatter(
                pc1[mask], pc2[mask], s=s, alpha=alpha,
                color=color, label=f"Cluster {ci}", edgecolors="none", zorder=2,
            )
        if centroids is not None:
            for ci, ctr in enumerate(centroids):
                ax.scatter(
                    ctr[0], ctr[1], s=150, marker="*",
                    color=cmap_obj(ci / max(k - 1, 1)),
                    edgecolors="k", lw=0.7, zorder=5,
                )
        ax.legend(fontsize=8, markerscale=2)
    elif t is not None:
        sc = ax.scatter(pc1, pc2, s=s, alpha=alpha, c=t, cmap="viridis",
                        edgecolors="none", zorder=2)
        plt.colorbar(sc, ax=ax, label="Time")
    else:
        ax.scatter(pc1, pc2, s=s, alpha=alpha, color="#1976d2",
                   edgecolors="none", zorder=2)

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.grid(alpha=0.2)


def plot_silhouette_sweep(
    ks: tuple[int, ...] | NDArray,
    silhouette_by_k: NDArray,
    ax,
    selected_k: int | None = None,
) -> None:
    """Bar chart of silhouette scores vs k with optional selected-k marker.

    Parameters
    ----------
    ks : sequence of int
    silhouette_by_k : ndarray matching ks
    ax : matplotlib Axes
    selected_k : int or None
    """
    ks_arr = np.asarray(ks)
    sil_arr = np.asarray(silhouette_by_k)
    finite = np.isfinite(sil_arr)
    ax.bar(
        ks_arr[finite], sil_arr[finite],
        color="#1976d2", alpha=0.8, edgecolor="white", width=0.6,
    )
    if selected_k is not None:
        ax.axvline(selected_k, color="#d32f2f", lw=1.5, ls="--",
                   label=f"Selected k={selected_k}")
        ax.legend(fontsize=9)
    ax.set_xlabel("k (number of clusters)", fontsize=11)
    ax.set_ylabel("Silhouette score", fontsize=11)
    ax.set_title("k-sweep model selection", fontsize=11)
    ax.grid(alpha=0.2, axis="y")


def plot_phase_flow(
    xx: NDArray,
    yy: NDArray,
    uu: NDArray,
    vv: NDArray,
    speed: NDArray,
    ax,
    fp_pc: tuple[float, float] | None = None,
    rsv_pc: tuple[float, float] | None = None,
    sigma_min: float | None = None,
) -> None:
    """Quiver plot of projected deterministic velocity field in PC1-PC2 space.

    Analogous to GARD's pca_phase_flow.py quiver visualization. Arrows are
    direction-normalized with adaptive length so slow zones remain visible;
    color encodes log10(speed).

    Parameters
    ----------
    xx, yy : ndarray (grid_size, grid_size)
        Meshgrid of PC1, PC2 coordinates.
    uu, vv : ndarray (grid_size, grid_size)
        Projected velocity components (dz1/dt, dz2/dt) at each grid point.
    speed : ndarray (grid_size, grid_size)
        Flow speed = sqrt(uu²+vv²) at each grid point.
    ax : matplotlib Axes
    fp_pc : (z1, z2) or None
        FP location in PC coordinates; drawn as a white star if given.
    rsv_pc : (dz1, dz2) or None
        Slowest right singular vector projected onto PC space; drawn as
        a white dashed arrow from the origin if given.
    sigma_min : float or None
        Label value printed next to the RSV arrow.
    """
    import matplotlib.pyplot as plt

    eps = 1e-12
    norm = np.sqrt(uu * uu + vv * vv)
    udir = uu / np.maximum(norm, eps)
    vdir = vv / np.maximum(norm, eps)

    # Adaptive length: min 25% visibility in slow zones
    finite_norm = norm[np.isfinite(norm)]
    ref = float(np.quantile(finite_norm, 0.92)) if finite_norm.size > 0 else 1.0
    ref = max(ref, eps)
    rel_len = np.clip(norm / ref, 0.0, 1.0)
    rel_len = 0.25 + 0.75 * rel_len

    log_speed = np.log10(norm + eps)

    q = ax.quiver(
        xx, yy, udir * rel_len, vdir * rel_len, log_speed,
        cmap="plasma", pivot="mid", scale=18, width=0.004,
    )
    plt.colorbar(q, ax=ax, label=r"$\log_{10}$(speed)")

    if fp_pc is not None:
        ax.scatter(*fp_pc, s=120, marker="*", color="white",
                   edgecolors="k", lw=0.7, zorder=6, label="FP")

    if rsv_pc is not None:
        dz1, dz2 = rsv_pc
        scale = 0.4 * max(np.ptp(xx), np.ptp(yy))
        mag = np.sqrt(dz1**2 + dz2**2)
        if mag > 0:
            dz1, dz2 = dz1 / mag * scale, dz2 / mag * scale
        label = (f"slow RSV  σ={sigma_min:.3e}" if sigma_min is not None
                 else "slow RSV")
        ax.annotate("", xy=(dz1, dz2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="white",
                                   lw=1.5, linestyle="dashed"))
        ax.text(dz1 * 1.1, dz2 * 1.1, label, color="white",
                fontsize=8, ha="center")

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("Velocity field in PC space", fontsize=11)
    ax.grid(alpha=0.15)


def plot_species_traces(
    t: NDArray,
    N: NDArray,
    ax,
    surviving: NDArray | None = None,
    n_show: int = 5,
    fp_N_star: NDArray | None = None,
    seed: int = 0,
) -> None:
    """Time series of a few randomly chosen surviving species.

    Parameters
    ----------
    t : ndarray (n_times,)
    N : ndarray (S, n_times)
    ax : matplotlib Axes
    surviving : ndarray int or None
        Indices of surviving species. If None uses all.
    n_show : int
        How many species to plot.
    fp_N_star : ndarray (S,) or None
        Fixed-point abundances; drawn as dashed horizontal lines.
    seed : int
        For reproducible random species selection.
    """
    rng = np.random.default_rng(seed)
    indices = surviving if surviving is not None else np.arange(N.shape[0])
    chosen = rng.choice(indices, size=min(n_show, len(indices)), replace=False)

    palette = ["#1976d2", "#d32f2f", "#388e3c", "#f57c00", "#7b1fa2"]
    for i, sp in enumerate(chosen):
        col = palette[i % len(palette)]
        ax.plot(t, N[sp], lw=0.9, alpha=0.85, color=col, label=f"sp {sp}")
        if fp_N_star is not None:
            ax.axhline(fp_N_star[sp], color=col, lw=0.8, ls="--", alpha=0.55)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Abundance N", fontsize=11)
    ax.set_title(f"Species trajectories ({n_show} survivors shown)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
