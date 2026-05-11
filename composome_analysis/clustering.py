"""K-means clustering of GLV stochastic trajectories in PC space.

Mirrors the GARD ComposomeAnalyzer pattern: k-sweep with silhouette-based
model selection and patience-based early stopping. Pure numpy, no sklearn.

The H-similarity / drift-marking logic from GARD is intentionally omitted:
GLV trajectories are continuous-time (no generation boundaries) and live
near a single fixed point in the SFP regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Metric = Literal["sqeuclidean", "cosine"]


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for GLV trajectory clustering."""

    ks: tuple[int, ...] = tuple(range(1, 11))
    metric: Metric = "sqeuclidean"
    replicas: int = 10          # k-means restarts per k value
    mink: int = 4               # patience: stop after this many non-improving k values
    random_seed: int | None = 0
    max_iter: int = 300
    tol: float = 1e-6


@dataclass(frozen=True)
class ClusterResult:
    """Output of GLV clustering."""

    labels: NDArray[np.int64]               # shape (n_points,), 0-indexed
    centroids: NDArray[np.float64]          # shape (k, n_features)
    selected_k: int
    silhouette_by_k: NDArray[np.float64]    # NaN for untried/invalid k values


class GLVClusterer:
    """K-means clustering with silhouette-based model selection.

    Typical usage::

        clusterer = GLVClusterer()
        result = clusterer.fit(pca.projections[:2, :].T)  # (n_times, 2)
        labels = result.labels
        selected_k = result.selected_k
    """

    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = ClusterConfig() if config is None else config

    # ── internal geometry helpers ──────────────────────────────────────────

    @staticmethod
    def _normalize_rows(x: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
        norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        return x / np.maximum(norms, eps)

    @staticmethod
    def _pairwise_distances(x: NDArray[np.float64], metric: Metric) -> NDArray[np.float64]:
        if metric == "sqeuclidean":
            x2 = np.sum(x * x, axis=1, keepdims=True)
            d = x2 + x2.T - 2.0 * (x @ x.T)
            return np.maximum(d, 0.0)
        x_n = GLVClusterer._normalize_rows(x)
        sim = x_n @ x_n.T
        return 1.0 - np.clip(sim, -1.0, 1.0)

    @staticmethod
    def _silhouette_mean(
        x: NDArray[np.float64], labels: NDArray[np.int64], metric: Metric
    ) -> float:
        n = x.shape[0]
        if n <= 1:
            return float("nan")
        uniq = np.unique(labels)
        if uniq.size <= 1:
            return float("nan")
        dmat = GLVClusterer._pairwise_distances(x, metric)
        s = np.zeros(n, dtype=float)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            a_i = float(np.mean(dmat[i, same])) if np.any(same) else 0.0
            b_i = np.inf
            for c in uniq:
                if c == labels[i]:
                    continue
                other = labels == c
                if np.any(other):
                    b_i = min(b_i, float(np.mean(dmat[i, other])))
            denom = max(a_i, b_i) if np.isfinite(b_i) else 0.0
            s[i] = 0.0 if denom <= 0.0 else (b_i - a_i) / denom
        return float(np.mean(s))

    # ── k-means core ──────────────────────────────────────────────────────

    def _kmeans_single(
        self,
        x: NDArray[np.float64],
        k: int,
        metric: Metric,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
        n, _ = x.shape
        if k == 1:
            c = np.mean(x, axis=0, keepdims=True)
            if metric == "cosine":
                c = self._normalize_rows(c)
            labels = np.zeros(n, dtype=np.int64)
            if metric == "sqeuclidean":
                dist = np.sum((x - c[0]) ** 2, axis=1)
            else:
                x_n = self._normalize_rows(x)
                dist = 1.0 - np.clip(np.sum(x_n * c[0], axis=1), -1.0, 1.0)
            return labels, c, float(np.sum(dist))

        cent_idx = rng.choice(n, size=k, replace=False)
        c = x[cent_idx].copy()
        x_work = self._normalize_rows(x) if metric == "cosine" else x
        if metric == "cosine":
            c = self._normalize_rows(c)

        labels = np.full(n, -1, dtype=np.int64)
        prev_obj = np.inf
        for _ in range(self.config.max_iter):
            if metric == "sqeuclidean":
                d2 = np.sum((x_work[:, None, :] - c[None, :, :]) ** 2, axis=2)
                new_labels = np.argmin(d2, axis=1).astype(np.int64)
                objective = float(np.sum(d2[np.arange(n), new_labels]))
            else:
                sim = x_work @ c.T
                new_labels = np.argmax(sim, axis=1).astype(np.int64)
                objective = float(np.sum(1.0 - sim[np.arange(n), new_labels]))

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for j in range(k):
                idx = np.where(labels == j)[0]
                c[j] = (
                    x_work[rng.integers(0, n)]
                    if idx.size == 0
                    else np.mean(x_work[idx], axis=0)
                )
                if metric == "cosine":
                    c[j] = self._normalize_rows(c[j][None, :])[0]

            if abs(prev_obj - objective) < self.config.tol:
                break
            prev_obj = objective

        return labels, c, objective

    def _kmeans_replicates(
        self,
        x: NDArray[np.float64],
        k: int,
        metric: Metric,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
        best_labels: NDArray[np.int64] | None = None
        best_centroids: NDArray[np.float64] | None = None
        best_obj = np.inf
        for _ in range(self.config.replicas):
            lbl, ctr, obj = self._kmeans_single(x, k, metric, rng)
            if obj < best_obj:
                best_obj = obj
                best_labels = lbl
                best_centroids = ctr

        assert best_labels is not None and best_centroids is not None

        if k == 1:
            d = self._pairwise_distances(x, metric)
            sil = float(-np.mean(d))
        else:
            sil = self._silhouette_mean(x, best_labels, metric)

        return best_labels, best_centroids, sil

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, points: NDArray[np.float64]) -> ClusterResult:
        """Cluster trajectory points with k-sweep and silhouette model selection.

        Parameters
        ----------
        points : ndarray (n_points, n_features)
            Typically 2D PC projections: ``pca.projections[:2, :].T``

        Returns
        -------
        ClusterResult
        """
        x = np.asarray(points, dtype=float)
        if x.ndim != 2:
            raise ValueError("points must be 2D with shape (n_points, n_features).")
        n = x.shape[0]
        metric = self.config.metric
        rng = np.random.default_rng(self.config.random_seed)

        sil_by_k = np.full(len(self.config.ks), np.nan, dtype=float)
        labels_by_k: list[NDArray[np.int64] | None] = [None] * len(self.config.ks)
        centroids_by_k: list[NDArray[np.float64] | None] = [None] * len(self.config.ks)

        best_sil = -np.inf
        streak = 0

        for idx_k, k in enumerate(self.config.ks):
            if k > n:
                break
            lbl, ctr, sil = self._kmeans_replicates(x, k, metric, rng)
            sil_by_k[idx_k] = sil
            labels_by_k[idx_k] = lbl
            centroids_by_k[idx_k] = ctr

            if np.isfinite(sil) and sil >= best_sil:
                best_sil = sil
                streak = 0
            else:
                streak += 1

            if streak >= self.config.mink:
                break

        finite = np.where(np.isfinite(sil_by_k))[0]
        selected_idx = int(finite[np.argmax(sil_by_k[finite])]) if finite.size > 0 else 0
        selected_k = int(self.config.ks[selected_idx])
        labels = labels_by_k[selected_idx]
        centroids = centroids_by_k[selected_idx]

        if labels is None:
            labels = np.zeros(n, dtype=np.int64)
            centroids = np.mean(x, axis=0, keepdims=True)
            selected_k = 1

        return ClusterResult(
            labels=labels.astype(np.int64),
            centroids=np.asarray(centroids, dtype=float),
            selected_k=selected_k,
            silhouette_by_k=sil_by_k,
        )
