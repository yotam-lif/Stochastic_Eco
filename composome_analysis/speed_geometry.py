"""SVD-based speed geometry of the GLV Jacobian at a fixed point.

The local flow speed when displaced from the FP in direction v is |J·v|.
The singular values of J (not its eigenvalues) set this speed in each
direction. Small singular values → slow zones where the system spends
more time. This is the GLV analog of GARD's analyze_local_speed_geometry_c().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpeedGeometry:
    """SVD-based speed geometry of J_reduced at the fixed point.

    Attributes
    ----------
    singular_values : ndarray (S*,) ascending
        Sorted smallest-first. σ_k = speed in direction of right_singular_vectors[:,k].
    right_singular_vectors : ndarray (S*, S*)
        Columns are right singular vectors in surviving-species space.
        Column 0 is the slowest direction.
    sigma_min : float
    sigma_max : float
    anisotropy : float
        sigma_max / sigma_min. Large anisotropy → strong slow zones.
    slow_mode_count : int
        Number of modes with σ_k ≤ slow_threshold × sigma_max.
    slow_threshold : float
        Relative threshold used to count slow modes.
    """

    singular_values: NDArray[np.float64]
    right_singular_vectors: NDArray[np.float64]
    sigma_min: float
    sigma_max: float
    anisotropy: float
    slow_mode_count: int
    slow_threshold: float


def analyze_speed_geometry(
    model,
    fp,
    slow_threshold: float = 0.1,
) -> SpeedGeometry:
    """Compute SVD of J_reduced and return speed geometry.

    Parameters
    ----------
    model : GLVModel
    fp : FixedPoint
    slow_threshold : float
        Modes with σ_k ≤ slow_threshold × σ_max are counted as slow.

    Returns
    -------
    SpeedGeometry
    """
    s = fp.surviving
    J = model.jacobian(fp.N_star)[np.ix_(s, s)]

    # SVD: J = U Σ Vᵀ  →  right singular vectors are rows of Vᵀ (columns of V)
    _, singular_desc, Vht = np.linalg.svd(J, full_matrices=False)

    # Sort ascending: slowest mode first
    order = np.argsort(singular_desc)
    singular = singular_desc[order].astype(float)
    rsv = Vht.T[:, order].astype(float)   # columns = right singular vectors

    sigma_min = float(singular[0]) if singular.size > 0 else 0.0
    sigma_max = float(singular[-1]) if singular.size > 0 else 0.0
    anisotropy = (sigma_max / max(sigma_min, np.finfo(float).tiny)
                  if sigma_max > 0 else float("inf"))
    slow_cutoff = slow_threshold * sigma_max
    slow_count = int(np.sum(singular <= slow_cutoff))

    return SpeedGeometry(
        singular_values=singular,
        right_singular_vectors=rsv,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        anisotropy=anisotropy,
        slow_mode_count=slow_count,
        slow_threshold=slow_threshold,
    )


def print_speed_geometry(geom: SpeedGeometry, n_show: int = 10) -> None:
    """Print singular value summary to stdout.

    Parameters
    ----------
    geom : SpeedGeometry
    n_show : int
        Number of slowest modes to list individually.
    """
    n = len(geom.singular_values)
    print(f"Speed geometry  (S*={n} survivors)")
    print(f"  σ_min={geom.sigma_min:.4e}  σ_max={geom.sigma_max:.4e}"
          f"  anisotropy={geom.anisotropy:.2f}")
    print(f"  Slow modes (σ ≤ {geom.slow_threshold}×σ_max={geom.slow_threshold*geom.sigma_max:.4e}):"
          f"  {geom.slow_mode_count}/{n}")
    print(f"\n  {'mode':>5}  {'σ_k':>12}  {'σ_k/σ_max':>10}")
    print("  " + "-" * 32)
    for i in range(min(n_show, n)):
        ratio = geom.singular_values[i] / max(geom.sigma_max, 1e-300)
        print(f"  {i+1:5d}  {geom.singular_values[i]:12.4e}  {ratio:10.4f}")
