"""Composome-style analysis for GLV stochastic trajectories."""

from .fixed_point import flow_to_fixed_point, polish_fixed_point
from .clustering import GLVClusterer, ClusterConfig, ClusterResult
from .speed_geometry import SpeedGeometry, analyze_speed_geometry, print_speed_geometry
from .visualization import (
    plot_eigenvalues_complex,
    plot_eigenvalue_histogram,
    plot_pca_trajectory,
    plot_silhouette_sweep,
    plot_species_traces,
    plot_phase_flow,
)

__all__ = [
    "flow_to_fixed_point",
    "polish_fixed_point",
    "GLVClusterer",
    "ClusterConfig",
    "ClusterResult",
    "SpeedGeometry",
    "analyze_speed_geometry",
    "print_speed_geometry",
    "plot_eigenvalues_complex",
    "plot_eigenvalue_histogram",
    "plot_pca_trajectory",
    "plot_silhouette_sweep",
    "plot_species_traces",
    "plot_phase_flow",
]
