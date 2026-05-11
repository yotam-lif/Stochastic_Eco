"""Interactive HTML animation of GLV stochastic trajectory in 2D PC space.

Runs the SDE from the deterministic FP, projects the trajectory into 2D
PCA space, clusters it with k-means, then writes a self-contained Plotly
HTML animation showing:
  - trailing trajectory (configurable memory window)
  - points colored by cluster label
  - black dot for the current position
  - play/pause buttons and time slider

Analogous to GARD/code/composome_trajectory_animation.py.

Usage (from project root):
    .venv/bin/python scripts/trajectory_animation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from base_structure import GLVModel
from composome_analysis import (
    flow_to_fixed_point,
    GLVClusterer,
)

# ── Model parameters ──────────────────────────────────────────────────────────
S       = 200
MU      = 4.0
SIGMA   = 0.8
GAMMA   = 0.0
SIGMA_K = 0.0
SEED    = 42

# ── Simulation parameters ─────────────────────────────────────────────────────
D          = 0.1       # larger noise — explore faster
T_MAX      = 20_000.0  # longer run
DT         = 0.01
SAVE_EVERY = 50        # dt_save = 0.5 time units; ~40 000 saved points
NOISE_TYPE = "demographic"

# ── Animation parameters ──────────────────────────────────────────────────────
MEMORY           = 200   # trailing saved steps shown (~100 time units)
FRAME_STEP       = 20    # one frame every FRAME_STEP saved steps (~2000 frames)
FRAME_DURATION   = 80    # ms per frame

FIG_DIR  = Path(__file__).parent / "figs"
OUT_PATH = FIG_DIR / f"trajectory_animation_D{D}_T{int(T_MAX)}.html"

# ── Colour palette (cluster 0 = blue, 1 = orange, …) ─────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _axis_range(values: np.ndarray, pad: float = 0.05) -> list[float]:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = vmax - vmin if vmax > vmin else 1.0
    return [vmin - pad * span, vmax + pad * span]


def _make_frame_data(
    coords: np.ndarray,       # (n_saved, 2)
    labels: np.ndarray,       # (n_saved,) int  0-indexed cluster labels
    end: int,
    memory: int,
) -> list[dict]:
    start = max(0, end - memory)
    idx   = np.arange(start, end)
    xy    = coords[idx]       # (window, 2)
    lbl   = labels[idx]       # (window,)

    # One trace per cluster in the window so Plotly legends work nicely.
    k = int(labels.max()) + 1
    traces: list[dict] = []

    # Thin grey connecting line for the whole window.
    traces.append({
        "type": "scatter", "mode": "lines",
        "name": "trajectory",
        "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist(),
        "line": {"color": "#aaaaaa", "width": 1},
        "hoverinfo": "skip", "showlegend": False,
    })

    # Cluster scatter (semi-transparent dots).
    for ci in range(k):
        mask = lbl == ci
        if not np.any(mask):
            continue
        traces.append({
            "type": "scatter", "mode": "markers",
            "name": f"Cluster {ci}",
            "x": xy[mask, 0].tolist(), "y": xy[mask, 1].tolist(),
            "marker": {"size": 6, "opacity": 0.75,
                       "color": PALETTE[ci % len(PALETTE)]},
            "hovertemplate": f"Cluster {ci}<extra></extra>",
        })

    # Bold current position.
    traces.append({
        "type": "scatter", "mode": "markers",
        "name": "current",
        "x": [float(coords[end - 1, 0])],
        "y": [float(coords[end - 1, 1])],
        "marker": {"size": 12, "opacity": 1.0, "color": "#111111",
                   "line": {"color": "white", "width": 1}},
        "hovertemplate": f"t={end * SAVE_EVERY * DT:.1f}<extra></extra>",
        "showlegend": False,
    })

    return traces


def write_animation_html(
    coords: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    t_saved: np.ndarray,
    output_path: Path,
) -> None:
    n = coords.shape[0]
    frame_indices = list(range(MEMORY, n + 1, FRAME_STEP))
    if not frame_indices or frame_indices[-1] != n:
        frame_indices.append(n)

    k = int(labels.max()) + 1
    pct1, pct2 = 100 * explained[0], 100 * explained[1]

    def title(end: int) -> str:
        t = t_saved[end - 1]
        return (
            f"GLV trajectory in 2D PC space  —  "
            f"t = {t:.1f}  |  memory = {MEMORY * SAVE_EVERY * DT:.0f} t.u.  |  "
            f"PC1={pct1:.1f}%  PC2={pct2:.1f}%  |  k={k} clusters"
        )

    frames = [
        {
            "name": str(i),
            "data": _make_frame_data(coords, labels, i, MEMORY),
            "traces": list(range(k + 2)),   # line + k clusters + current
            "layout": {"title": {"text": title(i)}},
        }
        for i in frame_indices
    ]

    initial_data = frames[0]["data"]
    x_range = _axis_range(coords[:, 0])
    y_range = _axis_range(coords[:, 1])

    slider_steps = [
        {
            "label": f"{t_saved[i - 1]:.0f}",
            "method": "animate",
            "args": [
                [str(i)],
                {"mode": "immediate",
                 "frame": {"duration": FRAME_DURATION, "redraw": True},
                 "transition": {"duration": 0}},
            ],
        }
        for i in frame_indices
    ]

    layout = {
        "title": {"text": title(frame_indices[0])},
        "xaxis": {"title": f"PC 1 ({pct1:.1f}%)", "range": x_range},
        "yaxis": {"title": f"PC 2 ({pct2:.1f}%)", "range": y_range},
        "legend": {"orientation": "h", "y": -0.12},
        "plot_bgcolor": "#f9f9f9",
        "margin": {"l": 60, "r": 20, "b": 80, "t": 80},
        "updatemenus": [{
            "type": "buttons", "showactive": False,
            "x": 0.02, "y": 1.08,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"fromcurrent": True,
                                 "frame": {"duration": FRAME_DURATION, "redraw": True},
                                 "transition": {"duration": 0},
                                 "mode": "immediate"}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": False},
                                   "transition": {"duration": 0}}]},
            ],
        }],
        "sliders": [{
            "active": 0,
            "currentvalue": {"prefix": "time: ", "suffix": " t.u."},
            "pad": {"t": 45},
            "steps": slider_steps,
        }],
    }

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "  <title>GLV trajectory animation (2D PC)</title>\n"
        '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        "</head>\n<body>\n"
        '  <div id="plot" style="width: 100%; height: 95vh;"></div>\n'
        "  <script>\n"
        f"    const data   = {json.dumps(initial_data)};\n"
        f"    const layout = {json.dumps(layout)};\n"
        f"    const frames = {json.dumps(frames)};\n"
        "    Plotly.newPlot('plot', data, layout, {responsive: true}).then(function(){\n"
        "      Plotly.addFrames('plot', frames);\n"
        "    });\n"
        "  </script>\n</body>\n</html>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"GLVModel  S={S}, mu={MU}, sigma={SIGMA}, gamma={GAMMA}, seed={SEED}")
    model = GLVModel(S=S, mu=MU, sigma=SIGMA, gamma=GAMMA, sigma_K=SIGMA_K, seed=SEED)

    print("Finding fixed point...")
    fp = flow_to_fixed_point(model, t_max=5000)
    print(f"  phi={fp.phi:.3f}  ({len(fp.surviving)} survivors)")

    print(f"Integrating SDE  T={T_MAX}, dt={DT}, D={D}, save_every={SAVE_EVERY}...")
    sde = model.integrate_sde(
        N0=fp.N_star,
        t_span=(0.0, T_MAX),
        dt=DT,
        D=D,
        noise_type=NOISE_TYPE,
        save_every=SAVE_EVERY,
    )
    print(f"  trajectory shape: {sde.N.shape}")

    print("Computing 2D PCA...")
    pca = model.pca(sde, fp=fp, n_pcs=2)
    coords = pca.projections.T          # (n_saved, 2)
    explained = pca.explained_ratio
    print(f"  PC1={100*explained[0]:.1f}%  PC2={100*explained[1]:.1f}%")

    print("Clustering...")
    result = GLVClusterer().fit(coords)
    labels = result.labels
    print(f"  Selected k={result.selected_k}")
    counts = np.bincount(labels)
    for ci, cnt in enumerate(counts):
        print(f"    Cluster {ci}: {cnt} ({100*cnt/len(labels):.1f}%)")

    print(f"Writing HTML animation ({len(range(MEMORY, sde.N.shape[1]+1, FRAME_STEP))} frames)...")
    write_animation_html(
        coords=coords,
        labels=labels,
        explained=explained,
        t_saved=sde.t,
        output_path=OUT_PATH,
    )
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
