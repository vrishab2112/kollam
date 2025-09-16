import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt


@dataclass
class KolamReport:
    num_points: int
    length: float
    rotation_deg_best: float
    grid_spacing: float
    grid_snap_error: float
    symmetry_rot90_error: float
    symmetry_mirror_x_error: float
    symmetry_mirror_y_error: float


def load_path(csv_path: str, idx: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    x = df[f"x-kolam {idx}"]
    y = df[f"y-kolam {idx}"]
    P = np.c_[x.to_numpy(), y.to_numpy()]
    P = P[~np.isnan(P).any(axis=1)]
    # center only; keep native scale to avoid over/under-compression in overlays
    P = P - P.mean(0)
    return P


def rotate(P: np.ndarray, deg: float) -> np.ndarray:
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return P @ R.T


def grid_snap_error(P: np.ndarray, g: float) -> float:
    # error to nearest multiple of g on both axes
    Q = P / g
    frac = np.abs(Q - np.round(Q))
    return float(np.mean(frac))


def estimate_grid(P: np.ndarray) -> Tuple[float, float]:
    """Estimate lattice rotation (0/45/90/135) and a visually appropriate grid spacing.
    We choose the angle that minimizes snap error using a coarse spacing, then
    set spacing so the overlay shows ~18 grid steps across the width.
    """
    candidate_angles = [0.0, 45.0, 90.0, 135.0]
    best_ang = 0.0
    best_err = 1e9
    for ang in candidate_angles:
        Pr = rotate(P, -ang)
        width = float(np.ptp(Pr[:, 0]))
        g0 = max(width / 18.0, 1e-6)
        err = grid_snap_error(Pr, g0)
        if err < best_err:
            best_err = err
            best_ang = ang
    # set spacing based on chosen angle's width
    Pr = rotate(P, -best_ang)
    width = float(np.ptp(Pr[:, 0]))
    g = max(width / 18.0, 1e-6)
    return best_ang, g


def path_length(P: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def chamfer_error(A: np.ndarray, B: np.ndarray) -> float:
    # symmetric nearest-neighbor mean distance (naive O(N^2) ok for small N)
    def one_side(U, V):
        d2 = ((U[:, None, :] - V[None, :, :]) ** 2).sum(-1)
        return np.mean(np.sqrt(d2.min(axis=1)))
    return float(0.5 * (one_side(A, B) + one_side(B, A)))


def make_overlay(P: np.ndarray, ang: float, g: float, out_png: str) -> None:
    # draw points, grid axes after rotation, and principal axes
    Pr = rotate(P, -ang)
    xmin, ymin = Pr.min(axis=0) - 0.5
    xmax, ymax = Pr.max(axis=0) + 0.5
    x_ticks = np.arange(math.floor(xmin / g) * g, math.ceil(xmax / g) * g + 0.5 * g, g)
    y_ticks = np.arange(math.floor(ymin / g) * g, math.ceil(ymax / g) * g + 0.5 * g, g)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#2e5f3b')
    ax.set_facecolor('#2e5f3b')
    ax.patch.set_alpha(1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # grid in rotated coordinates
    for x in x_ticks:
        ax.plot([x, x], [ymin, ymax], color='white', alpha=0.25, lw=1)
    for y in y_ticks:
        ax.plot([xmin, xmax], [y, y], color='white', alpha=0.25, lw=1)

    # path in rotated frame
    ax.plot(Pr[:, 0], Pr[:, 1], color='white', lw=2.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.savefig(out_png, dpi=180, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_simple_overlay(P: np.ndarray, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#2e5f3b')
    ax.set_aspect('equal')
    ax.set_facecolor('#2e5f3b')
    ax.axis('off')
    ax.plot(P[:, 0], P[:, 1], color='white', lw=2.8)
    pad = 0.05 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
    ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
    fig.savefig(out_png, dpi=180, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(csv: str, idx: int, out_prefix: str, simple: bool) -> None:
    P = load_path(csv, idx)
    ang, g = estimate_grid(P)
    L = path_length(P)

    # symmetry scores (lower is more symmetric). Evaluate in normalized units
    e_rot90 = chamfer_error(P, rotate(P, 90))
    e_mx = chamfer_error(P, np.c_[P[:, 0], -P[:, 1]])
    e_my = chamfer_error(P, np.c_[-P[:, 0], P[:, 1]])

    rep = KolamReport(
        num_points=len(P),
        length=L,
        rotation_deg_best=ang,
        grid_spacing=g,
        grid_snap_error=grid_snap_error(rotate(P, -ang), g),
        symmetry_rot90_error=e_rot90,
        symmetry_mirror_x_error=e_mx,
        symmetry_mirror_y_error=e_my,
    )

    out_png = f"{out_prefix}_overlay.png"
    if simple:
        make_simple_overlay(P, out_png)
    else:
        make_overlay(P, ang, g, out_png)
    out_json = f"{out_prefix}_report.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(asdict(rep), f, indent=2)
    print(f"Wrote {out_png} and {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--idx', type=int, default=1)
    ap.add_argument('--out', default='kolam_math')
    ap.add_argument('--simple', action='store_true', help='draw only the polyline (no rotation/grid)')
    args = ap.parse_args()
    main(args.csv, args.idx, args.out, args.simple)


