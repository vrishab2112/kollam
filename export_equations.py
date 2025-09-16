import argparse
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_points(csv_path: str, idx: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    x = df[f"x-kolam {idx}"].to_numpy()
    y = df[f"y-kolam {idx}"].to_numpy()
    P = np.c_[x, y]
    P = P[np.isfinite(P).all(axis=1)]
    # center (for nicer coefficients) but keep scale
    P = P - P.mean(axis=0)
    return P


def arclength_parameter(P: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    # merge any duplicate points
    keep = np.r_[True, d > 1e-12]
    return t[keep], P[keep]


def natural_cubic_spline_coeffs(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coefficients a,b,c,d for cubic spline S_i(τ) = a_i + b_i τ + c_i τ^2 + d_i τ^3,
    with τ = t - t_i on interval [t_i, t_{i+1}]. Natural boundary conditions.
    y: shape (n,) values at knots t.
    """
    n = len(t) - 1
    h = np.diff(t)
    # Build tridiagonal system for c (second-derivative-related term)
    alpha = np.zeros(n+1)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (y[i+1] - y[i]) - (3.0 / h[i-1]) * (y[i] - y[i-1])

    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    for i in range(1, n):
        l[i] = 2.0 * (t[i+1] - t[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    c = np.zeros(n+1)
    b = np.zeros(n)
    d = np.zeros(n)
    a = y[:-1].copy()
    # back substitution
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
    return a, b, c[:-1], d


def resample_knots(t: np.ndarray, P: np.ndarray, max_knots: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(t) <= max_knots:
        return t, P
    s = np.linspace(0.0, t[-1], max_knots)
    # linear interpolation to get new knots positions
    Px = np.interp(s, t, P[:, 0])
    Py = np.interp(s, t, P[:, 1])
    return s, np.c_[Px, Py]


def export_spline(csv: str, idx: int, out: str, max_knots: int = 300) -> Dict:
    P = load_points(csv, idx)
    t, P = arclength_parameter(P)
    t, P = resample_knots(t, P, max_knots)
    ax, bx, cx, dx = natural_cubic_spline_coeffs(t, P[:, 0])
    ay, by, cy, dy = natural_cubic_spline_coeffs(t, P[:, 1])

    data = {
        "notes": "Natural cubic spline parameterized by arc length t. For t in [t[i], t[i+1]], x(t)=ax[i]+bx[i]*(t-t[i])+cx[i]*(t-t[i])**2+dx[i]*(t-t[i])**3, y(t)=ay[i]+by[i]*(t-t[i])+cy[i]*(t-t[i])**2+dy[i]*(t-t[i])**3",
        "t": t.tolist(),
        "x": {"a": ax.tolist(), "b": bx.tolist(), "c": cx.tolist(), "d": dx.tolist()},
        "y": {"a": ay.tolist(), "b": by.tolist(), "c": cy.tolist(), "d": dy.tolist()},
        "center": P.mean(axis=0).tolist(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--idx", type=int, default=1)
    ap.add_argument("--out", default="kolam_spline.json")
    ap.add_argument("--max_knots", type=int, default=300)
    args = ap.parse_args()
    export_spline(args.csv, args.idx, args.out, args.max_knots)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


