import argparse
import time
import numpy as np
import pandas as pd
import turtle as T


def load_points(csv_path: str, idx: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if f"x-kolam {idx}" not in df.columns:
        raise ValueError(f"Column x-kolam {idx} not found in CSV")
    x = df[f"x-kolam {idx}"].to_numpy()
    y = df[f"y-kolam {idx}"].to_numpy()
    P = np.c_[x, y]
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 2:
        raise ValueError("Not enough points after filtering NaNs")
    # center and scale to fit window nicely
    P = P - P.mean(axis=0)
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    span = span if span > 0 else 1.0
    P = P / span * 300.0  # fit into roughly 600x600 area
    return P


def animate(P: np.ndarray, step_delay: float = 0.01, stride: int = 1, bg: str = '#2e5f3b') -> None:
    screen = T.Screen()
    screen.setup(width=900, height=900)
    screen.bgcolor(bg)
    pen = T.Turtle(visible=False)
    pen.color('white')
    pen.pensize(4)
    pen.speed(0)
    T.tracer(0, 0)

    pen.up(); pen.goto(P[0, 0], P[0, 1]); pen.down()
    for i in range(1, len(P)):
        pen.goto(P[i, 0], P[i, 1])
        if i % stride == 0:
            T.update()
            if step_delay > 0:
                time.sleep(step_delay)
    T.update()
    T.done()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--idx', type=int, default=1)
    ap.add_argument('--delay', type=float, default=0.02, help='seconds between updates')
    ap.add_argument('--stride', type=int, default=2, help='draw this many points per update')
    args = ap.parse_args()

    points = load_points(args.csv, args.idx)
    animate(points, step_delay=args.delay, stride=args.stride)


