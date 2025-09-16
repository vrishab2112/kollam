import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_path(csv_path: str, idx: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    x = df[f"x-kolam {idx}"]
    y = df[f"y-kolam {idx}"]
    P = np.c_[x.to_numpy(), y.to_numpy()]
    P = P[~np.isnan(P).any(axis=1)]
    P = P - P.mean(0)
    s = np.percentile(np.linalg.norm(P, axis=1), 95)
    s = s if (np.isfinite(s) and s > 0) else 1.0
    return P / s


def animate(csv_path: str, idx: int, fps: int = 30, save_gif: bool = False) -> None:
    P = load_path(csv_path, idx)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('#2e5f3b')
    ax.axis('off')

    # full reference (light) and progressive (bright)
    ref, = ax.plot(P[:, 0], P[:, 1], color='white', lw=1.5, alpha=0.25)
    prog, = ax.plot([], [], color='white', lw=3)

    def init():
        prog.set_data([], [])
        return prog,

    def update(frame: int):
        k = max(2, frame)
        prog.set_data(P[:k, 0], P[:k, 1])
        return prog,

    frames = len(P)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000 // fps, blit=True)

    if save_gif:
        out = f"kolam_anim_{os.path.splitext(os.path.basename(csv_path))[0]}_{idx}.gif"
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--gif', action='store_true')
    args = parser.parse_args()
    animate(args.csv, args.idx, fps=args.fps, save_gif=args.gif)


