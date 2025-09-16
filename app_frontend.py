import io
import os
import sys
import subprocess
from typing import Tuple

import streamlit as st
import pandas as pd
import numpy as np

from helper import resolve_archive_mapping
from turtle_anim_from_csv import load_points
from kolam_math_report import load_path as _load_path
from kolam_math_report import estimate_grid, path_length, rotate, grid_snap_error


st.set_page_config(page_title="Kolam Recreator", layout="centered")
st.title("Kolam (Rangoli) Recreator – Archive Demo")

st.write(
    "Upload an archive image like 'kolam29-37.jpg'. We map it to the dataset CSV, "
    "animate the reconstruction, and show mathematical details."
)

uploaded = st.file_uploader("Upload kolam image (from archive)", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([1, 2])

with st.sidebar:
    st.header("Animation controls")
    delay = st.slider("Delay (s)", 0.0, 0.2, 0.02, 0.005)
    stride = st.slider("Stride (points/update)", 1, 10, 2, 1)
    st.header("Spline sampling for GeoGebra")
    knots = st.slider("Sampling points (knots)", 1000, 8000, 4000, 100)


def save_points_csv(points: np.ndarray, out_path: str):
    df = pd.DataFrame(points, columns=["x", "y"])
    df.to_csv(out_path, index=False, header=False)


def resample_points(P: np.ndarray, n: int) -> np.ndarray:
    if len(P) <= n:
        return P
    t = np.linspace(0, 1, len(P))
    s = np.linspace(0, 1, n)
    x = np.interp(s, t, P[:, 0])
    y = np.interp(s, t, P[:, 1])
    return np.c_[x, y]


def render_math(csv_path: str, idx: int):
    P = _load_path(csv_path, idx)
    ang, g = estimate_grid(P)
    L = path_length(P)
    e_rot90 = float(np.mean(np.linalg.norm(rotate(P, 90) - P, axis=1)))
    e_mx = float(np.mean(np.linalg.norm(np.c_[P[:, 0], -P[:, 1]] - P, axis=1)))
    e_my = float(np.mean(np.linalg.norm(np.c_[-P[:, 0], P[:, 1]] - P, axis=1)))
    st.subheader("Mathematical details")
    st.markdown(
        f"- CSV: `{csv_path}`\n"
        f"- Index (1-based): **{idx}**\n"
        f"- Estimated rotation: **{ang:.1f}°**\n"
        f"- Grid spacing: **{g:.3f}**\n"
        f"- Path length: **{L:.2f}** (normalized units)\n"
        f"- Symmetry error (rot90/mirrorX/mirrorY): **{e_rot90:.3f} / {e_mx:.3f} / {e_my:.3f}**"
    )

    st.caption(
        "GeoGebra helper: `P := Sequence((Element(A1:A4001, i), Element(B1:B4001, i)), i, 1, 4001)` "
        "then draw `Polyline(P)` in Classic."
    )


if uploaded is not None:
    name = uploaded.name
    try:
        csv_path, idx = resolve_archive_mapping(name)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with col1:
        st.image(uploaded, caption=name, use_container_width=True)
    with col2:
        st.success(f"Mapped to {csv_path} (idx {idx})")

    # Load normalized points for animation and a denser set for GeoGebra CSV
    try:
        P = load_points(csv_path, idx)
    except Exception as e:
        st.error(f"Failed to load points: {e}")
        st.stop()

    # Save a points CSV for GeoGebra/Plotly if user wants (resampled by knots)
    if st.button("Export points CSV for GeoGebra/Plotly"):
        P_geo = resample_points(_load_path(csv_path, idx), knots)
        out_path = f"points_{os.path.splitext(name)[0]}_{knots}.csv"
        save_points_csv(P_geo, out_path)
        with open(out_path, 'rb') as f:
            st.download_button("Download CSV", f, file_name=out_path)

    # Show math
    render_math(csv_path, idx)

    st.subheader("Animate reconstruction (opens Turtle window)")
    st.write("Click to run; a separate Python process will open a Turtle window.")
    if st.button("Run animation"):
        try:
            subprocess.Popen([
                sys.executable,
                'turtle_anim_from_csv.py',
                '--csv', csv_path,
                '--idx', str(idx),
                '--delay', str(delay),
                '--stride', str(stride)
            ])
            st.success("Animation process started. Switch to the Turtle window.")
        except Exception as e:
            st.error(f"Failed to start animation: {e}")

else:
    st.info("Upload an image like kolam19-0.jpg from the archive.")


