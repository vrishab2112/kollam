import pandas as pd, numpy as np, matplotlib.pyplot as plt

csv_path = r"C:\Users\Vrishab Vishnu\Desktop\SIH\archive (1)\Kolam CSV files\Kolam CSV files\kolam29.csv"
df = pd.read_csv(csv_path)

def get_path(df, idx=1):
  x = df[f"x-kolam {idx}"].to_numpy()
  y = df[f"y-kolam {idx}"].to_numpy()
  P = np.c_[x, y]
  # drop rows with NaNs (many CSV columns have trailing empties)
  P = P[~np.isnan(P).any(axis=1)]
  if len(P) == 0:
    raise ValueError("Selected kolam has no valid points (all NaNs)")
  # normalize and center (optional)
  P -= P.mean(0)
  s = np.percentile(np.linalg.norm(P, axis=1), 95)
  s = s if (np.isfinite(s) and s > 0) else 1.0
  return P / s

P = get_path(df, idx=37)
plt.figure(figsize=(6,6)); ax = plt.gca()
ax.plot(P[:,0], P[:,1], color="white", lw=3)
ax.set_aspect("equal"); ax.set_facecolor("#2e5f3b"); ax.axis("off")
# If your viewer doesn't pop up, you'll still get a file next to this script
plt.savefig("kolam_preview.png", dpi=180, bbox_inches="tight", pad_inches=0, facecolor=ax.get_facecolor())
plt.show()