# Kolam (Rangoli) – Analysis, Math, and Reconstruction

This repo includes tools to:
- animate kolam paths from CSV,
- compute/visualize math (grid, rotation, symmetry),
- export exact parametric equations (natural cubic splines).

## Quick start (Windows/PowerShell)

Install:
```powershell
py -m pip install -U numpy pandas matplotlib pillow
```

Animate (step-by-step Turtle):
```powershell
py turtle_anim_from_csv.py --csv ".\archive (1)\Kolam CSV files\Kolam CSV files\kolam29.csv" --idx 37 --delay 0.01 --stride 2
```

Math overlay + JSON report:
```powershell
py kolam_math_report.py --csv ".\archive (1)\Kolam CSV files\Kolam CSV files\kolam29.csv" --idx 37 --out math_k29_37
```

Simple polyline overlay only:
```powershell
py kolam_math_report.py --csv ".\archive (1)\Kolam CSV files\Kolam CSV files\kolam29.csv" --idx 37 --out math_k29_37 --simple
```

Export equations (natural cubic spline):
```powershell
py export_equations.py --csv ".\archive (1)\Kolam CSV files\Kolam CSV files\kolam29.csv" --idx 37 --out kolam29_37_spline.json --max_knots 300
```

Notes:
- Image `kolamXX-k.jpg` ↔ CSV columns `x-kolam {k+1}`, `y-kolam {k+1}`.
- If modules are missing: `py -0p` to see versions, then `py -3.12 -m pip install ...`.

## Files
- `turtle_anim_from_csv.py` — step animation.
- `kolam_math_report.py` — rotation/grid estimation, overlay, symmetry JSON.
- `export_equations.py` — spline equations JSON.
- `animate_kolam.py`, `kolam_turtle.py`, `test.py` — extras/demos.

## Mathematics (fundamental formula)
For knots `t[i]` and τ = t − t[i], each interval uses cubic polynomials:
- x(t) = ax[i] + bx[i] τ + cx[i] τ² + dx[i] τ³
- y(t) = ay[i] + by[i] τ + cy[i] τ² + dy[i] τ³

Derivatives:
- x'(t) = bx + 2 cx τ + 3 dx τ², y'(t) = by + 2 cy τ + 3 dy τ²
- Curvature κ(t) = (x' y'' − y' x'') / (x'² + y'²)^(3/2)

## Visualize elsewhere
- GeoGebra Classic: Spreadsheet → paste two columns → Create → List of Points → `Polyline(L1)` → style.
- p5.js Web Editor: load `kolam29_37_spline.json`, sample and draw.
- Plotly Chart Studio: import `kolam_points.csv` → Scatter (Lines) → export.

Create `kolam_points.csv` from the JSON:
```python
import json, csv
S=json.load(open('kolam29_37_spline.json'))
T=S['t']; ax, bx, cx, dx=[S['x'][k] for k in 'abcd']; ay, by, cy, dy=[S['y'][k] for k in 'abcd']
N=4000; i=0; pts=[]
for k in range(N):
    tt=T[0]+(T[-1]-T[0])*k/(N-1)
    while i < len(T)-2 and tt > T[i+1]: i+=1
    u=tt-T[i]
    x=ax[i]+bx[i]*u+cx[i]*u*u+dx[i]*u*u*u
    y=ay[i]+by[i]*u+cy[i]*u*u+dy[i]*u*u*u
    pts.append((x,y))
csv.writer(open('kolam_points.csv','w',newline='')).writerows(pts)
```

## What to demo
1) Run Turtle animation.
2) Show `math_..._report.json` + overlay.
3) Show spline JSON equations and render in p5.js/GeoGebra.

updated frontend = app_frontend.py