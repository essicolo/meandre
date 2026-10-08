"""Hydrogramme vitrine pour le README (année held-out 2023, bandes quantiles).
  python .runs/slso/plot_hydrograph_readme.py 023402
Sortie : docs/img/hydrograph-<sta>-2023.png (300 dpi).
"""
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STA = sys.argv[1] if len(sys.argv) > 1 else "023402"
df = pd.read_csv(f".runs/slso/results/hydrograph-{STA}-2023.csv", parse_dates=["date"])
m = df.discharge.notna()
r = np.corrcoef(df.q_med[m], df.discharge[m])[0, 1]
b = df.q_med[m].mean() / df.discharge[m].mean()
g = (df.q_med[m].std() / df.q_med[m].mean()) / (df.discharge[m].std() / df.discharge[m].mean())
kge = 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)
cov90 = ((df.discharge >= df.q05) & (df.discharge <= df.q95))[m].mean()

C_MED = "#1b4b8f"; C_50 = "#5b8ec4"; C_90 = "#a8c6e4"; C_OBS = "#2e2e2e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#666666", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=300)
ax.fill_between(df.date, df.q05, df.q95, color=C_90, alpha=0.45, lw=0, label="90% interval")
ax.fill_between(df.date, df.q25, df.q75, color=C_50, alpha=0.45, lw=0, label="50% interval")
ax.plot(df.date, df.q_med, color=C_MED, lw=1.6, label="meandre (median)")
ax.plot(df.date[m], df.discharge[m], ".", color=C_OBS, ms=3.2, alpha=0.85, label="observed")
ax.set_ylabel("Discharge (m³/s)")
ax.set_xlim(df.date.iloc[0], df.date.iloc[-1])
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.6)
AREAS = {"023402": 5820, "024014": 2163}
ax.set_title(f"Gauge {STA} ({AREAS.get(STA, '?'):,} km²), Chaudière-Appalaches, Québec — "
             f"held-out year 2023 (never seen in training)",
             loc="left", fontsize=12.5, pad=10)
ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=10)
ax.annotate(f"KGE = {kge:.2f}   90% coverage = {cov90:.2f}",
            xy=(0.012, 0.94), xycoords="axes fraction", fontsize=10.5, color="#444444")
fig.tight_layout()
os.makedirs("docs/img", exist_ok=True)
out = f"docs/img/hydrograph-{STA}-2023.png"
fig.savefig(out, bbox_inches="tight")
print(f"[ok] {out} | KGE {kge:.3f} | cov90 {cov90:.3f} | n_obs {int(m.sum())}")
