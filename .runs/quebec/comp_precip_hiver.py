"""ECCC contre CaSR sur la precipitation hivernale : la donnee est-elle en cause ?

Motif (2026-08-27). Le manteau simule au Saguenay est deja 22 % trop leger en decembre,
avant toute fonte (SAGU 68 mm contre 84 mesures par CanSWE). CaSR ne fournit AUCUNE
variable de neige : le partage pluie/neige est un CALCUL de meandre depuis la
precipitation totale et la temperature (bulbe humide), jamais une donnee. Le deficit a
donc deux causes possibles, indiscernables sans ce test : CaSR sous-estime la
precipitation TOTALE, ou le partage envoie a tort trop d'eau vers la pluie.

Ce script compare, station par station, la precipitation TOTALE mesuree par le reseau
climatologique ECCC (jauge au sol) a celle de CaSR au noeud le plus proche, pour les
mois de decembre a fevrier. Si CaSR est deja bas contre ECCC, la cause est la donnee ;
sinon, elle est le partage (le seuil du bulbe humide).

RESERVE CONNUE : les jauges au sol SOUS-CAPTENT elles-memes la neige a cause du vent.
Un accord ECCC/CaSR n'exclut donc pas un biais commun aux deux ; il exclut seulement
que CaSR soit MOINS BON que la reference dont on dispose.

    python .runs/quebec/comp_precip_hiver.py sagu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.data.eccc_loader import fetch_daily, fetch_stations
from meandre.utils import paths as _paths

REG = (sys.argv[1] if len(sys.argv) > 1 else "sagu").lower()
h = BasinCache(_paths.data_path("quebec", f"{REG}.duckdb")).load(device="cpu")
nc = h["node_coords"].numpy()
bbox = (round(float(nc[:, 0].min()) - 0.5, 1), round(float(nc[:, 1].min()) - 0.5, 1),
        round(float(nc[:, 0].max()) + 0.5, 1), round(float(nc[:, 1].max()) + 0.5, 1))
print(f"[{REG}] bbox {bbox}", flush=True)

st = fetch_stations(bbox)
print(f"[{REG}] {len(st)} stations ECCC dans l'inventaire", flush=True)
df = fetch_daily(bbox, 2010, 2020)
df = df.merge(st[["climate_id", "elev"]], on="climate_id", how="left")
df["mois"] = df.date.dt.month
hiver = df[df.mois.isin([12, 1, 2]) & df.P.notna()]
compte = hiver.groupby("climate_id").size()
gard = compte[compte >= 200].index          # stations avec assez de jours d'hiver
hiver = hiver[hiver.climate_id.isin(gard)]
print(f"[{REG}] {len(gard)} stations retenues (>=200 jours DJF entre 2010 et 2020)",
      flush=True)

fx = f"{_paths.DATA_ROOT}/quebec/forcing-{REG}-hyb.nc"
if not os.path.exists(fx):
    fx = f"{_paths.DATA_ROOT}/quebec/forcing-{REG}-budyko.nc"
d = xr.open_dataset(fx)
times = pd.DatetimeIndex(d["time"].values)
P_casr = d["forcing"].values[:, :, 0]        # (T, n_noeuds) mm/jour
d.close()

lon_n, lat_n = nc[:, 0], nc[:, 1]
lignes = []
for cid, g in hiver.groupby("climate_id"):
    lon0, lat0 = g.lon.iloc[0], g.lat.iloc[0]
    j = int(np.argmin((lon_n - lon0) ** 2 + (lat_n - lat0) ** 2))
    dist_km = 111.0 * np.hypot(lon_n[j] - lon0, (lat_n[j] - lat0) * np.cos(np.radians(lat0)))
    if dist_km > 25:
        continue
    idx = times.get_indexer(g.date)
    ok = idx >= 0
    if ok.sum() < 100:
        continue
    p_eccc = g.P.to_numpy()[ok]
    p_casr = P_casr[idx[ok], j]
    fin = np.isfinite(p_eccc) & np.isfinite(p_casr)
    if fin.sum() < 100:
        continue
    lignes.append(dict(station=cid, dist_km=dist_km, n=int(fin.sum()),
                       eccc_mm_j=float(p_eccc[fin].mean()), casr_mm_j=float(p_casr[fin].mean()),
                       rapport=float(p_casr[fin].sum() / max(p_eccc[fin].sum(), 1e-9))))
r = pd.DataFrame(lignes)
if r.empty:
    print(f"[{REG}] aucune station appariee a moins de 25 km d'un noeud"); sys.exit()
print(f"\n[{REG}] {len(r)} stations appariees (DJF, 2010-2020, <25 km d'un noeud)")
print(r.sort_values("dist_km")[["station", "dist_km", "n", "eccc_mm_j", "casr_mm_j", "rapport"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nrapport CaSR/ECCC median : {r.rapport.median():.3f}  "
      f"(< 1 : CaSR sous-estime la precipitation hivernale totale, cause = la DONNEE)")
print("(reserve : les jauges au sol sous-captent elles-memes la neige sous le vent --")
print(" un rapport proche de 1 n'exclut pas un biais COMMUN aux deux)")
