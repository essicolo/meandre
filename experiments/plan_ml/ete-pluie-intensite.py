"""Pluie journalière d'été, CaSR contre GOENS, bassins des stations du sud (Saint-Laurent sud-ouest,
Montérégie) et de deux territoires témoins (Outaouais, Saint-Laurent nord-ouest), juin-septembre
2019-2024 : corrélation journalière au meilleur décalage, part de la pluie tombée les jours de plus
de 20 mm, nombre de jours de plus de 20 mm. Sans simulation.

    .venv/Scripts/python.exe experiments/plan_ml/ete-pluie-intensite.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

spec = importlib.util.spec_from_file_location("pg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ete-pluie-goens.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)
REGIONS = ["slso", "mont", "outv", "slno"]


def main():
    fx = pd.read_parquet(f"{pg.REF}/forcage_bassin.parquet")
    stations = sorted({c[0] for c in fx.columns if c[0].split(":")[0] in REGIONS})
    d = xr.open_dataset(pg.GOENS)
    W = pg.poids_bassins(stations, d.lon.values, d.lat.values)
    cel = np.flatnonzero(W.sum(0) > 0)
    Wc = W[:, cel]
    blocs = []
    for an in range(2019, 2025):
        b = d.pr.sel(time=slice(f"{an}-05-31T08:00", f"{an}-10-01T07:00"))
        v = b.values.reshape(-1, b.sizes["time"])[cel]
        jours = pd.DatetimeIndex(b.time.values - np.timedelta64(8, "h")).normalize()
        st = pd.DataFrame((Wc @ np.nan_to_num(v)).T, index=jours, columns=stations).groupby(level=0).sum()
        blocs.append(st)
    G = pd.concat(blocs)
    G = G[G.index.month.isin([6, 7, 8, 9])]
    lignes = []
    for s in stations:
        c = fx[s]["P"].reindex(G.index)
        g = G[s]
        corr = {k: c.corr(g.shift(k)) for k in (-1, 0, 1)}
        k = max(corr, key=corr.get)
        lignes.append(dict(station=s, territoire=s.split(":")[0], decalage=k, corr=corr[k], part_20_casr=c[c > 20].sum() / c.sum(), part_20_goens=g[g > 20].sum() / g.sum(), jours_20_casr=(c > 20).sum() / 6, jours_20_goens=(g > 20).sum() / 6, max_casr=c.groupby(c.index.year).max().median(), max_goens=g.groupby(g.index.year).max().median(), jours_secs_casr=(c < 1).mean(), jours_secs_goens=(g < 1).mean()))
    L = pd.DataFrame(lignes)
    L.to_csv(f"{pg.DOSSIER}/ete-pluie-intensite.csv", index=False)
    print("Juin-septembre 2019-2024, pluie de bassin, médiane des stations")
    print(L.groupby("territoire")[["corr", "part_20_casr", "part_20_goens", "jours_20_casr", "jours_20_goens", "max_casr", "max_goens", "jours_secs_casr", "jours_secs_goens"]].median().round(2).to_string())
    print("décalage retenu (jours, GOENS décalé) :", L.decalage.value_counts().to_dict())


if __name__ == "__main__":
    main()
