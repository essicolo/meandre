"""La pluie d'été de CaSR (forçage de meandre) contre celle de GOENS (forçage de l'ensemble Hydrotel),
sur le bassin amont de chaque station, 2019-2024. Sans simulation.

Hydrotel rend le volume des événements d'été (0,97 de l'observé au Saint-Laurent sud-ouest) là où
meandre en perd 40 % (0,60). Si la pluie de CaSR est nettement sous celle de GOENS sur ces bassins,
une bonne part de l'écart est de forçage, pas de modèle.

GOENS : précipitation en pas de 3 heures sur grille de 0,1°, rangée une grille entière par pas de
temps ; lue mois par mois et ramenée au bassin par une matrice de poids (aire locale des tronçons
amont, cellule la plus proche du centroïde du tronçon). CaSR : moyenne de bassin déjà calculée avec
les mêmes poids (experiments/gru_residual/donnees.py).

    .venv/Scripts/python.exe experiments/plan_ml/ete-pluie-goens.py
"""
import os
import sys
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
GOENS = "D:/rqh/meteo/goens/GRILLE_PRECIP_QCMERI_GOENS00_MB0000.nc"


def poids_bassins(stations, lon_g, lat_g):
    """Matrice (stations × cellules GOENS) des poids de bassin."""
    W = np.zeros((len(stations), len(lon_g) * len(lat_g)), dtype=np.float32)
    par_reg = defaultdict(list)
    for k, s in enumerate(stations):
        par_reg[s.split(":")[0]].append((k, s.split(":")[1]))
    for reg, liste in par_reg.items():
        con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
        nodes = con.sql("select n.node_idx, n.lon, n.lat, t.area_km2_local from nodes n join territorial t using (node_idx) order by node_idx").df()
        edges = con.sql("select src, dst from edges").df()
        st = dict(con.sql("select station_id, node_idx from stations").fetchall())
        con.close()
        parents = defaultdict(list)
        for a, b in zip(edges.src, edges.dst):
            parents[int(b)].append(int(a))
        ix = np.abs(lon_g[None, :] - nodes.lon.to_numpy()[:, None]).argmin(1)
        iy = np.abs(lat_g[None, :] - nodes.lat.to_numpy()[:, None]).argmin(1)
        cell = ix * len(lat_g) + iy
        aire = nodes.area_km2_local.to_numpy()
        for k, code in liste:
            seen, pile = {st[code]}, [st[code]]
            while pile:
                for p in parents.get(pile.pop(), []):
                    if p not in seen:
                        seen.add(p)
                        pile.append(p)
            idx = np.fromiter(seen, int)
            np.add.at(W[k], cell[idx], aire[idx] / aire[idx].sum())
    return W


def main():
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    stations = sorted({c[0] for c in fx.columns})
    d = xr.open_dataset(GOENS)
    lon_g, lat_g = d.lon.values, d.lat.values
    W = poids_bassins(stations, lon_g, lat_g)
    cellules = np.flatnonzero(W.sum(0) > 0)
    Wc = W[:, cellules]
    mois = pd.period_range("2019-01", "2024-12", freq="M")
    goens = {}
    for m in mois:
        bloc = d.pr.sel(time=slice(str(m.start_time), str(m.end_time))).values  # (lon, lat, t)
        g = bloc.reshape(-1, bloc.shape[-1])[cellules]                          # (cellules, t)
        goens[m] = Wc @ np.nan_to_num(g.sum(1))                                   # mm du mois, par station
    G = pd.DataFrame(goens, index=stations).T
    C = pd.concat({s: fx[s]["P"] for s in stations}, axis=1)
    C = C[(C.index >= "2019-01-01") & (C.index <= "2024-12-31")].resample("MS").sum()
    C.index = C.index.to_period("M")
    R = (C / G).replace([np.inf, -np.inf], np.nan)
    long = R.stack().rename("rapport").reset_index()
    long.columns = ["mois", "station", "rapport"]
    long["territoire"] = long.station.str.split(":").str[0]
    long["m"] = long.mois.dt.month
    long.to_parquet(f"{DOSSIER}/ete-pluie-goens.parquet")
    print("Pluie CaSR sur pluie GOENS, bassin amont des stations, 2019-2024, médiane des stations et des années, par mois")
    T = long.groupby(["territoire", "m"]).rapport.median().unstack().round(2)
    T["stations"] = long.groupby("territoire").station.nunique()
    print(T.sort_values("stations", ascending=False).to_string())
    ete = long[long.m.isin([6, 7, 8, 9])]
    tot = pd.DataFrame({"casr": C.loc[[p for p in C.index if p.month in (6, 7, 8, 9)]].sum(), "goens": G.loc[[p for p in G.index if p.month in (6, 7, 8, 9)]].sum()})
    tot["rapport"] = tot.casr / tot.goens
    tot["territoire"] = tot.index.str.split(":").str[0]
    print("\nCumul de juin à septembre 2019-2024, CaSR sur GOENS, médiane des stations par territoire :")
    print(tot.groupby("territoire").rapport.describe()[["count", "25%", "50%", "75%"]].round(2).sort_values("count", ascending=False).to_string())
    tot.to_csv(f"{DOSSIER}/ete-pluie-goens-cumul.csv")


if __name__ == "__main__":
    main()
