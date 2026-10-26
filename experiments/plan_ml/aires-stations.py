"""Aire du bassin d'une station dans le modèle (somme des aires locales des tronçons en amont du
tronçon de la station) contre l'aire drainée déclarée de la station, et volume simulé sur observé.
Sans simulation. Si les deux rapports suivent, une part de l'erreur de volume par station est un
défaut de rattachement de la station au réseau, pas de la physique.

    .venv/Scripts/python.exe experiments/plan_ml/aires-stations.py
"""
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

DATA = "D:/meandre-data"


def main():
    z = np.load(f"{DATA}/quebec/modele-provincial-2026-10/q-quebec-proba.npz", allow_pickle=True)
    t = pd.DatetimeIndex(z["dates"])
    ids = [str(s) for s in z["station_ids"]]
    S = pd.DataFrame(z["q_sim"], index=t, columns=ids)
    O = pd.DataFrame(z["q_obs"], index=t, columns=ids)
    m = O.notna() & S.notna()
    vol = (S.where(m).sum() / O.where(m).sum()).replace([np.inf, -np.inf], np.nan)
    ete = t.month.isin([6, 7, 8, 9])
    vol_ete = (S.where(m)[ete].sum() / O.where(m)[ete].sum()).replace([np.inf, -np.inf], np.nan)
    lignes = []
    for reg in sorted({s.split(":")[0] for s in ids}):
        con = duckdb.connect(f"{DATA}/quebec/{reg}.duckdb", read_only=True)
        st = con.sql("select station_id, node_idx, drainage_area_km2 from stations").df()
        edges = con.sql("select src, dst from edges").df()
        aire = con.sql("select area_km2_local from territorial order by node_idx").fetchnumpy()["area_km2_local"]
        con.close()
        par = defaultdict(list)
        for a, b in zip(edges.src, edges.dst):
            par[int(b)].append(int(a))
        for r in st.itertuples():
            sid = f"{reg}:{r.station_id}"
            if sid not in vol.index:
                continue
            vus, pile = {int(r.node_idx)}, [int(r.node_idx)]
            while pile:
                for p in par.get(pile.pop(), []):
                    if p not in vus:
                        vus.add(p)
                        pile.append(p)
            a_mod = float(aire[np.fromiter(vus, int)].sum())
            lignes.append(dict(station=sid, territoire=reg, aire_station=r.drainage_area_km2, aire_modele=a_mod, rapport_aire=a_mod / r.drainage_area_km2 if r.drainage_area_km2 else np.nan, volume=vol[sid], volume_ete=vol_ete[sid], troncons=len(vus)))
    L = pd.DataFrame(lignes).dropna()
    L.to_csv(f"{DATA}/experiments/plan_ml/aires-stations.csv", index=False)
    print(f"{len(L)} stations ; rapport aire du modèle / aire déclarée : médiane {L.rapport_aire.median():.2f}, quartiles {L.rapport_aire.quantile(0.25):.2f} à {L.rapport_aire.quantile(0.75):.2f} ; hors de 0,8-1,25 : {int(((L.rapport_aire < 0.8) | (L.rapport_aire > 1.25)).sum())} stations")
    print(f"corrélation de rang rapport d'aire ~ volume simulé/observé (2019-2024) : {L[['rapport_aire', 'volume']].corr('spearman').iloc[0, 1]:.2f} ; ~ volume d'été : {L[['rapport_aire', 'volume_ete']].corr('spearman').iloc[0, 1]:.2f}")
    print(L.groupby("territoire").agg(stations=("station", "size"), rapport_aire=("rapport_aire", "median"), hors_bornes=("rapport_aire", lambda v: int(((v < 0.8) | (v > 1.25)).sum())), volume=("volume", "median"), volume_ete=("volume_ete", "median")).round(2).to_string())
    print("\nstations dont l'aire du modèle s'écarte de plus de 25 % :")
    print(L[(L.rapport_aire < 0.8) | (L.rapport_aire > 1.25)].sort_values("rapport_aire").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
