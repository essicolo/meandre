"""Prépare, sans simulation, les séries de l'analyse du résidu aux stations.

Débit simulé et observé : sorties journalières du modèle provincial gelé aux 170 stations,
2019-2024 (`q-quebec-proba.npz`, écrit par le pilote, ETL_DUMP_Q). Le modèle a été entraîné
sur 2001-2018 : les six années sont hors entraînement, 2019-2021 servaient à la validation.

Forçage : moyenne sur le bassin en amont de chaque station, pondérée par l'aire locale des
tronçons, du forçage CaSR brut du modèle (`[forcing].suffix = "-casr-brut"`), 2000-2024.

Attributs : ceux du tronçon exutoire dans la table territoriale (valeurs TRANSFORMÉES pour le
réseau, pas des unités physiques ; `aire_km2` vient de la table des stations), et les
paramètres du champ spatial au même tronçon (`reach-quebec-gele.npz`).

    .venv/Scripts/python.exe experiments/gru_residual/donnees.py
"""
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _p

MODELE = f"{_p.DATA_ROOT}/quebec/modele-provincial-2026-10"
SORTIE = f"{_p.DATA_ROOT}/experiments/gru_residual"


def amont(graph, depart):
    """Indices des tronçons en amont d'un tronçon, lui compris."""
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()
    pere = defaultdict(list)
    for s, t in zip(src, dst):
        pere[int(t)].append(int(s))
    vus, q = {int(depart)}, deque([int(depart)])
    while q:
        u = q.popleft()
        for v in pere[u]:
            if v not in vus:
                vus.add(v)
                q.append(v)
    return np.array(sorted(vus))


def main():
    os.makedirs(SORTIE, exist_ok=True)
    z = np.load(f"{MODELE}/q-quebec-proba.npz", allow_pickle=True)
    ids = [str(s) for s in z["station_ids"]]
    rz = np.load(f"{MODELE}/reach-quebec-gele.npz", allow_pickle=True)
    champs = {k[6:]: rz[k][rz["station_idx"]] for k in rz.files if k.startswith("param_")}

    par_reg = defaultdict(list)
    for k, s in enumerate(ids):
        reg, st = s.split(":")
        par_reg[reg].append((k, st))

    forcage, attributs = {}, []
    for reg, liste in sorted(par_reg.items()):
        base = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
        d = BasinCache(base).load(device=torch.device("cpu"))
        terr = d["territorial"]
        aire_loc = terr.physical["area_km2_local"].numpy()
        con = duckdb.connect(base, read_only=True)
        noeuds = dict(con.execute("select station_id, node_idx from stations").fetchall())
        # Les colonnes de la table territoriale sont TRANSFORMÉES pour le réseau (l'aire y vaut
        # 1,3 pour 1 300 km²) : l'aire en km² vient de la table des stations.
        aires = dict(con.execute("select station_id, drainage_area_km2 from stations").fetchall())
        con.close()
        fx = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-casr-brut.nc")
        temps = pd.DatetimeIndex(fx.time.values)
        variables = [str(v) for v in fx["var"].values]
        for k, st in liste:
            idx = amont(d["graph"], noeuds[st])
            w = aire_loc[idx] / aire_loc[idx].sum()
            f = fx["forcing"].isel(node=idx).values            # (T, n, var)
            moy = np.einsum("tnv,n->tv", f, w)
            forcage[ids[k]] = pd.DataFrame(moy, index=temps, columns=variables)
            a = {c: float(terr.data[noeuds[st], j]) for j, c in enumerate(terr.columns)}
            a.update(station=ids[k], region=reg, n_troncons=len(idx), aire_km2=float(aires[st] or 0.0))
            a.update({f"champ_{c}": float(v[k]) for c, v in champs.items()})
            attributs.append(a)
        fx.close()
        print(f"{reg} : {len(liste)} stations", flush=True)

    dates = pd.DatetimeIndex(z["dates"])
    q = pd.concat({"q_sim": pd.DataFrame(z["q_sim"], index=dates, columns=ids),
                   "q_obs": pd.DataFrame(z["q_obs"], index=dates, columns=ids)}, axis=1)
    q.to_parquet(f"{SORTIE}/debits.parquet")
    pd.concat(forcage, axis=1).to_parquet(f"{SORTIE}/forcage_bassin.parquet")
    pd.DataFrame(attributs).set_index("station").to_parquet(f"{SORTIE}/attributs.parquet")
    print(f"écrit dans {SORTIE}")


if __name__ == "__main__":
    main()
