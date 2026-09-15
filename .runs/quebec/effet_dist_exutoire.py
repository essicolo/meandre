"""Que change la correction de la distance a l'exutoire aux parametres predits ?

La colonne `dist_to_outlet_km` est l'un des seize canaux que le champ spatial consomme, et
elle est cumulee sur une longueur d'arete qui contient l'AIRE des lacs (mesure du
2026-09-15). On compare les parametres predits par un point de reprise avec la valeur
actuelle puis avec la valeur corrigee, tout le reste egal. Passe avant seule, sur le
processeur central.

    .venv/Scripts/python.exe .runs/quebec/effet_dist_exutoire.py sagu
"""
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.basin_cache import BasinCache
from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

PLATE = "LN24HA"
CKPT = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
        "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}


def distance_corrigee(reg, nd, ed):
    p = Path(f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020/physitel/troncon.trl")
    lg = {t["id"]: t["length_m"] for t in _parse_troncon(p)}
    nid = dict(zip(nd.node_idx.astype(int), nd.node_id.astype(int)))
    n = len(nd)
    lgn = np.zeros(n)
    for s in ed.src.values.astype(int):
        lgn[s] = lg.get(nid.get(s), 0.0)
    aval = -np.ones(n, dtype=int)
    aval[ed.src.values.astype(int)] = ed.dst.values.astype(int)
    d = np.zeros(n)
    for j in nd.sort_values("topo_order", ascending=False).node_idx.values.astype(int):
        k = aval[j]
        d[j] = lgn[j] + (d[k] if k >= 0 else 0.0)
    return d / 1000.0


def main(reg):
    from meandre.model import HydroModel
    ck = f".runs/quebec/checkpoints/{CKPT.get(reg, 'best-gasp-etl-ds')}.pt"
    if not os.path.exists(ck):
        print(f"point de reprise absent : {ck}")
        return 1
    base = _paths.data_path("quebec", f"{reg}.duckdb")
    h = BasinCache(base).load(device="cpu")
    terr = h["territorial"]
    cols = list(terr.columns)
    if "dist_to_outlet_km" not in cols:
        print("la distance à l'exutoire n'est pas un canal du champ")
        return 1
    j = cols.index("dist_to_outlet_km")
    cx = duckdb.connect(base, read_only=True)
    nd = cx.sql("select node_idx, node_id, topo_order from nodes order by node_idx").fetchdf()
    ed = cx.sql("select src, dst from edges").fetchdf()
    brut = cx.sql("select node_idx, dist_to_outlet_km from territorial order by node_idx").fetchdf()
    cx.close()
    d_neuf = distance_corrigee(reg, nd, ed)
    d_vieux = brut.dist_to_outlet_km.values.astype(float)
    print(f"{reg} : distance actuelle méd {np.median(d_vieux):.0f} km, corrigée "
          f"{np.median(d_neuf):.0f} km")

    m = HydroModel(n_nodes=h["n_nodes"], n_territorial=terr.n_features, n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False,
                   use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   spatial_melt=True, routing_mode="operator-lagged",
                   predict_lake_params=True, compile_soil=False, use_aquifer=True)
    m.load(ck)
    m.eval()
    coords = h["node_coords"]

    # La table territoriale stocke le canal DEJA centre et reduit : sa mediane vaut zero.
    # Le canal corrige doit donc etre centre et reduit par SES PROPRES statistiques, sinon
    # on injecte des kilometres bruts a la place d'un score standard, soit une perturbation
    # de cent trente ecarts-types, et l'on mesure cette bevue au lieu de la correction.
    x0 = terr.data.clone()
    x1 = x0.clone()
    _z = (d_neuf - d_neuf.mean()) / (d_neuf.std() + 1e-9)
    x1[:, j] = torch.tensor(_z, dtype=torch.float32)
    print(f"canal actuel : moyenne {float(x0[:, j].mean()):+.3f}, écart-type "
          f"{float(x0[:, j].std()):.3f} | canal corrigé : moyenne {_z.mean():+.3f}, "
          f"écart-type {_z.std():.3f}")
    print(f"corrélation de rang entre les deux canaux : "
          f"{pd.Series(x0[:, j].numpy()).corr(pd.Series(_z), method='spearman'):.3f}")
    with torch.no_grad():
        p0 = m.spatial_encoder(coords, x0)
        p1 = m.spatial_encoder(coords, x1)
    lignes = []
    for k in p0.__dataclass_fields__:
        a, b = getattr(p0, k), getattr(p1, k)
        if not torch.is_tensor(a) or a.shape[:1] != (h["n_nodes"],):
            continue
        a, b = a.float(), b.float()
        if float(a.std()) < 1e-12 and float(a.mean()) == 0:
            continue
        rel = float((b - a).abs().median() / (a.abs().median() + 1e-12))
        lignes.append({"paramètre": k, "médiane actuelle": float(a.median()),
                       "médiane corrigée": float(b.median()),
                       "écart relatif médian %": 100 * rel})
    d = pd.DataFrame(lignes).sort_values("écart relatif médian %", ascending=False)
    print(d.head(14).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n{len(d)} paramètres ; écart relatif médian sur l'ensemble "
          f"{d['écart relatif médian %'].median():.2f} pour cent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu"))
