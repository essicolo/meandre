"""Paramètres de sol de meandre (champ du modèle provincial gelé, par tronçon) contre ceux du calage
d'Hydrotel (projet LN24HA, agrégés des UHRH au tronçon), Saint-Laurent sud-ouest, Montérégie et
Outaouais. Sans simulation. Conductivités en m/j, épaisseurs en m.

    .venv/Scripts/python.exe experiments/plan_ml/ete-parametres-sol.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.data.basin_cache import BasinCache
from meandre.data.hydrotel_calib import load_calibrated_soil
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
MODELE = f"{_p.DATA_ROOT}/quebec/modele-provincial-2026-10"


def main():
    r = np.load(f"{MODELE}/reach-quebec-gele.npz", allow_pickle=True)
    raw = pd.read_parquet(f"{_p.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
    raw = raw[raw.region != "vaud"].reset_index(drop=True)
    lignes = []
    for reg in ("slso", "mont", "outv"):
        m = (raw.region == reg).to_numpy()
        ids = BasinCache(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb").load(device=torch.device("cpu"))["node_ids"]
        h = load_calibrated_soil(f"{_p.PLATFORMS_ROOT}/LN24HA/{reg.upper()}_LN24HA_2020", ids, z1_fixed=0.1)
        assert m.sum() == len(ids)
        H = {k: v.numpy() for k, v in h.items() if hasattr(v, "numpy") and v.ndim == 1}
        for nom, mv, hv in (("K_sat couche 1 (m/j)", r["param_K_sat_1"][m], H["ks1"] * 24), ("K_sat couche 2 (m/j)", r["param_K_sat_2"][m], H["ks2"] * 24), ("K_sat couche 3 (m/j)", r["param_K_sat_3"][m], H["ks3"] * 24), ("porosité couche 1", r["param_porosity_1"][m], H["thetas1"]), ("porosité couche 2", r["param_porosity_2"][m], H["thetas2"]), ("épaisseur couche 2 (m)", r["param_Z2"][m], H["z2"]), ("épaisseur couche 3 (m)", r["param_Z3"][m], H["z3"]), ("épaisseur couche 1 (m)", np.full(m.sum(), np.nan), H["z1"]), ("capacité au champ couche 1", r["param_theta_fc_1"][m], H["thetacc1"])):
            lignes.append(dict(territoire=reg, parametre=nom, meandre_p10=np.nanquantile(mv, 0.1), meandre_med=np.nanmedian(mv), meandre_p90=np.nanquantile(mv, 0.9), hydrotel_p10=np.nanquantile(hv, 0.1), hydrotel_med=np.nanmedian(hv), hydrotel_p90=np.nanquantile(hv, 0.9)))
    L = pd.DataFrame(lignes)
    L.to_csv(f"{DOSSIER}/ete-parametres-sol.csv", index=False)
    print(L.set_index(["parametre", "territoire"]).round(3).sort_index().to_string())


if __name__ == "__main__":
    main()
