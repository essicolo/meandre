"""Part des tronçons du modèle provincial gelé dont chaque sortie du champ est à moins de 2 % de
l'étendue de sa borne basse ou haute (bornes de FIELD_BOUNDS, sigmoïde douce). Sans simulation.

    .venv/Scripts/python.exe experiments/plan_ml/bornes-collees.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.spatial.field_network import FIELD_BOUNDS
from meandre.utils import paths as _p

r = np.load(f"{_p.DATA_ROOT}/quebec/modele-provincial-2026-10/reach-quebec-gele.npz", allow_pickle=True)
champs = {k[6:]: r[k].astype(float) for k in r.keys() if k.startswith("param_")}
for c in (1, 2, 3):
    champs[f"theta_fc_frac_{c}"] = champs[f"theta_fc_{c}"] / champs[f"porosity_{c}"]
    champs[f"theta_wp_frac_{c}"] = champs[f"theta_wp_{c}"] / champs[f"theta_fc_{c}"]
lignes = []
for nom, v in champs.items():
    cle = nom if nom in FIELD_BOUNDS else nom.rsplit("_", 1)[0]
    if cle not in FIELD_BOUNDS:
        continue
    b = FIELD_BOUNDS[cle]
    lo, hi = b[0], b[1]
    if len(b) > 2 and b[2] == "log":
        v, lo, hi = np.log(v), np.log(lo), np.log(hi)
    e = hi - lo
    lignes.append(dict(champ=nom, borne_basse=b[0], borne_haute=b[1], mediane=float(np.nanmedian(np.exp(v) if len(b) > 2 else v)), part_basse=float(np.mean(v < lo + 0.02 * e)), part_haute=float(np.mean(v > hi - 0.02 * e))))
L = pd.DataFrame(lignes).sort_values(["part_haute", "part_basse"], ascending=False)
L.to_csv(f"{_p.DATA_ROOT}/experiments/plan_ml/bornes-collees.csv", index=False)
with pd.option_context("display.width", 200):
    print(f"{len(v)} tronçons")
    print(L[(L.part_basse > 0.05) | (L.part_haute > 0.05)].round(3).to_string(index=False))
