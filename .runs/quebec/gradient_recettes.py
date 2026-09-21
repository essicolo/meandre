"""Dans quelle direction chaque recette POUSSE le débit, au point de départ réel.

Le banc de perte dit ce qu'une recette PRÉFÈRE ; il ne dit pas ce que la descente de gradient
en fait. C'est la seule question qui demandait jusqu'ici d'entraîner. Elle se répond en une
passe avant-arrière : on prend la sortie RÉELLE du modèle aux stations, on calcule la perte de
chaque recette, et on lit le gradient par rapport au débit simulé.

Le signe qui compte est celui du CHANGEMENT DEMANDÉ, c'est-à-dire moins le gradient : positif,
la recette veut que le débit de ce jour-là monte. On résume par niveau de débit observé. Une
recette qui veut plus d'écoulement de base demande de monter les jours d'étiage et de baisser
les jours de crue ; une recette qui aplatit demande l'inverse des deux.

Quelques secondes, contre trois heures pour six entraînements qui répondaient à la même chose.

    .venv/bin/python .runs/quebec/gradient_recettes.py --variante g-temoin-g1234
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.training.loss import HydroLoss

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"

RECETTES = {
    "en vigueur": {"w_kge": 1.0, "w_pbias": 0.5, "w_mse": 0.1, "w_log_mse": 0.3, "w_peak": 0.5},
    "décomposée au jugé": {"w_r": 1.0, "w_beta": 0.5, "w_gamma": 0.5, "w_peak_ratio": 0.5,
                           "w_fdc_bas": 0.3},
    "décomposée équilibrée": {"w_r": 0.15, "w_beta": 1.0, "w_gamma": 0.15, "w_peak_ratio": 0.15,
                              "w_fdc_bas": 1.0, "w_dq": 0.15},
}


def series(region, variante):
    """Débit observé et SIMULÉ à chaque station, depuis une sortie réelle du modèle."""
    f = f"{DERIVES}/reach-{region}-{variante}-journalier.npz"
    r = f"{DERIVES}/reach-{region}-{variante}.npz"
    if not (os.path.exists(f) and os.path.exists(r)):
        return None
    z, meta = np.load(f, allow_pickle=True), np.load(r, allow_pickle=True)
    if "station_idx" not in meta.files:
        return None
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    g = sorted(glob.glob(f"{flotte}/q-{region}-*.npz"))
    if not g:
        return None
    obs = np.load(g[0], allow_pickle=True)["q_obs"]
    n = min(len(z["dates"]), obs.shape[0])
    paires = []
    for j, nd in enumerate(meta["station_idx"]):
        if j >= obs.shape[1] or nd < 0 or nd >= z["q"].shape[1]:
            continue
        o = obs[:n, j].astype(float)
        fini = np.isfinite(o) & (o > 0)
        if fini.sum() < 700:
            continue
        s = z["q"][:n, int(nd)].astype(float)
        if not np.isfinite(s).all() or s.sum() <= 0:
            continue
        paires.append((o[fini], s[fini]))
    return paires


def changement_demande(recette, o, s, q75):
    """Moins le gradient de la perte par rapport au débit simulé, jour par jour."""
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0, w_peak=0.0,
                w_r=0.0, w_beta=0.0, w_gamma=0.0, w_peak_ratio=0.0, per_station=True,
                station_var=torch.tensor([float(o.var())]),
                peak_threshold=torch.tensor([q75]))
    f = HydroLoss(**{**base, **recette})
    qo = torch.tensor(o, dtype=torch.float32).unsqueeze(1)
    qs = torch.tensor(s, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    sortie = f(q_obs=qo, q_sim=qs, station_mask=torch.ones(1, dtype=torch.bool))
    perte = sortie[0] if isinstance(sortie, tuple) else sortie
    perte.backward()
    return -qs.grad.squeeze(1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="outv")
    p.add_argument("--variante", required=True)
    a = p.parse_args()
    paires = series(a.region, a.variante)
    if not paires:
        print(f"aucune serie exploitable pour {a.variante}")
        return 1
    lignes = []
    for o, s in paires:
        q75 = float(np.quantile(o, 0.75))
        bas = o <= np.quantile(o, 0.25)
        haut = o >= np.quantile(o, 0.90)
        for nom, rec in RECETTES.items():
            g = changement_demande(rec, o, s, q75)
            ech = np.abs(g).mean()
            if ech <= 0:
                continue
            lignes.append({"recette": nom,
                           "etiage": float(g[bas].mean() / ech),
                           "crue": float(g[haut].mean() / ech)})
    t = pd.DataFrame(lignes).groupby("recette").median()
    t["contraste"] = t["etiage"] - t["crue"]
    pd.set_option("display.width", 160)
    print(f"{len(paires)} stations, sortie reelle du modele « {a.variante} »\n")
    print("Changement DEMANDE par la recette, normalise par son amplitude moyenne.")
    print("Positif : la recette veut que le debit de ce jour-la MONTE.\n")
    print(t.round(3).to_string())
    print("\nLe contraste est ce qui decide : positif, la recette pousse vers plus")
    print("d'ecoulement de base, en montant les etiages et en baissant les crues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
