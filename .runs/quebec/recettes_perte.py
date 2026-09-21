"""Deux recettes de perte mises devant le choix que le modèle fait réellement.

Le banc de perte pose la bonne question : le modèle ne choisit pas entre la vérité et une
déformation, il choisit entre DEUX ERREURS. Il est en retard d'une journée, ce qui est la règle
dès qu'une averse est mal localisée. Deux issues s'offrent alors, rester net et en retard, ou
lisser pour réduire l'écart quotidien. Une perte qui note mieux la version lissée récompense le
nivellement, et aucun entraînement n'y changera rien.

Ce banc compare les recettes sur ce choix, et non terme par terme. Le nombre qui décide est le
rapport entre la note de la version lissée et celle de la version nette en retard : supérieur à
un, la recette préfère le retard au lissage, ce qu'on veut ; inférieur à un, elle paie le
modèle pour aplatir.

Deux recettes, à poids total égal pour que la comparaison porte sur la composition :
  en vigueur   KGE 1,0 | biais 0,5 | écart quadratique 0,1 | écart quad. log 0,3 | pics 0,5
  décomposée   calendrier 1,0 | volume 0,5 | amplitude 0,5 | étiage 0,3 | variations 0,1

    .venv/Scripts/python.exe .runs/quebec/recettes_perte.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

from meandre.training.loss import HydroLoss

_bp = SourceFileLoader("bp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_perte.py")).load_module()

RECETTES = {
    "en vigueur": {"w_kge": 1.0, "w_pbias": 0.5, "w_mse": 0.1, "w_log_mse": 0.3, "w_peak": 0.5},
    "décomposée": {"w_r": 1.0, "w_beta": 0.5, "w_gamma": 0.5, "w_fdc_bas": 0.3, "w_dq": 0.1},
    "décomposée sans étiage": {"w_r": 1.0, "w_beta": 0.5, "w_gamma": 0.5, "w_dq": 0.1},
}


def note(recette, q_obs, q_sim, var, q75):
    """Note totale d'une série sous une recette, tous les autres poids étant nuls."""
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0, w_peak=0.0,
                w_r=0.0, w_beta=0.0, w_gamma=0.0, per_station=True,
                station_var=torch.tensor([var]), peak_threshold=torch.tensor([q75]))
    f = HydroLoss(**{**base, **recette})
    o = torch.tensor(q_obs, dtype=torch.float32).unsqueeze(1)
    s = torch.tensor(q_sim, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        r = f(q_obs=o, q_sim=s, station_mask=torch.ones(1, dtype=torch.bool))
    return float(r[0] if isinstance(r, tuple) else r)


def main():
    lignes = []
    for reg in _bp.REGS:
        f = f"{_bp.DOS}/q-{reg}-A-v4.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        mois = pd.to_datetime([str(v)[:10] for v in z["dates"]]).month.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            var, q75 = float(q.var()), float(np.quantile(q, 0.75))
            # LE CHOIX REEL : net et en retard d'un jour, contre lisse en plus du retard.
            tard = np.concatenate([[q[0]], q[:-1]])
            cand = {"net, en retard": tard}
            for n in (7, 15, 30):
                k = np.ones(n) / n
                cand[f"retard puis lissage {n} j"] = np.convolve(
                    np.pad(tard, (n, n), mode="edge"), k, mode="same")[n:-n]
            # Hiver fige : la plus longue suite plate mesuree sur les sorties reelles.
            gel = tard.copy()
            hiv = np.isin(mois, _bp.HIVER)
            i = 0
            while i < len(gel):
                if not hiv[i]:
                    i += 1
                    continue
                e = i
                while e < len(gel) and hiv[e]:
                    e += 1
                if e - i > 20:
                    gel[i:e] = gel[i] * np.exp(np.log(0.83) * np.arange(e - i) / (e - i))
                i = e
            cand["retard puis hiver figé"] = gel
            for nom_r, recette in RECETTES.items():
                notes = {k: note(recette, q, v, var, q75) for k, v in cand.items()}
                ref = notes["net, en retard"]
                for k, v in notes.items():
                    if k == "net, en retard":
                        continue
                    lignes.append({"recette": nom_r, "region": reg, "station": j,
                                   "defaut": k, "rapport": v / max(ref, 1e-12)})
    if not lignes:
        print("aucune station lisible")
        return 1
    t = pd.DataFrame(lignes)
    print(f"{t.station.nunique()} stations, {t.defaut.nunique()} facons de mal faire\n")
    print("Rapport de la note du defaut a celle de la version nette en retard.")
    print("Superieur a 1 : la recette prefere le retard au defaut, ce qu'on veut.\n")
    piv = t.pivot_table(index="defaut", columns="recette", values="rapport", aggfunc="median")
    piv = piv[[c for c in RECETTES if c in piv.columns]]
    print(piv.round(2).to_string())
    print("\nPart des stations ou la recette PREFERE le defaut au retard (rapport < 1) :")
    mauvais = t.assign(pire=t.rapport < 1.0).pivot_table(
        index="defaut", columns="recette", values="pire", aggfunc="mean")
    print((100 * mauvais[[c for c in RECETTES if c in mauvais.columns]]).round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
