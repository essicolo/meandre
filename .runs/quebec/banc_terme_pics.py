"""Le terme de pics est-il récupérable, et sous quelle forme.

Mesuré le 2026-09-20 : le terme de pics en vigueur RÉCOMPENSE l'aplatissement. C'est un écart
quadratique restreint aux hauts débits, donc à l'endroit exact où un décalage d'une journée
coûte le plus cher ; lisser y réduit l'erreur plus qu'ailleurs, et il paie pour cela sur 78 %
des stations. Le terme ajouté pour protéger les pics les détruit.

Avant de le retirer ou de le remplacer, la question se pose sur la fonction elle-même. Un terme
qui compare des STATISTIQUES de pointe, et non des écarts jour par jour, ne peut pas être dupé
par un décalage : lisser réduit la pointe simulée et le terme le voit. Ce banc met quatre
formes devant le même choix que le modèle affronte, rester net et en retard ou lisser.

Le nombre qui décide est la part des stations où la forme PRÉFÈRE le lissage. Une forme qui
dépasse quelques pour cent est disqualifiée, quel que soit son poids.

    .venv/Scripts/python.exe .runs/quebec/banc_terme_pics.py
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


def pics_en_vigueur(o, s, q75):
    """Forme actuelle : écart quadratique au-dessus du troisième quartile observé."""
    f = HydroLoss(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                  w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0, w_peak=1.0,
                  per_station=True, station_var=torch.tensor([float(o.var())]),
                  peak_threshold=torch.tensor([q75]))
    with torch.no_grad():
        r = f(q_obs=torch.tensor(o, dtype=torch.float32).unsqueeze(1),
              q_sim=torch.tensor(s, dtype=torch.float32).unsqueeze(1),
              station_mask=torch.ones(1, dtype=torch.bool))
    return float(r[0] if isinstance(r, tuple) else r)


def rapport_des_pointes(o, s, q75, quantile=0.95):
    """Rapport des moyennes au-dessus d'un quantile OBSERVÉ, pénalisé en carré du logarithme.

    Le seuil vient de l'observation et ne bouge pas avec la simulation, sans quoi un modèle
    plat déplacerait son propre seuil et le terme ne verrait rien. Le logarithme rend la
    pénalité symétrique entre sur-estimation et sous-estimation.
    """
    seuil = np.quantile(o, quantile)
    haut = o >= seuil
    if haut.sum() < 10:
        return np.nan
    rapport = s[haut].mean() / max(o[haut].mean(), 1e-12)
    return float(np.log(max(rapport, 1e-6)) ** 2)


def rapport_des_maxima_annuels(o, s, mois, annees):
    """Rapport des maxima annuels moyens, la grandeur des ouvrages et des crues."""
    cles = np.unique(annees)
    if len(cles) < 3:
        return np.nan
    mo = np.array([o[annees == a].max() for a in cles])
    ms = np.array([s[annees == a].max() for a in cles])
    return float(np.log(max(ms.mean() / max(mo.mean(), 1e-12), 1e-6)) ** 2)


def ecart_type_des_hauts(o, s, quantile=0.90):
    """Rapport des écarts-types au-dessus d'un quantile observé : la NERVOSITÉ des crues."""
    seuil = np.quantile(o, quantile)
    haut = o >= seuil
    if haut.sum() < 10:
        return np.nan
    return float(np.log(max(s[haut].std() / max(o[haut].std(), 1e-12), 1e-6)) ** 2)


def main():
    lignes = []
    for reg in _bp.REGS:
        f = f"{_bp.DOS}/q-{reg}-A-v4.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        dts = pd.to_datetime([str(v)[:10] for v in z["dates"]])
        mois, annees = dts.month.to_numpy(), dts.year.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            q75 = float(np.quantile(q, 0.75))
            tard = np.concatenate([[q[0]], q[:-1]])
            for n in (7, 15):
                k = np.ones(n) / n
                liss = np.convolve(np.pad(tard, (n, n), mode="edge"), k, mode="same")[n:-n]
                formes = {
                    "en vigueur (écart quadratique)":
                        (pics_en_vigueur(q, tard, q75), pics_en_vigueur(q, liss, q75)),
                    "rapport des pointes (q95)":
                        (rapport_des_pointes(q, tard, q75), rapport_des_pointes(q, liss, q75)),
                    "rapport des maxima annuels":
                        (rapport_des_maxima_annuels(q, tard, mois, annees),
                         rapport_des_maxima_annuels(q, liss, mois, annees)),
                    "nervosité des hauts débits":
                        (ecart_type_des_hauts(q, tard), ecart_type_des_hauts(q, liss)),
                }
                for nom, (a, b) in formes.items():
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    lignes.append({"forme": nom, "lissage": f"{n} j", "retard": a, "lisse": b,
                                   "prefere_le_lissage": bool(b < a)})
    if not lignes:
        print("aucune station lisible")
        return 1
    t = pd.DataFrame(lignes)
    print(f"{len(t) // t.forme.nunique() // t.lissage.nunique()} stations\n")
    print("Part des stations ou la forme PREFERE le lissage au retard. Plus petit est mieux.")
    piv = t.pivot_table(index="forme", columns="lissage", values="prefere_le_lissage",
                        aggfunc="mean")
    print((100 * piv).round(0).to_string())
    print("\nUne forme qui depasse quelques pour cent est disqualifiee, quel que soit son poids :")
    print("elle paie le modele pour aplatir exactement la ou on voulait le proteger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
