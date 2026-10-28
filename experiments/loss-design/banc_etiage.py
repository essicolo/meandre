"""La fonction objectif voit-elle un prélèvement en étiage.

C'est la question de la cible du projet. Prédire l'effet d'un prélèvement ou d'un rejet sur le
débit d'étiage suppose que la fonction objectif RÉAGISSE à une ponction de cette taille. Si
aucun terme ne la distingue du bruit, le modèle ne pourra jamais l'apprendre, quel que soit
l'entraînement et quel que soit le poids. La question porte sur la perte et se répond sur la
perte, sans simuler.

Les déformations sont celles qui comptent pour un étiage, et non celles des crues :
  prélèvement   une ponction constante retirée des mois de juillet à septembre, exprimée en
                pour cent du débit moyen annuel, de un à dix ;
  récession     la décrue estivale accélérée ou ralentie, ce qui déplace la date de l'étiage
                sans changer son volume ;
  soutien       l'étiage relevé ou creusé d'un facteur, ce qui change son niveau sans toucher
                aux crues.

Le nombre qui décide est la SENSIBILITÉ relative : de combien la note d'un terme augmente,
rapportée à sa valeur sur la série non déformée. Un terme dont la note bouge de moins d'un
pour cent pour un prélèvement de cinq pour cent est aveugle à la grandeur visée.

    .venv/Scripts/python.exe .runs/quebec/banc_etiage.py
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

ETIAGE = (7, 8, 9)

TERMES = [("KGE", "w_kge"), ("biais de volume", "w_pbias"), ("écart quadratique", "w_mse"),
          ("écart quadratique log", "w_log_mse"), ("pics", "w_peak"),
          ("pics par rapport", "w_peak_ratio"), ("calendrier", "w_r"), ("volume", "w_beta"),
          ("amplitude", "w_gamma"), ("variations", "w_dq"), ("soutien d'étiage", "w_fdc_bas"),
          ("vitesse de vidange", "w_recession")]


def deformations(q, mois):
    """Ce qu'un prélèvement, une récession et un soutien font à un hydrogramme."""
    d = {}
    ete = np.isin(mois, ETIAGE)
    for pct in (1, 2, 5, 10):
        s = q.copy()
        s[ete] = np.maximum(s[ete] - pct / 100.0 * q.mean(), 0.01 * q.mean())
        d[f"prélèvement {pct} %"] = s
    for f, nom in ((0.8, "étiage creusé de 20 %"), (1.25, "étiage soutenu de 25 %")):
        s = q.copy()
        bas = q <= np.quantile(q, 0.25)
        s[bas] = s[bas] * f
        d[nom] = s
    for f, nom in ((1.5, "récession estivale accélérée"), (0.7, "récession estivale ralentie")):
        s = q.copy()
        i = 0
        while i < len(q):
            if not ete[i]:
                i += 1
                continue
            j = i
            while j < len(q) and ete[j]:
                j += 1
            bloc = q[i:j]
            dep = bloc[0]
            s[i:j] = dep * (bloc / max(dep, 1e-9)) ** f
            i = j
        d[nom] = s
    return d


def note(cle, q_obs, q_sim, var, q75):
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0, w_peak=0.0,
                w_r=0.0, w_beta=0.0, w_gamma=0.0, w_peak_ratio=0.0, w_recession=0.0,
                per_station=True,
                station_var=torch.tensor([var]), peak_threshold=torch.tensor([q75]))
    f = HydroLoss(**{**base, cle: 1.0})
    with torch.no_grad():
        r = f(q_obs=torch.tensor(q_obs, dtype=torch.float32).unsqueeze(1),
              q_sim=torch.tensor(q_sim, dtype=torch.float32).unsqueeze(1),
              station_mask=torch.ones(1, dtype=torch.bool))
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
            # Reference : une serie legerement bruitee, pour que la note de depart ne soit
            # pas nulle et que la sensibilite relative ait un sens.
            rng = np.random.default_rng(0)
            ref = q * (1.0 + 0.02 * rng.standard_normal(len(q)))
            for nom_t, cle in TERMES:
                n0 = note(cle, q, ref, var, q75)
                for nom_d, s in deformations(ref, mois).items():
                    lignes.append({"terme": nom_t, "deformation": nom_d,
                                   "sensibilite": (note(cle, q, s, var, q75) - n0)
                                                  / max(abs(n0), 1e-12)})
    if not lignes:
        print("aucune station lisible")
        return 1
    t = pd.DataFrame(lignes)
    piv = t.pivot_table(index="terme", columns="deformation", values="sensibilite",
                        aggfunc="median")
    ordre = [c for c in ("prélèvement 1 %", "prélèvement 2 %", "prélèvement 5 %",
                         "prélèvement 10 %", "étiage creusé de 20 %", "étiage soutenu de 25 %",
                         "récession estivale accélérée", "récession estivale ralentie")
             if c in piv.columns]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("Sensibilite relative de chaque terme, mediane sur les stations.")
    print("Valeur NEGATIVE : le terme PREFERE la serie deformee a la reference.\n")
    print(piv[ordre].round(3).to_string())
    print("\nUn terme qui bouge de moins d'un centieme pour un prelevement de cinq pour cent")
    print("est aveugle a la grandeur que le projet doit predire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
