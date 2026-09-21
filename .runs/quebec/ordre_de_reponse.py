"""À quel ORDRE chaque terme de la perte répond-il à une petite erreur.

Le défaut a été trouvé trois fois en deux jours, sur des termes écrits à des dates et par des
mains différentes : les facteurs du KGE, le soutien d'étiage, et le terme de pics par rapport.
Tous pénalisaient le CARRÉ d'un écart relatif, donc répondaient au second ordre. Pour une
erreur relative de e petite, une forme du premier ordre varie comme e et une du second comme
e² ; sur des erreurs de quelques pour cent, le facteur entre les deux se compte en dizaines.

Trois fois n'est pas une coïncidence, c'est une famille. Ce banc mesure l'ordre de TOUS les
termes, plutôt que d'attendre de tomber sur le quatrième.

La mesure ne suppose rien de la forme. On applique une déformation d'amplitude e, puis une
d'amplitude 2e, et on lit l'exposant p tel que la perte varie comme e^p : il vaut le logarithme
en base deux du rapport des deux variations. Un terme du premier ordre rend p proche de 1, un
terme du second ordre p proche de 2.

    .venv/Scripts/python.exe .runs/quebec/ordre_de_reponse.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

_bp = SourceFileLoader("bp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_perte.py")).load_module()
_be = SourceFileLoader("be", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_etiage.py")).load_module()

TERMES = _be.TERMES

# Deformations elementaires, chacune ne touchant QU'UNE dimension de l'erreur, pour que
# l'ordre mesure la forme du terme et non un melange.
def deformations(q, mois, e):
    """Trois erreurs d'amplitude relative e, sur le volume, l'amplitude et l'etiage."""
    ete = np.isin(mois, (7, 8, 9))
    volume = q * (1.0 + e)
    amplitude = q.mean() + (q - q.mean()) * (1.0 - e)
    etiage = q.copy()
    etiage[ete] = np.maximum(etiage[ete] - e * q.mean(), 0.01 * q.mean())
    pointes = q.copy()
    haut = q >= np.quantile(q, 0.75)
    pointes[haut] = pointes[haut] * (1.0 - e)
    return {"volume": volume, "amplitude": amplitude, "étiage": etiage, "pointes": pointes}


def main():
    petit, grand = 0.02, 0.04
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
            d1 = deformations(q, mois, petit)
            d2 = deformations(q, mois, grand)
            for nom_t, cle in TERMES:
                n0 = _be.note(cle, q, q, var, q75)
                for nom_d in d1:
                    a = _be.note(cle, q, d1[nom_d], var, q75) - n0
                    b = _be.note(cle, q, d2[nom_d], var, q75) - n0
                    if a <= 1e-12 or b <= 1e-12:
                        continue
                    lignes.append({"terme": nom_t, "deformation": nom_d,
                                   "ordre": np.log2(b / a)})
    if not lignes:
        print("aucune station lisible")
        return 1
    t = pd.DataFrame(lignes)
    piv = t.pivot_table(index="terme", columns="deformation", values="ordre", aggfunc="median")
    pd.set_option("display.width", 200)
    print(f"Ordre de reponse, mediane sur les stations. Erreur doublee de {petit:.0%} a {grand:.0%}.")
    print("1 = premier ordre, la perte double. 2 = second ordre, elle quadruple.")
    print("Une case vide signifie que le terme est INSENSIBLE a cette deformation.\n")
    print(piv.round(2).to_string())
    suspects = piv[(piv > 1.6).any(axis=1)].index.tolist()
    print("\nTermes repondant au second ordre sur au moins une deformation :")
    print("  " + (", ".join(suspects) if suspects else "aucun"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
