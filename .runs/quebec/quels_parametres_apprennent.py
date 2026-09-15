"""Quels parametres le champ apprend-il REELLEMENT, et lesquels restent a leur depart ?

Le 2026-09-15 au matin, la ligne de `fc_out` qui produit le temps de transfert Muskingum
avait une norme de 0,058 contre 0,878 pour la conductivite a saturation : le routage etait
reste a son initialisation. La meme mesure, faite sur les quarante-deux sorties, dit lequel
de ces parametres le champ differencie d'un troncon a l'autre et lequel il laisse constant.

Elle eclaire en particulier la question des plateaux d'hiver. Le banc de perte etablit que
la recette REJETTE un hiver fige, +49,9 pour cent, et ne le prefere sur aucune station :
les plateaux ne sont donc pas recompenses par la perte. S'ils persistent, c'est que le
modele ne sait pas produire un hiver mouvant, et les parametres du gel sont les premiers
suspects.

    .venv/Scripts/python.exe .runs/quebec/quels_parametres_apprennent.py best-mont-etl-ds
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.spatial.field_network import SpatialParams

HIVER = {"diff_gel", "fs_neige", "T_melt", "C_f", "seuil_neige", "dT_canopee_feu",
         "dT_canopee_conif", "fonte_min"}


def main(nom):
    f = f".runs/quebec/checkpoints/{nom}.pt"
    if not os.path.exists(f):
        print(f"point de reprise absent : {f}")
        return 1
    sd = torch.load(f, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    W = sd.get("spatial_encoder.fc_out.weight")
    if W is None:
        print("pas de couche de sortie du champ dans ce point de reprise")
        return 1
    noms = [k for k in SpatialParams.__dataclass_fields__][:getattr(SpatialParams, "N_PARAMS", W.shape[0])]
    n = min(len(noms), W.shape[0])
    lignes = [{"parametre": noms[i], "norme": float(W[i].norm()),
               "hiver": noms[i] in HIVER} for i in range(n)]
    d = pd.DataFrame(lignes).sort_values("norme")
    med = d.norme.median()
    print(f"{nom} : {W.shape[0]} sorties, norme médiane {med:.4f}\n")
    print("les dix plus FIGÉS (le champ ne les différencie pas)")
    print(d.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nles huit plus APPRIS")
    print(d.tail(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    h = d[d.hiver]
    if len(h):
        print(f"\nparamètres d'HIVER : norme médiane {h.norme.median():.4f}, "
              f"soit {h.norme.median() / med:.2f} fois la médiane générale")
        print(h.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsorties dont la norme est sous un dixième de la médiane : "
          f"{int((d.norme < 0.1 * med).sum())} sur {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "best-mont-etl-ds"))
