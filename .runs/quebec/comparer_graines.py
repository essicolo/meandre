"""Compare les cartes de paramètres du champ spatial entre points de reprise du banc.

Charge chaque point de reprise dans le même modèle, évalue le champ sur les nœuds du
sous-bassin, et rend pour chaque paramètre sa moyenne sur le bassin par point de reprise,
puis la corrélation de cette moyenne avec la validation de chaque graine. Aucune simulation.

    python .runs/quebec/comparer_graines.py mont 030905 banc-tetes1e16 1234:0.507 777:0.548 4321:0.601 2468:0.268
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from meandre.model import HydroModel
from meandre.utils import paths as _p
from banc_sousbassin import extraire


def main(reg, station, prefixe, graines):
    s = extraire(reg, station)
    terr, coords = s["territorial"], s["node_coords"]
    n = len(s["idx"])
    lignes = []
    for g, v in graines:
        m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6, use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness", use_temperature=False, use_latent_codes=False, spatial_melt=True, routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False, use_aquifer=True)
        ck = torch.load(f"{_p.DATA_ROOT}/quebec/sousbassin/best-{reg}-{station}{prefixe}-{g}.pt", map_location="cpu", weights_only=False)
        etat = ck.get("model_state_dict", ck.get("state_dict", ck))
        manque = m.load_state_dict(etat, strict=False)
        with torch.no_grad():
            sp = m.spatial_encoder(coords, terr.data)
        row = {"graine": g, "validation": v}
        for k, t in vars(sp).items():
            if torch.is_tensor(t) and t.ndim >= 1 and t.shape[0] == n:
                a = t.float().reshape(n, -1).mean(dim=1).numpy()
                row[k] = float(a.mean())
                row[f"cv_{k}"] = float(a.std() / (abs(a.mean()) + 1e-12))
        lignes.append(row)
        print(f"graine {g} chargee, cles manquantes {len(manque.missing_keys)}", flush=True)
    d = pd.DataFrame(lignes).set_index("graine")
    moy = [c for c in d.columns if c != "validation" and not c.startswith("cv_")]
    ecart = {}
    for c in moy:
        x = d[c].to_numpy()
        if np.std(x) == 0:
            continue
        ecart[c] = (float(np.corrcoef(x, d.validation)[0, 1]), float(np.std(x) / (abs(np.mean(x)) + 1e-12)))
    t = pd.DataFrame(ecart, index=["corr_validation", "dispersion_relative"]).T.sort_values("dispersion_relative", ascending=False)
    pd.set_option("display.width", 200)
    print("\nparametres dont la moyenne de bassin varie le plus entre graines :")
    print(t.head(15).round(3).to_string())
    print("\nmoyennes de bassin par graine, ces parametres :")
    print(d[["validation"] + list(t.head(10).index)].round(4).to_string())


if __name__ == "__main__":
    reg, station, prefixe = sys.argv[1], sys.argv[2], sys.argv[3]
    graines = [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[4:]]
    main(reg, station, prefixe, graines)
