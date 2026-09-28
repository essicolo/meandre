"""Identifiabilité des cartes de paramètres entre graines, lue sur les points de reprise.

Un champ est identifiable s'il ne dépend pas de la graine : des initialisations différentes,
entraînées sur les mêmes données, doivent retrouver la même carte. Pour chaque champ et
chaque bras, on mesure entre graines (1) la dispersion relative de la moyenne sur le bassin,
(2) la corrélation moyenne des cartes par paires de graines, et (3) le déplacement depuis
l'initialisation, pour distinguer un champ appris d'un champ resté à son départ.

    python .runs/quebec/identifiabilite_graines.py mont 030905 banc-semi banc-semisiig 1234 4321 777 2468
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataclasses
import itertools

import numpy as np
import pandas as pd
import torch

from meandre.model import HydroModel
from meandre.utils import paths as _p
from banc_sousbassin import extraire


def cartes(reg, station, prefixe, graine, s, extra):
    if extra:
        os.environ["MEANDRE_TERRITORIAL_EXTRA"] = extra
    else:
        os.environ.pop("MEANDRE_TERRITORIAL_EXTRA", None)
    ss = extraire(reg, station) if extra else s
    terr, coords = ss["territorial"], ss["node_coords"]
    n = len(ss["idx"])
    torch.manual_seed(int(graine))
    m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6, use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness", use_temperature=False, use_latent_codes=False, spatial_melt=True, routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False, use_aquifer=True)
    with torch.no_grad():
        init = m.spatial_encoder(coords, terr.data)
    ck = torch.load(f"{_p.DATA_ROOT}/quebec/sousbassin/best-{reg}-{station}{prefixe}-{graine}.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck.get("state_dict", ck)), strict=False)
    with torch.no_grad():
        fin = m.spatial_encoder(coords, terr.data)
    out = {}
    for f in dataclasses.fields(fin):
        a, b = getattr(fin, f.name), getattr(init, f.name)
        if torch.is_tensor(a) and a.ndim >= 1 and a.shape[0] == n:
            out[f.name] = (a.float().reshape(n, -1).mean(dim=1).numpy(), b.float().reshape(n, -1).mean(dim=1).numpy())
    return out


def main(reg, station, prefixes, graines):
    s = extraire(reg, station)
    lignes = []
    for prefixe in prefixes:
        extra = "territorial_siigsol3" if "siig" in prefixe else None
        par_graine = {g: cartes(reg, station, prefixe, g, s, extra) for g in graines}
        for nom in par_graine[graines[0]]:
            fins = np.stack([par_graine[g][nom][0] for g in graines])
            inits = np.stack([par_graine[g][nom][1] for g in graines])
            moy = fins.mean(axis=1)
            depl = np.abs(fins - inits).mean() / (np.abs(inits).mean() + 1e-12)
            cors = []
            for i, j in itertools.combinations(range(len(graines)), 2):
                if fins[i].std() > 0 and fins[j].std() > 0:
                    cors.append(np.corrcoef(fins[i], fins[j])[0, 1])
            lignes.append({"bras": prefixe, "champ": nom, "deplacement": depl, "dispersion_moyenne": moy.std() / (abs(moy.mean()) + 1e-12), "correlation_cartes": np.mean(cors) if cors else np.nan})
    d = pd.DataFrame(lignes)
    appris = d.groupby("champ").deplacement.max()
    appris = appris[appris > 0.02].index
    t = d[d.champ.isin(appris)].pivot(index="champ", columns="bras", values=["deplacement", "dispersion_moyenne", "correlation_cartes"])
    pd.set_option("display.width", 220)
    print(f"champs deplaces de plus de 2 % depuis l'initialisation : {len(appris)} sur {d.champ.nunique()}")
    print(t.round(3).to_string())
    print("\nmedianes sur ces champs :")
    print(d[d.champ.isin(appris)].groupby("bras")[["deplacement", "dispersion_moyenne", "correlation_cartes"]].median().round(3).to_string())


if __name__ == "__main__":
    reg, station = sys.argv[1], sys.argv[2]
    prefixes = [a for a in sys.argv[3:] if not a.isdigit()]
    graines = [a for a in sys.argv[3:] if a.isdigit()]
    main(reg, station, prefixes, graines)
