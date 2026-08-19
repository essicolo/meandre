"""Pieces de la RECETTE D'EXECUTION partagees entre le pilote et les diagnostics.

Un point de reprise ne definit pas un modele : occupation du sol, milieux humides,
phenologie, noyau de versant, surfaces de lac, ancrages ETP/fonte et regle de courbe
de retention sont poses A L'EXECUTION. Chaque fois qu'une de ces pieces a ete recopiee
d'un script a l'autre, elle a fini par diverger, et un diagnostic a mesure autre chose
que le champion :

  2026-08-19, surface de lac : le pilote la corrige par HydroLAKES (+0.015 mesure en
  tenu de cote), les diagnostics ne le faisaient pas -> le champion aq30 ressortait a
  0.7591 au lieu de 0.7880.
  2026-08-19, courbe de retention : regle divergente sur krec (cf.
  hydrotel_calib.courbe_retention_imposee).

Toute piece de recette qui se retrouve dans DEUX fichiers doit atterrir ici.
"""
from __future__ import annotations

import pandas as pd
import torch

RAW_QC = "D:/meandre-data/quebec/territorial-raw-QC.parquet"
HYDROLAKES = "D:/meandre-data/quebec/lacs_hydrolakes.parquet"


def poser_surface_lac(model, reg: str, area_km2_local, n_nodes: int, bavard: bool = True):
    """Corrige la surface des lacs (defaut du pilote depuis le 2026-08-09).

    Le module de lac calcule la hauteur d'eau comme S/A mais recevait l'aire de
    DRAINAGE du troncon, mediane 175x la surface reelle du plan d'eau. Q varie comme
    A^(1-beta) : avec beta = 1.5, surestimer A d'un facteur 175 divise le debit sortant
    par ~13. La surface vient de HydroLAKES (Messager et al. 2016) la ou l'appariement
    existe, sinon de lake_fraction x aire locale.

    Retourne la source utilisee, ou None si les donnees sont absentes.
    """
    A = area_km2_local.cpu().numpy() if torch.is_tensor(area_km2_local) else area_km2_local
    try:
        rw = pd.read_parquet(RAW_QC)
        rw = rw[rw.region == reg]
    except Exception:
        return None
    alac = A * (rw["lake_fraction"].values.clip(0, 1) if len(rw) == n_nodes else 1.0)
    try:
        hl = pd.read_parquet(HYDROLAKES)
        hl = hl[hl.region == reg]
        alac[hl.node_idx.values] = hl["lake_area_km2"].values
        src = f"HydroLAKES ({len(hl)} noeuds) + repli lake_fraction"
    except Exception:
        src = "lake_fraction x aire locale"
    model.set_lake_area(torch.tensor(alac, dtype=torch.float32))
    if bavard:
        print(f"[recette] surface de lac corrigee ({src})")
    return src
