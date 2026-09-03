"""Quel signal chaque sortie du champ spatial recoit-elle vraiment ?

Constat du 2026-09-02 : sur les 42 champs produits par le reseau spatial, seuls K_sat
(x3), k_gw et T_melt varient d'un troncon a l'autre au-dela de 5 %. Les 35 autres sont
des constantes deguisees APRES trente epoques d'entrainement. Une partie est voulue (la
recette du socle impose 23 champs de la courbe de retention), mais il reste une
quinzaine de parametres libres qui ne bougent pas : rugosite du lit, routage, coefficient
cultural, gel, interception, canopee. Le champ de canopee ajoute le 2026-08-28 vaut
exactement 1.50, le milieu de ses bornes : il n'a jamais recu de gradient.

Ce script mesure, sur UN pas de temps differentiable, la norme du gradient de la perte
par rapport a chaque sortie du champ, AVANT contrainte (donc a la sortie de fc_out) et
APRES contrainte (la valeur physique). La comparaison des deux dit lequel des trois cas
s'applique :

  gradient nul avant ET apres    le parametre n'entre pas dans le calcul, ou il est
                                 ecrase par une valeur imposee a l'execution
  gradient apres, pas avant      la contrainte (sigmoide saturee) tue le signal : le
                                 parametre est colle a une borne
  gradient dans les deux         le parametre apprend, et son immobilite vient d'ailleurs
                                 (prior trop fort, ou pas de levier sur la perte)

  PYTHONIOENCODING=utf-8 ETL_REGION=outv python .runs/quebec/audit_gradients.py
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".runs/quebec")

import json
import tomllib

import numpy as np
import pandas as pd
import torch

from meandre.model import HydroModel
from meandre.spatial.field_network import SpatialParams
from meandre.utils.state import HydroState
from ckpt_util import a_des_latents
import joint_data
from et_module import compute_demand

REG = os.environ.get("ETL_REGION", "outv").lower()
CK = os.environ.get("AUDIT_CKPT", ".runs/quebec/checkpoints/best-outv-etl-canon.pt")
N_JOURS = int(os.environ.get("AUDIT_JOURS", "120"))
DEVICE = os.environ.get("AUDIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))

r = joint_data.load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]
n = r["n_nodes"]
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords,
                        r["territorial"], DEVICE) * AD.get(REG, {}).get("debias_et", 1.0)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
               use_temporal=False, use_residual=False, use_travel_time_attn=False,
               use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
               column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
               use_latent_codes=a_des_latents(CK, n), latent_mode="additive",
               spatial_melt=True, routing_mode="operator-lagged", predict_lake_params=True,
               compile_soil=False, use_aquifer=True).to(DEVICE)
m.load(CK)
m.train()
m.vertical_column.etp_channel = 6

# On intercepte la sortie du champ : brute (fc_out) et contrainte (valeur physique).
brut, contraint = {}, {}


def piege_brut(_mod, _entree, sortie):
    sortie.retain_grad()
    brut["x"] = sortie
    return sortie


m.spatial_encoder.fc_out.register_forward_hook(piege_brut)

_fwd = m.spatial_encoder.forward


def fwd_trace(*a, **k):
    # to_tensor() fabrique un NOUVEAU tenseur hors du graphe utilise par la physique :
    # son gradient restait None. On retient le gradient sur chaque champ lui-meme.
    p = _fwd(*a, **k)
    import dataclasses
    for f in dataclasses.fields(p):
        v = getattr(p, f.name)
        if torch.is_tensor(v) and v.requires_grad:
            v.retain_grad()
    contraint["p"] = p
    return p


m.spatial_encoder.forward = fwd_trace

sl = slice(0, N_JOURS)
Q, _ = m.simulate(forcing=f7[sl], initial_state=HydroState.zeros(n, device=DEVICE),
                  graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                  withdrawals=td.withdrawals, day_of_year=td.day_of_year[sl])
qo = td.q_obs[sl][:, :].to(DEVICE) if td.q_obs.dim() == 2 else None
qs = Q[:, td.station_idx]
obs = td.q_obs[sl].to(DEVICE)
masque = torch.isfinite(obs)
perte = ((qs[masque] - obs[masque]) ** 2).mean()
perte.backward()

noms = [f for f in SpatialParams.__dataclass_fields__]
noms = noms[:SpatialParams.N_PARAMS]
gb = brut["x"].grad
lignes = []
for i, nom in enumerate(noms):
    b = float(gb[:, i].abs().mean()) if gb is not None else float("nan")
    _v = getattr(contraint["p"], nom)
    _g = _v.grad if torch.is_tensor(_v) else None
    c = float(_g.abs().mean()) if _g is not None else 0.0
    lignes.append({"champ": nom, "gradient brut": b, "gradient contraint": c})
t = pd.DataFrame(lignes)
t["rapport"] = t["gradient brut"] / t["gradient contraint"].replace(0, np.nan)


def verdict(l):
    if l["gradient contraint"] <= 1e-20:
        return "AUCUN SIGNAL (hors du calcul)"
    if l["gradient brut"] <= 1e-12 * max(l["gradient contraint"], 1e-30):
        return "CONTRAINTE SATUREE (colle a une borne)"
    return "apprend"


t["verdict"] = t.apply(verdict, axis=1)
t = t.sort_values("gradient brut")
pd.set_option("display.width", 160)
print(t.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
print("\nresume :", t.verdict.value_counts().to_dict())
t.to_csv(f"D:/meandre-data/quebec/audit-gradients-{REG}.csv", index=False)
print(f"-> D:/meandre-data/quebec/audit-gradients-{REG}.csv")
