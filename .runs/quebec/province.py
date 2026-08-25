"""LA PROVINCE : un modele, un domaine, une boucle.

Remplace `joint.py` pour le passage a l'echelle. joint.py gardait les plateformes comme
unites de CALCUL et les faisait tourner en rotation : un entraineur et une simulation par
plateforme et par epoch. Le modele etant limite par le lancement de noyaux, le cout
suivait le NOMBRE de plateformes -- mesure le 2026-08-25 : plus de 5 h 36 pour un epoch 0
inacheve, GPU a 0-1 %. Ici les quatorze caches sont fondus en un domaine unique
(`domain_data.load_domain`) et l'epoch est UNE boucle sur 25 656 troncons vectorises.

Recette 1.0, sans aucun bouton regional : partage pluie-neige au bulbe humide a -0.8
(R35/R43), fonte saisonniere d'amplitude 0.5 (R36), krec appris par le champ avec la
moyenne ancree a 2e-5 (R34), prior k_gw a 0.0273 mesure sur 1316 recessions.

    PROV_EPOCHS=4 PROV_CHUNK=45 python .runs/quebec/province.py
    PROV_EPOCHS=4 python .runs/quebec/province.py gasp mont slno abit   # sous-domaine

La tenue de cote 2022-2024 est rapportee PAR PLATEFORME, non parce que la plateforme
signifie quelque chose pour le modele -- elle n'est plus qu'une tranche d'indices -- mais
parce que la flotte gen1 du 2026-08-23 a mesure ses references territoire par territoire
et que c'est contre elles qu'il faut se comparer.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import tomllib

import numpy as np
import torch

from domain_data import load_domain
from meandre.model import HydroModel
from meandre.training.trainer import Trainer, TrainingConfig
from meandre.utils.metrics import kge as kge_fn
from meandre.utils.state import HydroState

PLATEFORMES = [a.lower() for a in sys.argv[1:]] or [
    "outv", "gasp", "mont", "sagu", "slno", "abit", "slso",
    "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "vaud"]
TAG = os.environ.get("PROV_TAG", "province")
N_EPOCHS = int(os.environ.get("PROV_EPOCHS", "4"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CFG = ".runs/quebec/config/gasp-v4.toml"
CKPT = f".runs/quebec/checkpoints/best-{TAG}.pt"

cfg = tomllib.load(open(BASE_CFG, "rb"))
lcfg = dict(cfg["loss"]); tcfg = cfg["training"]; mcfg = cfg["model"]

print(f"[province] plateformes fondues : {PLATEFORMES} | device {DEVICE}", flush=True)
dom = load_domain(PLATEFORMES, lcfg, device=DEVICE)

model = HydroModel(
    n_nodes=dom["n_nodes"],
    n_territorial=dom["territorial"].n_features,
    n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=bool(mcfg.get("use_frost_rankinen", True)),
    column_theta_init_frac=float(mcfg.get("column_theta_init_frac", 0.9)),
    param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
    use_temperature=False,
    # z_n desactive : un effet aleatoire par troncon sur 25 656 troncons ajouterait
    # autant de parametres libres que de noeuds sans contrainte spatiale, et c'est
    # exactement ce que le champ continu doit porter a sa place.
    use_latent_codes=False,
    spatial_melt=bool(mcfg.get("spatial_melt", True)),
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=bool(mcfg.get("predict_lake_params", True)),
    compile_soil=bool(mcfg.get("compile_soil", True)),
    use_aquifer=True,
).to(DEVICE)

# Occupation du sol, milieux humides et fonte calée, livrés par plateforme et fondus
# par le chargeur. Sans eux la physique reçoit 0 % de forêt et 0 % d'eau libre, et
# OUTV perd 0.27 de KGE (mesure du 2026-08-10). joint.py ne les posait pas.
if dom.get("land_cover"):
    model.vertical_column.set_land_cover(dom["land_cover"])
if dom.get("melt_params"):
    model.vertical_column.set_melt_params(dom["melt_params"])
if dom.get("phenology"):
    model.vertical_column.set_phenology(dom["phenology"])

_lp = dict(cfg.get("literature_prior") or {})
_lp["K_sat_1"] = 0.04; _lp["K_c"] = 1.0; _lp["k_gw"] = 0.07; _lp.setdefault("krec", 5e-5)
model.spatial_encoder.init_from_literature(_lp)
model.vertical_column.split_mode = "wet_bulb"
model.vertical_column.t_neige_seuil = float(os.environ.get("PROV_TWB", "-0.8"))
model.vertical_column.melt_seasonal_amp = float(os.environ.get("PROV_AMP", "0.5"))
model.spatial_encoder.prior_on_krec = True
_t = getattr(model.spatial_encoder, "_prior_targets", None) or {}
_t["krec"] = 2e-5; _t["k_gw"] = 0.0273
model.spatial_encoder._prior_targets = _t
print(f"[province] recette 1.0 : bulbe humide {model.vertical_column.t_neige_seuil:+.2f}, "
      f"fonte saisonniere {model.vertical_column.melt_seasonal_amp}, krec ancre 2e-5, "
      f"k_gw 0.0273 | {sum(p.numel() for p in model.parameters()):,} parametres", flush=True)
if "PROV_WARM" in os.environ:
    model.load(os.environ["PROV_WARM"])
    print(f"[province] warm-start depuis {os.environ['PROV_WARM']}", flush=True)

tcfg_obj = TrainingConfig(
    lr=float(os.environ.get("PROV_LR", tcfg.get("lr", 3e-4))),
    n_epochs=N_EPOCHS,
    chunk_steps=int(os.environ.get("PROV_CHUNK", tcfg.get("chunk_steps", 45))),
    tbptt_steps=int(tcfg.get("tbptt_steps", 365)),
    grad_clip=float(tcfg.get("grad_clip", 1.0)),
    best_metric=tcfg.get("best_metric", "kge_median"),
    patience=int(tcfg.get("patience", 0)),
)
trainer = Trainer(model, dom["loss_fn"], train_data=dom["train_data"],
                  val_data=dom["val_data"], config=tcfg_obj, checkpoint_path=CKPT)
trainer.fit()

# ── tenue de cote 2022-2024, rapportee par plateforme ────────────────────────
print(f"\n[province] HELD-OUT 2022-2024", flush=True)
if os.path.exists(CKPT):
    model.load(CKPT)
model.eval()
td = dom["train_data"]
with torch.no_grad():
    Q, _ = model.simulate(forcing=td.forcing[:],
                          initial_state=HydroState.zeros(dom["n_nodes"], device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords,
                          territorial=td.territorial, withdrawals=td.withdrawals,
                          day_of_year=td.day_of_year)
times = dom["times"]
sl = (times >= "2022-01-01") & (times <= "2024-12-31")
q_obs_full = dom["val_data"].q_obs
kges = {}
for j, sid in enumerate(dom["station_ids"]):
    plate = sid.split(":")[0]
    node = int(td.station_idx[j])
    o = q_obs_full[-int(sl.sum()):, j] if q_obs_full.shape[0] >= int(sl.sum()) else None
    if o is None:
        continue
    s = Q[sl][:, node]
    m = torch.isfinite(o) & torch.isfinite(s)
    if int(m.sum()) < 60:
        continue
    kges.setdefault(plate, []).append(float(kge_fn(o[m], s[m])))
GEN1 = {"gasp": 0.7134, "mont": 0.6953, "slno": 0.7631, "sagu": 0.7438, "abit": 0.5244}
tous = []
for plate in PLATEFORMES:
    v = kges.get(plate, [])
    tous += v
    if v:
        ref = GEN1.get(plate)
        ecart = f" | gen1 {ref:.4f} ({np.median(v) - ref:+.4f})" if ref else ""
        print(f"  {plate:6s} n={len(v):3d} | kge median {np.median(v):.4f}{ecart}", flush=True)
    else:
        print(f"  {plate:6s} aucune station evaluable", flush=True)
if tous:
    print(f"\n[province] mediane provinciale {np.median(tous):.4f} sur {len(tous)} stations",
          flush=True)
os._exit(0)
