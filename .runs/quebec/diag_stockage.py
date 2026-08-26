"""GRACE contre le stockage simule : l'ecart est-il RECONCILIABLE ?

Motif (2026-08-26). Les termes GRACE ont pris le controle de l'optimisation provinciale :
la perte d'entrainement est passee de 4.4 a 48.9 en huit epochs pendant que la mediane
par station tombait de 0.3662 a 0.2756, et la tenue de cote finale de 0.6193 a 0.4518.
Borner les z-scores (Huber, delta 3) N'A PAS SUFFI : une borne empeche le gradient
d'exploser, pas de pointer toujours dans le meme sens. Si le modele ne peut pas
satisfaire la contrainte, un gradient borne mais constant pousse indefiniment.

D'ou la question qu'il fallait poser en premier, et qu'aucun reglage de poids ne
remplace : l'ecart est-il d'AMPLITUDE, de PHASE, ou de DERIVE ? Les trois appellent des
remedes opposes. Une amplitude simulee trop grande veut dire que la colonne respire trop
et qu'aucun poids ne corrigera sans casser le debit. Une phase decalee se corrige par la
dynamique (recharge, vidange). Une derive est un probleme de reference, pas de physique.

    python .runs/quebec/diag_stockage.py <checkpoint.pt> [plateformes...]

Aucun entrainement : une simulation, puis des statistiques. Rapporte par bassin le
rapport d'amplitude, le decalage de phase en mois, la correlation, et la climatologie
mensuelle des deux series cote a cote.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import tomllib
import torch

from domain_data import load_domain
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "D:/meandre-data/quebec/runpod/best-province.pt"
PLATEFORMES = [a.lower() for a in sys.argv[2:]] or ["gasp", "mont", "sagu", "outv"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
mcfg = cfg["model"]

print(f"[diag] {PLATEFORMES} | checkpoint {os.path.basename(CKPT)}", flush=True)
dom = load_domain(PLATEFORMES, dict(cfg["loss"]), device=DEVICE)

model = HydroModel(
    n_nodes=dom["n_nodes"], n_territorial=dom["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, spatial_melt=True,
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=True, compile_soil=False, use_aquifer=True).to(DEVICE)
if dom.get("land_cover"):
    model.vertical_column.set_land_cover(dom["land_cover"])
if dom.get("melt_params"):
    model.vertical_column.set_melt_params(dom["melt_params"])
if dom.get("linacre"):
    model.vertical_column.set_linacre_params(*dom["linacre"])
    model.vertical_column.etp_channel = None
model.vertical_column.split_mode = "wet_bulb"
model.vertical_column.t_neige_seuil = -0.8
model.vertical_column.melt_seasonal_amp = 0.5
model.load(CKPT)
model.eval()

td = dom["train_data"]
with torch.no_grad():
    _, diag = model.simulate(
        forcing=td.forcing[:], initial_state=HydroState.zeros(dom["n_nodes"], device=DEVICE),
        graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
        withdrawals=td.withdrawals, day_of_year=td.day_of_year)

# STOCKAGE TOTAL, exactement comme le calcule la perte : sol (trois couches) + neige +
# nappe + canopee + milieu humide. Toute divergence avec la perte rendrait le diagnostic
# inutilisable pour regler la perte.
sp = model.spatial_encoder(td.node_coords, td.territorial.to_tensor())
z1 = getattr(model.vertical_column, "z1", 0.15)
stor = ((diag.theta1 * z1 + diag.theta2 * sp.Z2 + diag.theta3 * sp.Z3) * 1000.0
        + diag.swe + diag.s_gw + diag.canopy
        + (getattr(diag, "wet_vol", None) if getattr(diag, "wet_vol", None) is not None
           else torch.zeros_like(diag.swe)))

grp = td.tws_group
n_g = int(td.tws_obs.shape[1])
cnt = torch.zeros(n_g, device=stor.device).index_add_(
    0, grp, torch.ones_like(grp, dtype=stor.dtype)).clamp(min=1)
sim = (torch.zeros(stor.shape[0], n_g, device=stor.device)
       .index_add_(1, grp, stor) / cnt).cpu().numpy()
obs = td.tws_obs.cpu().numpy()
mois = np.array([t.month for t in dom["times"]])

print(f"\n{'bassin':8s} {'amplitude sim':>14s} {'amplitude GRACE':>16s} {'rapport':>8s} "
      f"{'pic sim':>8s} {'pic GRACE':>10s} {'correlation':>12s}")
for g, nom in enumerate(PLATEFORMES):
    o = obs[:, g]
    ok = np.isfinite(o)
    if ok.sum() < 24:
        print(f"{nom:8s} pas assez de mois GRACE")
        continue
    s = sim[:, g]
    # climatologie mensuelle centree des deux cotes : c'est la FORME qu'on compare,
    # jamais le niveau, les deux references etant arbitraires.
    cs = np.array([s[(mois == m) & ok].mean() for m in range(1, 13)])
    co = np.array([o[(mois == m) & ok].mean() for m in range(1, 13)])
    cs -= cs.mean(); co -= co.mean()
    amp_s, amp_o = cs.max() - cs.min(), co.max() - co.min()
    r = float(np.corrcoef(cs, co)[0, 1])
    print(f"{nom:8s} {amp_s:11.0f} mm {amp_o:13.0f} mm {amp_s / max(amp_o, 1e-9):8.2f} "
          f"{int(np.argmax(cs)) + 1:8d} {int(np.argmax(co)) + 1:10d} {r:12.3f}")
    print("         sim  : " + " ".join(f"{v:6.0f}" for v in cs))
    print("         GRACE: " + " ".join(f"{v:6.0f}" for v in co))

print("""
LECTURE. Un rapport d'amplitude proche de 1 avec une correlation elevee : la contrainte
est satisfaisable, le probleme est un reglage de poids. Un rapport tres superieur a 1 :
la colonne respire beaucoup plus que ce que le satellite voit, et aucun poids ne
reconciliera sans aplatir la dynamique qui fait le debit -- il faut alors soit corriger
la physique du stockage, soit n'imposer GRACE que sur la FORME (correlation) et non sur
l'amplitude. Un decalage de pic d'un ou deux mois avec une bonne amplitude : la
dynamique de recharge et de vidange est en cause, pas le volume.""")
