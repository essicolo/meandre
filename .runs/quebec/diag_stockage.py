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
if dom.get("phenology"):
    model.vertical_column.set_phenology(dom["phenology"])
# SOL CALIBRE : le point de reprise a ete entraine AVEC, et sa fiche d'execution le dit.
# L'omettre faisait lever deux avertissements « scores FAUX si ce n'est pas volontaire »
# au premier essai (2026-08-26) : un diagnostic tourne sur une autre physique que le
# modele qu'il pretend diagnostiquer ne vaut rien.
if dom.get("soil"):
    from meandre.data.hydrotel_calib import imposed_retention_curve
    model.vertical_column.set_calibrated_soil(imposed_retention_curve(dom["soil"], True))
# ── LEVIERS DE RETENTION, testes SANS ENTRAINEMENT ──────────────────────────
# Le diagnostic du 2026-08-26 a montre que le defaut n'est ni l'amplitude (rapports 0.77
# a 1.44) ni une derive, mais la DESCENTE : le modele vide son stockage un a deux mois
# avant GRACE, sur les quatre bassins. Les deux mecanismes qui gouvernent cette descente
# sont le drainage de la couche 3 et le temps de residence de la nappe. On les balaie en
# inference, juges par la climatologie satellitaire, avant de payer le moindre epoch.
# DIFFUSIVITE THERMIQUE : le seul levier de retention qui reste candidat apres l'echec
# des quatre leviers de reservoir lent (R48). Contrairement au multiplicateur de
# profondeur de gel que j'avais essaye puis retire, celui-ci porte sur un PARAMETRE
# PHYSIQUE et non sur une sortie : une diffusivite plus faible ralentit la descente ET
# la remontee du front, donc maintient le drainage bloque plus longtemps au printemps.
# L'entrainement du 2026-08-27 a spontanement reduit ce rapport de 34 % ; on verifie ici
# que cette reduction fait bien ce qu'on croit sur le stockage.
_lp0 = dict(cfg.get("literature_prior") or {})
_lp0["K_sat_1"] = 0.04; _lp0["K_c"] = 1.0; _lp0["k_gw"] = 0.07; _lp0.setdefault("krec", 5e-5)
model.spatial_encoder.init_from_literature(_lp0)
_dd = float(os.environ.get("DIAG_DIFF", "1.0"))
if _dd != 1.0:
    _o_d = model.spatial_encoder.forward
    def _se_diff(*a, _o=_o_d, _m=_dd, **kw):
        sp = _o(*a, **kw); sp.diff_gel = sp.diff_gel * _m
        return sp
    model.spatial_encoder.forward = _se_diff
_l3 = os.environ.get("DIAG_L3EXP")
if _l3:
    model.vertical_column.l3_drain_exp = float(_l3)
_kg = float(os.environ.get("DIAG_KGW", "1.0"))
if _kg != 1.0:
    # Le champ k_gw provincial a une mediane de 0.0856 par jour, soit douze jours de
    # temps de residence. GRACE demande une retention d'un a deux MOIS. Les recessions
    # qui ont servi a l'estimer mesurent le tarissement du COURS D'EAU, donc le
    # compartiment le plus rapide : rien n'oblige la masse de la nappe a le suivre.
    _o_se = model.spatial_encoder.forward
    def _se_lent(*a, _o=_o_se, _m=_kg, **kw):
        sp = _o(*a, **kw); sp.k_gw = sp.k_gw * _m
        return sp
    model.spatial_encoder.forward = _se_lent
print(f"[diag] leviers : l3_drain_exp={_l3 or 'lineaire'} | k_gw x{_kg} "
      f"| diffusivite x{_dd}", flush=True)

model.vertical_column.split_mode = "wet_bulb"
model.vertical_column.t_neige_seuil = -0.8
model.vertical_column.melt_seasonal_amp = 0.5
model.load(CKPT)
model.eval()

td = dom["train_data"]
with torch.no_grad():
    _, _, diag = model.simulate(
        forcing=td.forcing[:], initial_state=HydroState.zeros(dom["n_nodes"], device=DEVICE),
        graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
        withdrawals=td.withdrawals, day_of_year=td.day_of_year,
        return_diagnostics=True)

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
       .index_add_(1, grp, stor) / cnt).detach().cpu().numpy()
obs = td.tws_obs.cpu().numpy()
mois = np.array([t.month for t in dom["times"]])

# PROFONDEUR DE GEL SIMULEE. Rankinen la calcule sur tout le profil, mais la colonne ne
# s'en sert que rapportee a l'epaisseur de la couche 1 (15 cm) : au-dela, un front plus
# profond ne change plus rien aux flux. Avant de toucher a ce couplage, il faut savoir si
# le front simule est realiste -- au Quebec il descend couramment de 40 a 100 cm.
if getattr(diag, "prof_gel_cm", None) is not None:
    _pg = diag.prof_gel_cm.detach()
    _pgb = (torch.zeros(_pg.shape[0], n_g, device=_pg.device)
            .index_add_(1, grp, _pg) / cnt).cpu().numpy()
    print("\nprofondeur de gel simulee (cm), climatologie mensuelle")
    print(f"{'bassin':8s} " + " ".join(f"{m:6d}" for m in range(1, 13)) + "   max")
    for g, nom in enumerate(PLATEFORMES):
        c = [float(_pgb[mois == m, g].mean()) for m in range(1, 13)]
        print(f"{nom:8s} " + " ".join(f"{v:6.1f}" for v in c) + f" {max(c):6.1f}")
    print("  (au-dela de 15 cm l'etranglement du sol est deja complet : tout ce qui")
    print("   depasse cette ligne est calcule puis jete)")

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
