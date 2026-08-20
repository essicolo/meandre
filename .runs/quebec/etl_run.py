"""Étape 2 du design ET appris : run mono-région avec la demande évaporative du
module MLP (banc et_bench, supervisé MOD16) injectée comme 7e canal de forçage
(etp_channel), à la place de formule ETP × K_c. Un seul changement vs la recette
v4 de la région ; baselines GASP : v4 0.489 / v7 (ancrages) 0.577 held-out.

  ETL_REGION=gasp ETL_EPOCHS=12 python .runs/quebec/etl_run.py
"""
import os
import sys

# GRAINE (2026-08-19). Les poids du NeRF etaient tires au HASARD a chaque
# entrainement : init_from_literature ne biaise que la derniere couche. Deux runs de
# la MEME recette ne partaient donc pas du meme modele. Mesure : la recette du champion
# rejouee rend 0.7283 a l'initialisation contre 0.7432 le 16 aout, soit 0.015 d'ecart
# AVANT le moindre pas de gradient. Toute experience tranchee sur un ecart plus petit
# que ce bruit ne vaut rien. ETL_SEED change le tirage pour mesurer la dispersion.
import random as _rnd
import numpy as _npseed
import torch as _tseed
_GRAINE = int(os.environ.get("ETL_SEED", "1234"))
_rnd.seed(_GRAINE); _npseed.random.seed(_GRAINE)
_tseed.manual_seed(_GRAINE); _tseed.cuda.manual_seed_all(_GRAINE)
print(f"[etl] graine = {_GRAINE}")
from dataclasses import replace as _dc_replace
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tomllib
import numpy as np
import torch
import torch.nn as nn
from meandre.model import HydroModel
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.utils.metrics import kge as kge_fn
from meandre.utils.state import HydroState
from joint_data import load_region

REG = os.environ.get("ETL_REGION", "gasp").lower()
N_EPOCHS = int(os.environ.get("ETL_EPOCHS", "12"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CFG = ".runs/quebec/config/gasp-v4.toml"
# GARDE-FOU DE CONTENTION (2026-08-10) : deux entraînements simultanés sur la même carte
# passent de 450 s/époque à 4000-11000 s/époque (mesuré : un job de 3.5 h en prend 85).
# `pgrep` depuis un shell POSIX NE VOIT PAS les processus Windows — d'où le lancement en
# parallèle qui a coûté une nuit. Vérification par tasklist, contournable par ETL_FORCE=1.
if os.environ.get("ETL_FORCE", "0") != "1":
    import subprocess as _sp
    try:
        _out = _sp.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                       capture_output=True, text=True, timeout=20).stdout
        _n = sum(1 for _l in _out.splitlines() if _l.lower().startswith('"python.exe"'))
    except Exception:
        _n = 0
    if _n > 2:
        raise SystemExit(f"[etl] REFUS : {_n} processus python déjà actifs (contention GPU). "
                         f"Attendre, ou forcer avec ETL_FORCE=1.")

CKPT = f".runs/quebec/checkpoints/best-{REG}-etl{os.environ.get('ETL_TAG', '')}.pt"   # ETL_TAG évite d'écraser les checkpoints de diagnostic
ETB = "D:/meandre-data/quebec/checkpoints-etbench"

cfg = tomllib.load(open(BASE_CFG, "rb"))
lcfg = dict(cfg["loss"]); tcfg = cfg["training"]; mcfg = cfg["model"]
if "ETL_WSNOW" in os.environ:
    # seuils de fonte appris contre MOD10 (fonte à 0 jusqu'à Tmax+5.5 au banc freshet
    # = 2 semaines de retard ; la donnée entre par la loss, leçon pilote4b/4c)
    lcfg["w_snow"] = float(os.environ["ETL_WSNOW"])
    print(f"[etl] w_snow = {lcfg['w_snow']} (fonte supervisée MOD10)")
if "ETL_WTWS" in os.environ:
    # GRACE (anomalie de stockage total). Actif par défaut à 0.2 via le fichier de
    # config ; on l'expose pour pouvoir mesurer ce qu'il apporte OU coûte, sa ligne de
    # base ayant été corrigée le 2026-08-10 (elle était calculée par tronçon de séquence).
    lcfg["w_tws"] = float(os.environ["ETL_WTWS"])
    print(f"[etl] w_tws override = {lcfg['w_tws']} (GRACE)")
if "ETL_WET" in os.environ:
    # mode appris : w_et(MOD16) est un DOUBLE ancrage (le module encode déjà MOD16,
    # biaisé +15-30 % à l'est vs bilan) — il poussait K_c à 1.07 malgré beta 0.78 (etl2)
    lcfg["w_et"] = float(os.environ["ETL_WET"])
    print(f"[etl] w_et override = {lcfg['w_et']}")

r = load_region(REG, lcfg, device=DEVICE)
td, vd = r["train_data"], r["val_data"]
n_nodes = r["n_nodes"]
print(f"[etl] {REG}: {n_nodes} nœuds, {r['n_gauges']} jauges")

# ── demande évaporative du module appris (MLP du banc, gelé) ────────────────
norm = torch.load(f"{ETB}/norm.pt", weights_only=False)
H_HIST, H_COMP = norm["h_hist"], norm["h_comp"]
F_STATIC = r["territorial"].n_features
mlp = nn.Sequential(nn.Linear(12 + F_STATIC + 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()).to(DEVICE)
sd = torch.load(f"{ETB}/mlp.pt", weights_only=True)
mlp.load_state_dict({k.replace("head.", ""): v for k, v in sd.items()})
mlp.eval()

with torch.no_grad():
    F = td.forcing            # (T, N, 6+) sur device
    T = F.shape[0]
    mean, std = norm["mean"].to(DEVICE), norm["std"].to(DEVICE)
    C = torch.cat([torch.zeros(1, n_nodes, 6, device=DEVICE), F[:, :, :6].cumsum(0)], dim=0)
    t_ar = torch.arange(T, device=DEVICE)
    lo8 = torch.clamp(t_ar - (H_COMP - 1), min=0)
    a8 = (C[t_ar + 1] - C[lo8]) / (t_ar + 1 - lo8).reshape(-1, 1, 1)
    hi90, lo90 = torch.clamp(t_ar - (H_COMP - 1), min=1), torch.clamp(t_ar - (H_COMP - 1) - H_HIST, min=0)
    a90 = (C[hi90] - C[lo90]) / torch.clamp(hi90 - lo90, min=1).reshape(-1, 1, 1)
    # NB : fenêtres TRAÎNANTES (8 j finissant à t, 90 j avant) — au banc la fenêtre 8 j
    # était le composite [t, t+8) ; décalage ~4 j << cycle saisonnier de l'ET.
    doy = td.day_of_year
    sc = torch.stack([torch.sin(2 * np.pi * doy / 365.25), torch.cos(2 * np.pi * doy / 365.25)], dim=1)
    lat_col = 0 if 40 < float(td.node_coords[:, 0].mean()) < 62 else 1
    lat = td.node_coords[:, lat_col].float() / 50.0
    stat = torch.cat([r["territorial"].data, lat[:, None]], dim=1)   # (N, F+1)
    demand = torch.empty(T, n_nodes, device=DEVICE)
    for lo in range(0, T, 365):
        hi = min(lo + 365, T)
        a8n = (a8[lo:hi] - mean) / std
        a90n = (a90[lo:hi] - mean) / std
        scb = sc[lo:hi, None, :].expand(hi - lo, n_nodes, 2)
        x = torch.cat([a8n, a90n, stat[None, :, :-1].expand(hi - lo, -1, -1), scb, stat[None, :, -1:].expand(hi - lo, -1, -1)], dim=2)
        demand[lo:hi] = mlp(x.reshape(-1, x.shape[-1])).reshape(hi - lo, n_nodes)
print(f"[etl] demande ET apprise : {float(demand.mean()) * 365.25:.0f} mm/an moyen | max {float(demand.max()):.1f} mm/j")

_ds = float(os.environ.get("ETL_DEMAND_SCALE", "1.0"))
if _ds != 1.0:
    # débiaisage RÉGIONAL structurel de la demande (ratio bilan P-Q / MOD16) : appliqué
    # au canal, le gradient ne peut pas le défaire (le prior K_c doux était re-défait
    # à l'entraînement : mont-kc 0.583 < 0.617 inférence)
    demand = demand * _ds
    print(f"[etl] demande ET débiaisée × {_ds} (bilan/MOD16 régional)")
f7 = torch.cat([F[:, :, :6], demand[:, :, None]], dim=2)


def with_forcing(d):
    return TrainingData(forcing=f7, q_obs=d.q_obs, station_mask=d.station_mask,
                        station_idx=d.station_idx, graph=d.graph, node_coords=d.node_coords,
                        territorial=d.territorial, withdrawals=d.withdrawals,
                        day_of_year=d.day_of_year, train_slice=d.train_slice, val_slice=d.val_slice,
                        et_obs=d.et_obs, tws_obs=d.tws_obs)


td, vd = with_forcing(td), with_forcing(vd)

model = HydroModel(
    n_nodes=n_nodes,
    n_territorial=F_STATIC,
    n_forcing=6,
    use_temporal=False,
    use_residual=False,
    use_travel_time_attn=False,
    use_frost_rankinen=bool(mcfg.get("use_frost_rankinen", True)),
    column_theta_init_frac=float(mcfg.get("column_theta_init_frac", 0.9)),
    # ETL_STATIC : parametres GLOBAUX (37 scalaires) au lieu d'un champ spatial, comme
    # Hydrotel. Argument d'Essi (2026-08-08) : Hydrotel est cale sur les MEMES jauges et
    # atteint 0.82 en tenu de cote sur OUTV la ou meandre plafonne a 0.50 ; les observations
    # ne sont donc pas en cause. Ce qui differe est le nombre de degres de liberte : une
    # poignee de coefficients globaux chez Hydrotel contre un champ spatial + un effet
    # aleatoire par noeud chez meandre, pour 16 jauges. Test de parcimonie.
    param_mode=("static" if os.environ.get("ETL_STATIC", "0") == "1" else "nerf"),
    column_mode="hydrotel",
    et_mode="mcguinness",   # court-circuité par etp_channel
    use_temperature=False,
    # ETL_NO_LATENT : les codes latents sont par NOEUD, donc non transferables entre
    # regions (dimensions differentes). A desactiver pour un depart a chaud depuis le
    # champion d'une AUTRE region.
    use_latent_codes=(os.environ.get("ETL_NO_LATENT", "0") != "1"
                      and bool(mcfg.get("use_latent_codes", True))),
    latent_mode="additive",
    spatial_melt=bool(mcfg.get("spatial_melt", True)),
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=bool(mcfg.get("predict_lake_params", True)),
    compile_soil=bool(mcfg.get("compile_soil", True)),
    # AQUIFÈRE RESTITUANT (spec Essi 2026-07-28) : recharge -> réservoir lent par nœud,
    # vidange k_gw NeRF (prior = récessions MESURÉES des jauges). Banc partition :
    # krec 5e-5 + k_gw 0.068 = baseflow 24%, +0.07 KGE en inférence pure.
    use_aquifer=os.environ.get("ETL_AQUIFER", "0") == "1",
    # LEVIERS PICS (r vs Hydrotel) : advection pure (onde cinématique sans diffusion),
    # célérité dépendante du débit, UH de versant. Activés par env pour le banc GASP.
    pure_advection=os.environ.get("ETL_ADVECTION", "0") == "1",
    discharge_dependent_celerity=os.environ.get("ETL_DQCEL", "0") == "1",
    use_hillslope_uh=os.environ.get("ETL_HILLSLOPE", "0") == "1",
).to(DEVICE)
lp = dict(cfg.get("literature_prior") or {})
lp["K_c"] = float(os.environ.get("ETL_KC", "1.0"))   # autour de la demande apprise : 1.0 neutre,
# ou PRIOR MESURÉ = ratio bilan P-Q / MOD16 par région (MOD16 sur-évapore le sud +17-25%,
# et_bilan_check 2026-07-21 ; K_c×0.8 en inférence : MONT test 0.544->0.617, beta 0.73->0.91)
# K_sat_1 (surface) : l'init/prior littérature à 0.080 m/j est 6× trop perméable
# (diag GASP : le sol absorbe 83% de l'orage, coeff ruiss 17% vs 30-50% réel).
# Recaler l'ancre du prior plus bas re-génère la crue (banc d'impulsion), K_sat_3
# intact => baseflow préservé. Cible via env (défaut = valeur actuelle).
if "ETL_KSAT1" in os.environ:
    lp["K_sat_1"] = float(os.environ["ETL_KSAT1"])
    print(f"[etl] K_sat_1 prior recalé -> {lp['K_sat_1']} m/j (génération de crue)")
if "ETL_KGW" in os.environ:
    lp["k_gw"] = float(os.environ["ETL_KGW"])   # prior mesuré (récessions jauges)
    print(f"[etl] k_gw prior -> {lp['k_gw']} /j (récessions mesurées)")
if "ETL_TMELT" in os.environ:
    # seuil de fonte NeRF : cible init+prior Hydrotel-comme-littérature (QC ~ +2°C),
    # le champ reste libre par nœud (PAS un delta autour d'un squelette figé)
    lp["T_melt"] = float(os.environ["ETL_TMELT"])
    print(f"[etl] T_melt prior -> {lp['T_melt']} °C (seuil de fonte NeRF)")
model.spatial_encoder.init_from_literature(lp)
if os.environ.get("ETL_FONTE_LIT", "0") == "1":
    # taux de fonte par couvert : init Hydrotel-littérature (4.5/9/18 mm/j/°C au lieu
    # de 12/14/16) — apprenables comme avant, C_f NeRF = leur variation spatiale
    import math as _mth
    with torch.no_grad():
        for nm, v in [("sp_fonte_conif", 4.5), ("sp_fonte_feu", 9.0), ("sp_fonte_dec", 18.0)]:
            getattr(model.vertical_column, nm).copy_(torch.tensor(_mth.log(_mth.expm1(v))))
    print("[etl] taux de fonte init littérature-Hydrotel : 4.5/9/18 mm/j/°C")
# ETP : par défaut la demande APPRISE injectée en 7e canal. ETL_ETP=linacre bascule sur
# la formule Linacre CALÉE du projet, celle qu'utilise la configuration ancrée à 0.7748.
# Motif (2026-08-14) : ancrer le sol d'Hydrotel pendant l'entraînement EFFONDRE le score
# (0.349 sol complet, 0.542 texture seule, contre 0.605 sans ancrage), alors que le MÊME
# sol en inférence pure donne 0.7748. La différence entre les deux situations n'est pas
# l'entraînement mais l'ETP : on mariait le sol d'Hydrotel à l'évaporation de méandre.
# Une calibration est un PAQUET co-adapté ; en prendre la moitié casse l'équilibre.
# SEUIL PLUIE/NEIGE calibré (thiessen.csv) : présent dans la configuration ancrée à
# 0.7748, ABSENT du chemin d'entraînement jusqu'au 2026-08-14 — trouvé par un diff
# systématique des deux configurations, après deux expériences ratées à deviner la pièce
# manquante une par une.
# PLATEFORME D'ANCRAGE. Hydrotel est un ENSEMBLE de 6 plateformes qui different par
# leurs PARAMETRES *et* par leurs FONCTIONS : les prefixes l'encodent, LN = Linacre
# (une seule, LN24HA) et MG = McGuinness (les cinq autres). Notre socle etait ancre sur
# LN24HA, la seule Linacre et la MOINS BONNE des six (0.7531 contre 0.8299 pour MG24HK).
# ETL_MEMBRE choisit la plateforme ; l'ETP suit automatiquement sa formule.
_MEMBRE = os.environ.get("ETL_MEMBRE", "LN24HA")
_PLATB = "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel"
_PROJ_M = f"{_PLATB}/{_MEMBRE}/{REG.upper()}_{_MEMBRE}_2020"
if _MEMBRE != "LN24HA":
    os.environ.setdefault("ETL_MELT_DIR", _PROJ_M)
    print(f"[etl] plateforme d'ancrage : {_MEMBRE} ({_PROJ_M.split('/')[-1]})")

# ETL_SEUIL_VALEUR impose un seuil EXPLICITE (en °C) au lieu de celui du projet.
# Motif (2026-08-20) : le profil mensuel du champion montre decembre en EXCES (1.207)
# et avril en DEFICIT (0.729), signature d'eau relachee en debut d'hiver au lieu d'etre
# stockee jusqu'a la crue. Le seuil du projet est a -2.2168 °C, donc tout ce qui est
# au-dessus compte comme PLUIE : un seuil aussi bas fabrique de la pluie en decembre.
if "ETL_SEUIL_VALEUR" in os.environ:
    _sv = float(os.environ["ETL_SEUIL_VALEUR"])
    model.vertical_column.t_neige_seuil = _sv
    print(f"[etl] seuil pluie/neige IMPOSE : {_sv:+.4f} °C")
elif os.environ.get("ETL_SEUIL_NEIGE", "1") == "1":
    from meandre.data.hydrotel_calib import load_passage_pluie_neige as _lppn
    _pls = os.environ.get("ETL_MELT_DIR") or         _PROJ_M
    _sn = _lppn(_pls)
    if _sn != 0.0:
        model.vertical_column.t_neige_seuil = _sn
        print(f"[etl] seuil pluie/neige du projet : {_sn:+.4f} °C (méandre codait 0.0)")

if os.environ.get("ETL_ETP", "appris") == "mcguinness":
    from meandre.data.hydrotel_calib import load_mcguinness_nodes as _lmg
    model.vertical_column.et_mode = "mcguinness"
    _cmg = _lmg(_PROJ_M, r["node_ids"], device=DEVICE)
    model.vertical_column.set_mcguinness_coeff(_cmg)
    model.vertical_column.etp_channel = None
    print(f"[etl] ETP : McGuinness CALÉE de {_MEMBRE} "
          f"(coefficient méd {float(_cmg.median()):.3f})" if _cmg is not None
          else "[etl] ETP : McGuinness SANS coefficient (fichier absent)")
elif os.environ.get("ETL_ETP", "appris") == "linacre":
    from meandre.data.hydrotel_calib import load_linacre_nodes as _lln
    _pll = os.environ.get("ETL_MELT_DIR") or         _PROJ_M
    model.vertical_column.et_mode = "linacre"
    model.vertical_column.set_linacre_params(*_lln(_pll, r["node_ids"], device=DEVICE))
    model.vertical_column.etp_channel = None
    print(f"[etl] ETP : Linacre CALÉE du projet (paquet cohérent avec le sol ancré)")
else:
    model.vertical_column.etp_channel = 6
if os.environ.get("ETL_INIT_HYDROTEL", "0") in ("1", "courbe", "sauf_ks"):
    # DÉPART SUR LE CHAMP D'HYDROTEL puis optimisation libre (proposition d'Essi,
    # 2026-08-13). Différence essentielle avec ETL_SOIL_CALIB : celui-ci COURT-CIRCUITE
    # la sortie du réseau, qui n'apprend alors plus rien sur le sol. Ici on AJUSTE le
    # réseau par régression sur les valeurs calibrées par nœud, puis on le laisse
    # entièrement libre. Mesuré : le réseau reproduit le champ à 2 % près, dispersion
    # 0.741 contre 0.740 pour la cible — la capacité n'était pas le verrou, le point de
    # départ l'était (champ initial plat à 0.0017 de dispersion, K_sat 8× trop bas).
    from meandre.data.hydrotel_calib import load_calibrated_soil as _lcs
    from meandre.data.hydrotel_calib import courbe_retention_imposee
    _plh = os.environ.get("ETL_MELT_DIR") or         _PROJ_M
    _cs = _lcs(_plh, r["node_ids"], 0.15, device=DEVICE)
    _cib = {"K_sat_1": _cs["ks1"].float() * 24, "K_sat_2": _cs["ks2"].float() * 24,
            "K_sat_3": _cs["ks3"].float() * 24, "porosity_1": _cs["thetas1"].float(),
            "porosity_2": _cs["thetas2"].float(), "porosity_3": _cs["thetas3"].float(),
            "Z2": _cs["z2"].float(), "Z3": _cs["z3"].float()}
    print(f"[etl] départ sur le champ Hydrotel ({os.path.basename(_plh)}) : ajustement du NeRF")
    model.spatial_encoder.ajuster_sur_champ(
        td.node_coords, r["territorial"].data, _cib,
        n_iter=int(os.environ.get("ETL_INIT_ITER", "2000")))
    # Le prior ne contraint que la MOYENNE du champ : si sa cible reste la valeur
    # littérature (K_sat 0.04) il ramènerait le champ ajusté (0.32) vers le bas. On
    # réaligne donc les cibles du prior sur les médianes calibrées.
    # COURBE DE RÉTENTION : l'exposant de Campbell et le potentiel matriciel sont des
    # scalaires GLOBAUX de la colonne, pas des sorties du champ, donc l'ajustement du
    # champ ne les touche pas. Or méandre a b=2.65 contre 3.97 pour Hydrotel, ce qui
    # rend la conductivité EFFECTIVE 4× trop forte à humidité réaliste (K ∝ ω^(2b+3)) :
    # premier essai du 14 août, sol à la bonne perméabilité saturée mais à la mauvaise
    # forme, score effondré à 0.40. On pose donc aussi les paramètres de courbe.
    # COURBE PAR NŒUD (ETL_INIT_HYDROTEL=courbe). Mesuré le 14 août par un contrôle à
    # ZÉRO époque : le champ ajusté vaut 0.5629 quand l'ancrage complet vaut 0.7748.
    # L'écart ne vient pas de l'optimiseur (0 et 1 époque donnent la même chose) mais des
    # paramètres NON transférables : b, psis, krec, cin sont des SCALAIRES GLOBAUX dans
    # notre colonne, un par région, alors qu'Hydrotel les définit par classe de TEXTURE.
    # Le réseau n'a aucune sortie pour eux, donc le modèle ajusté est structurellement
    # incapable de représenter ce sol. Correctif minimal : imposer PAR NŒUD la seule
    # courbe de rétention (propriété de texture), en laissant au champ les conductivités,
    # porosités et épaisseurs.
    _mode_init = os.environ.get("ETL_INIT_HYDROTEL")
    if _mode_init in ("courbe", "sauf_ks"):
        if _mode_init == "sauf_ks":
            # CONFLIT AQUIFÈRE/CALAGE (2026-08-17). La courbe imposée comprend krec,
            # la recharge profonde du calage Hydrotel (~1e-7 m/h) — une valeur quasi
            # NULLE par construction, puisque chez Hydrotel c'est une fuite jamais
            # restituée que son calage étrangle. Notre aquifère est alimenté par ce
            # même terme : réservoir branché en aval d'un robinet fermé, d'où un gain
            # hivernal minuscule (février 0.655 -> 0.688 seulement). Quand l'aquifère
            # est actif, krec et coef_recharge restent donc LIBRES (init ETL_KREC).
            # RÉGRESSION SILENCIEUSE (trouvée le 2026-08-19). Libérer krec DÈS QUE
            # l'aquifère est actif faisait tomber le champion aq30 de 0.7880 à 0.4912
            # en évaluation pure : la recharge part alors à 5e-5, la nappe fournit 69 %
            # du débit et février passe de 0.688 à 1.40. Le champion du 17 août a tourné
            # AVANT ce correctif, donc avec krec IMPOSÉ. On revient au comportement qui
            # tient le record, et la libération devient EXPLICITE (ETL_KREC_LIBRE=1),
            # à n'utiliser qu'avec une valeur choisie et gelée (ETL_KREC + ETL_KREC_GEL).
            _aq_actif = os.environ.get("ETL_KREC_LIBRE", "0") == "1"
            # Tout imposer SAUF ce qu'on veut laisser apprendre (conductivités et
            # porosités). Sert à localiser les 0.145 qui séparent le sol entièrement
            # imposé (0.7368) du champ ajusté avec la seule courbe (0.5921) : épaisseurs,
            # fractions de surface, pente et recharge sont les candidats restants.
            # La règle est dans courbe_retention_imposee() pour que les diagnostics
            # appliquent EXACTEMENT la même (ils en divergeaient, cf. sa docstring).
            _courbe = courbe_retention_imposee(_cs, _aq_actif)
        else:
            _courbe = {k: v for k, v in _cs.items()
                       if k.startswith(("b", "psis", "omegpi", "mm", "nn"))
                       or k in ("krec", "cin", "coef_recharge")}
        model.vertical_column.set_calibrated_soil(_courbe)
        print(f"[etl] courbe de rétention imposée PAR NŒUD ({len(_courbe)} champs) ; "
              f"K_sat, porosités et épaisseurs restent au champ appris")
    import math as _mh2
    def _inv_sig(v, bornes):
        _x = min(max((float(v) - bornes[0]) / (bornes[1] - bornes[0]), 1e-6), 1 - 1e-6)
        return _mh2.log(_x / (1.0 - _x))
    _vc = model.vertical_column
    with torch.no_grad():
        for _i in (1, 2, 3):
            for _nm, _cal, _bnd in [(f"b{_i}_raw", f"b{_i}", "_b_bounds"),
                                    (f"psis{_i}_raw", f"psis{_i}", "_psis_bounds")]:
                if hasattr(_vc, _nm) and _cal in _cs:
                    _t2 = float(_cs[_cal].median())
                    getattr(_vc, _nm).copy_(torch.tensor(_inv_sig(_t2, getattr(_vc, _bnd))))
        if hasattr(_vc, "krec_raw") and "krec" in _cs:
            _vc.krec_raw.copy_(torch.tensor(_inv_sig(float(_cs["krec"].median()),
                                                     _vc._krec_bounds)))
    print(f"[etl] courbe de rétention posée sur le calage : b {float(_vc._sig(_vc.b1_raw, _vc._b_bounds)):.3f} "
          f"| psis {float(_vc._sig(_vc.psis1_raw, _vc._psis_bounds)):.4f} m")

    _pt = getattr(model.spatial_encoder, "_prior_targets", None) or {}
    for _k, _v in _cib.items():
        _pt[_k] = float(_v.median())
    model.spatial_encoder._prior_targets = _pt
    print(f"[etl] cibles du prior réalignées sur le calage (K_sat_1 -> {_pt['K_sat_1']:.4f} m/j)")

if os.environ.get("ETL_SOIL_CALIB", "0") == "1":
    # ANCRAGE DU SOL SUR LE CALAGE HYDROTEL (bv3c.csv + textures, agrégé UHRH->tronçon).
    # Réfuté deux fois en juillet (MONT-v3 -0.31, MONT-v9 0.125) d'où la « loi des
    # ancrages ». Mais ces deux essais précèdent l'AQUIFÈRE (28 juillet) : l'explication
    # écrite à l'époque était justement que le NeRF compensait par le sol les divergences
    # structurelles de la colonne, aquifère manquant en tête. Hypothèse d'Essi (2 août) :
    # avec le réservoir souterrain actif, geler le sol devrait passer. Test rejoué ici.
    from meandre.data.hydrotel_calib import load_calibrated_soil
    _pd_ = os.environ.get("ETL_SOIL_DIR",
        _PROJ_M)
    _z1 = float(getattr(model.vertical_column, "z1", 0.15))
    _calib = load_calibrated_soil(_pd_, r["node_ids"], _z1, device=DEVICE)
    # Le calage Hydrotel porte coef_recharge = 0 et krec ~ 1.3e-7 (vérifié sur MONT) :
    # Hydrotel n'a PAS de voie profonde, toute l'eau ressort latéralement. Geler ces
    # deux-là éteindrait l'aquifère de méandre et saborderait le test. Par défaut on
    # hérite donc de ce qu'Hydrotel a réellement calibré (géométrie + hydraulique des
    # couches) et on laisse libre la PARTITION verticale, qui est notre extension.
    if os.environ.get("ETL_SOIL_CALIB_FULL", "0") != "1":
        for _k in ("krec", "coef_recharge"):
            _calib.pop(_k, None)
        print("[etl] partition verticale (krec, coef_recharge) laissée LIBRE (aquifère)")
    if os.environ.get("ETL_SOIL_CALIB_TEXTURE", "0") == "1":
        # GEOMETRIE LIBRE : les épaisseurs du calage MONT sont UNIFORMES (0.22/0.16/2.65 m
        # sur tous les nœuds) — ce n'est pas un champ calibré mais une colonne unique de
        # 3 m appliquée partout, qui amortit tout (val bloquée, beta 0.71, gamma 0.58,
        # run mont-ancresol du 2 août). On n'hérite donc que de la TEXTURE, qui elle varie
        # spatialement et qu'Hydrotel a réellement calée.
        for _k in ("z1", "z2", "z3"):
            _calib.pop(_k, None)
        print("[etl] épaisseurs de couches laissées LIBRES (calage uniforme, non informatif)")
    model.vertical_column.set_calibrated_soil(_calib)
    print(f"[etl] sol ANCRÉ sur le calage Hydrotel : {_pd_}")
if os.environ.get("ETL_KGW_FIELD", "0") == "1":
    # CHAMP k_gw provincial (GP sur les récessions de 127 stations, R2 blocs 0.62,
    # incertitude calibrée) : remplace le k_gw régional constant par un champ continu
    # par nœud. Le NeRF module autour ; le champ fixe le niveau local mesuré.
    import pandas as _pd
    _cf = _pd.read_parquet("D:/meandre-data/quebec/champ_kgw_QC.parquet")
    _cf = _cf[_cf.region == REG].sort_values("node_idx")
    if len(_cf) == n_nodes:
        _kv = torch.tensor(_cf.k_gw.values, dtype=torch.float32, device=DEVICE)
        _o_se = model.spatial_encoder.forward
        def _se_kgw(*a, _o=_o_se, _k=_kv, **kw):
            sp = _o(*a, **kw); sp.k_gw = _k
            return sp
        model.spatial_encoder.forward = _se_kgw
        print(f"[etl] champ k_gw appliqué : méd {float(_kv.median()):.4f} | q10-q90 "
              f"{float(_kv.quantile(0.1)):.4f}-{float(_kv.quantile(0.9)):.4f}")
    else:
        print(f"[etl] champ k_gw ignoré ({len(_cf)} vs {n_nodes} nœuds)")
if os.environ.get("ETL_FRESHET_FIELD", "0") == "1":
    # CHAMP DE TIMING DE FONTE : le centre de masse observé du freshet (GP sur 126
    # stations, R2 blocs 0.62, contraste 32 jours du sud-ouest au nord-est) est converti
    # en décalage du seuil de fonte par nœud, via la sensibilité ΔCM/ΔT_melt mesurée au
    # banc (.runs/quebec/freshet_bench.py). Le NeRF module autour ; le champ fixe la DATE.
    import pandas as _pd
    _sens = float(os.environ.get("ETL_FRESHET_SENS", "0"))  # jours de CM par +1 °C de T_melt
    _cf = _pd.read_parquet("D:/meandre-data/quebec/champ_freshet_QC.parquet")
    _cf = _cf[_cf.region == REG].sort_values("node_idx")
    if len(_cf) == n_nodes and abs(_sens) > 1e-6:
        _cm = torch.tensor(_cf.cm_freshet.values, dtype=torch.float32, device=DEVICE)
        _dT = ((_cm - _cm.median()) / _sens).clamp(-2.0, 2.0)  # écart AU CHAMP, pas niveau absolu
        _o_se = model.spatial_encoder.forward
        def _se_freshet(*a, _o=_o_se, _d=_dT, **kw):
            sp = _o(*a, **kw); sp.T_melt = sp.T_melt + _d
            return sp
        model.spatial_encoder.forward = _se_freshet
        print(f"[etl] champ freshet appliqué : CM méd j{float(_cm.median()):.0f} | "
              f"ΔT_melt {float(_dT.min()):+.2f} à {float(_dT.max()):+.2f} °C (sens {_sens:.2f} j/°C)")
    else:
        print(f"[etl] champ freshet ignoré ({len(_cf)} vs {n_nodes} nœuds, sens={_sens})")
if "ETL_KREC" in os.environ:
    import math as _mk
    _kv = float(os.environ["ETL_KREC"])
    _lo, _hi = model.vertical_column._krec_bounds
    _x = min(max((_kv - _lo) / (_hi - _lo), 1e-6), 1 - 1e-6)
    with torch.no_grad():
        model.vertical_column.krec_raw.copy_(torch.tensor(_mk.log(_x / (1 - _x))))
    print(f"[etl] krec init -> {_kv:.0e} (drainage profond, banc partition)")
    # ETL_KREC_GEL=1 : la recharge est un LIVRABLE, pas un bouton de calage. Le débit
    # seul la pousse aux extrêmes (banc du 2026-08-19 : à 5e-5 la nappe fournit 69 % du
    # débit et le KGE tombe à 0.589 ; à 1e-4 il devient négatif). Quand on la pose à une
    # valeur choisie pour des raisons PHYSIQUES, il faut la geler, sinon l'apprentissage
    # la déplace et on ne sait plus ce qu'on a mesuré.
    if os.environ.get("ETL_KREC_GEL", "0") == "1":
        model.vertical_column.krec_raw.requires_grad_(False)
        print(f"[etl] krec GELÉ à {_kv:.1e} (exclu de l'apprentissage)")
if "ETL_MELT_DIR" in os.environ:
    # fonte RÉGIONALE calée (taux+seuils plateforme), NeRF mscale module autour.
    # A/B inférence 2026-07-25 : +0.149 KGE sur checkpoint gasp (v7 : +0.088 entraîné).
    from meandre.data.hydrotel_calib import load_melt_nodes
    _mp = load_melt_nodes(os.environ["ETL_MELT_DIR"], r["node_ids"], device=DEVICE)
    model.vertical_column.set_melt_params(_mp)
    print(f"[etl] fonte régionale ancrée ({os.environ['ETL_MELT_DIR'].split('/')[-1]}) | "
          f"taux méd {float(_mp['taux_c'].median()):.1f}/{float(_mp['taux_f'].median()):.1f}/{float(_mp['taux_d'].median()):.1f}")

# OCCUPATION DU SOL de PHYSITEL (défaut ON depuis le 2026-08-10). Sans elle, la
# physique reçoit 0 % de forêt, 0 % d'eau libre, 0 % d'imperméable et 0 % de milieu
# humide : tout le Québec en classe DÉCOUVERT (la plus fondante), fse=fsi=0, et une
# phénologie unique. Mesuré sur OUTV à intrants identiques et paramètres figés :
# r réseau 0.526 -> 0.896, KGE aux jauges 0.482 -> 0.749. ETL_OCCUPATION=0 pour l'ancien
# comportement (comparaisons historiques).
if os.environ.get("ETL_OCCUPATION", "1") == "1":
    _plat = "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
    _pl = os.environ.get("ETL_MELT_DIR") or f"{_plat}/{REG.upper()}_LN24HA_2020"
    try:
        from meandre.data.hydrotel_calib import (load_occupation_sol, load_milieux_humides,
                                                 load_phenologie)
        _lc = load_occupation_sol(_pl, r["node_ids"], device=DEVICE)
        _mh = load_milieux_humides(_pl, r["node_ids"], device=DEVICE)
        _lc.update(_mh)
        model.vertical_column.set_land_cover(_lc)
        _ph = load_phenologie(_pl)
        model.vertical_column.set_phenology(_ph or None)
        if _ph:
            print(f"[etl] phénologie du projet : {len(_ph)} classes "
                  f"(racines conifères {_ph['conifers'][2][0]:.2f} m, codé en dur 1.00)")
        if _mh:
            _wa = _mh["wet_a_raw"]
            print(f"[etl] milieux humides isolés : {int((_wa > 0).sum())} tronçons | "
                  f"aire méd {float(_wa[_wa > 0].median()):.2f} km² (module jamais actif avant le 10 août)")
        print(f"[etl] occupation PHYSITEL : forêt {float(_lc['f_forest_raw'].mean()):.3f} "
              f"(conif {float(_lc['f_forest_conifer_raw'].mean()):.3f}) | "
              f"eau {float(_lc['f_water_raw'].mean()):.3f} | "
              f"imperméable {float(_lc['f_urban_raw'].mean()):.3f} | "
              f"humide {float(_lc['f_wetland_raw'].mean()):.3f}")
    except Exception as _e:
        print(f"[etl] AVERTISSEMENT occupation du sol NON chargée ({type(_e).__name__}: {_e}) "
              f"-> la physique verra 0 % de forêt et 0 % d'eau")

# CÉLÉRITÉ : le K_musk appris collapse à 24h/tronçon (init) -> retard cumulé 6j du pic
# (diag GASP). Facteur d'échelle sur K_musk_hours (célérité de base plus rapide), en
# gardant le routage opérateur rapide (contrairement à dq_celerity qui le casse).
_kms = float(os.environ.get("ETL_KMUSK_SCALE", "1"))
if _kms != 1.0:
    _orig_fwd = model.spatial_encoder.forward
    def _fwd_kscale(*a, _o=_orig_fwd, _s=_kms, **k):
        sp = _o(*a, **k)
        sp.K_musk_hours = torch.clamp(sp.K_musk_hours * _s, min=1.0, max=48.0)
        return sp
    model.spatial_encoder.forward = _fwd_kscale
    print(f"[etl] K_musk × {_kms} (célérité accélérée, routage opérateur préservé)")
if os.environ.get("ETL_LAKE_ANCHOR", "0") == "1":
    # ANCRAGE D'EXUTOIRE : k_lake ancré sur k0*(A_ref/A), la tête module autour.
    # Mesuré le 5 août : imposé en inférence, ce prior rapporte +0.026 sur OUTV ; la même
    # tête laissée LIBRE apprend une direction opposée qui ne transfère pas (+0.002).
    import pandas as _pdl
    _rw = _pdl.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")
    _rw = _rw[_rw.region == REG]
    _A = r["territorial"].get_physical("area_km2_local").cpu().numpy()
    if len(_rw) == n_nodes:
        _Alac = _A * _rw["lake_fraction"].values.clip(0, 1)
        model.spatial_encoder.set_lake_anchor(
            torch.tensor(_Alac, dtype=torch.float32),
            a_ref_km2=float(os.environ.get("ETL_LAKE_AREF", "20")),
            alpha=float(os.environ.get("ETL_LAKE_ALPHA", "1.0")))
        _anc = model.spatial_encoder._lake_k_anchor
        print(f"[etl] ancrage d'exutoire : k0*(A_ref/A) | ancre méd {float(_anc.median()):.2e} "
              f"| q10-q90 {float(_anc.quantile(0.1)):.2e}-{float(_anc.quantile(0.9)):.2e}")
    else:
        print(f"[etl] ancrage d'exutoire ignoré ({len(_rw)} vs {n_nodes} nœuds)")
if os.environ.get("ETL_LAKE_AREA", "1") == "1":
    # ASSEMBLAGE (promu par défaut le 2026-08-09) : le module de lac recevait l'aire de
    # DRAINAGE au lieu de la surface d'eau libre (facteur 66 sur outv). Mesuré +0.015 en
    # tenu de côté, gain survivant au réentraînement. Désactivable par ETL_LAKE_AREA=0.
    # La pièce vit dans recette.py pour que les DIAGNOSTICS l'appliquent aussi : ils ne
    # le faisaient pas, et le champion y ressortait à 0.7591 au lieu de 0.7880.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from recette import poser_surface_lac
    poser_surface_lac(model, REG, td.territorial.get_physical("area_km2_local"), n_nodes)
    _lm = model._lake_area_km2[td.graph.is_lake.bool().cpu()]
    print(f"[etl] surface de lac : med {float(_lm.median()):.2f} km2")
if os.environ.get("ETL_PEDO", "0") == "1":
    # STRUCTURE PEDOTRANSFERT (Saxton & Rawls 2006) appliquee aux 12 parametres de sol.
    # On n'importe QUE le motif spatial, normalise a mediane 1 : le NIVEAU du modele a ete
    # mesure (recalage K_sat_1 = 0.04) et la conductivite de Saxton-Rawls est une valeur de
    # matrice au point, 40-80x au-dessus de la conductivite effective journaliere.
    # Mesure du 6 aout en inference : a mi-intensite, outv +0.065, gasp +0.006, sagu -0.038,
    # mont -0.079 -> ce n'est pas une amelioration universelle mais un RATTRAPAGE, utile la
    # ou le reseau a mal appris sa structure spatiale. ETL_PEDO_F regle l'intensite.
    import pandas as _pdp
    from meandre.data.pedotransfert import saxton_rawls as _sr
    _rwp = _pdp.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")
    _rwp = _rwp[_rwp.region == REG]
    if len(_rwp) == n_nodes:
        _f = float(os.environ.get("ETL_PEDO_F", "0.5"))
        _p = _sr(_rwp.f_sand.values, _rwp.f_clay.values)
        _mot = {}
        for _b, _k in [("K_sat", "k_sat"), ("porosity", "theta_s"),
                       ("theta_fc", "theta_fc"), ("theta_wp", "theta_wp")]:
            _v = _p[_k] / float(np.median(_p[_k]))
            _mot[_b] = torch.tensor(1.0 + _f * (_v - 1.0), dtype=torch.float32, device=DEVICE)
        _o_pd = model.spatial_encoder.forward
        def _fwd_pedo(*a, _o=_o_pd, _m=_mot, **k):
            sp = _o(*a, **k)
            for _b, _fac in _m.items():
                for _i in (1, 2, 3):
                    _nm = f"{_b}_{_i}"
                    if hasattr(sp, _nm):
                        setattr(sp, _nm, getattr(sp, _nm) * _fac)
            return sp
        model.spatial_encoder.forward = _fwd_pedo
        print(f"[etl] structure pedotransfert appliquee (intensite {_f}) : "
              f"K_sat q10-q90 {float(_mot['K_sat'].quantile(.1)):.2f}-{float(_mot['K_sat'].quantile(.9)):.2f}")
    else:
        print(f"[etl] pedotransfert ignoree ({len(_rwp)} vs {n_nodes} noeuds)")
# ASSEMBLAGE (promu par défaut le 2026-08-09) : Hydrotel étale la production au VERSANT
# par son hydrogramme géomorphologique (<=10 j) avant le canal ; méandre livrait tout le
# jour même. Mesuré : r réseau contre Hydrotel 0.335 -> 0.470, +0.027 aux jauges en
# inférence pure. Désactivable par ETL_HGM=0.
if os.environ.get("ETL_HGM", "1") == "1":
    # NOYAU DE VERSANT D'HYDROTEL (cache .hgm du projet) : etale la production laterale
    # comme le C++ (convolution <=10 j, onde cinematique pixel par pixel precalculee).
    # Sans lui, l'eau du jour J arrive au troncon le jour J et les tetes de bassin
    # decorrelaient d'Hydrotel (r 0.27-0.31 quotidien). En inference sur le champion :
    # +0.027 aux jauges. Ici : ACTIF PENDANT L'ENTRAINEMENT, pour que le reseau
    # n'apprenne plus a compenser l'etalement manquant.
    from meandre.data.hgm_loader import lire_hgm
    _projh = _PROJ_M
    _K = lire_hgm(_projh, r["node_ids"])
    model.set_hgm_kernel(torch.tensor(_K, device=DEVICE))
    import numpy as _nph
    print(f"[etl] noyau HGM actif : jour0 med {float(_nph.median(_K[:,0])):.2f}, L={_K.shape[1]}")
# ASSEMBLAGE (promu par défaut le 2026-08-09) : loi de tarage Q = c·h^k calibrée du
# troncon.trl, que le lecteur PHYSITEL jetait. Fidélité des tronçons-lacs 0.202 -> 0.669.
# Désactivable par ETL_LAKE_TRL=0.
if os.environ.get("ETL_LAKE_TRL", "1") == "1":
    # LACS D'HYDROTEL depuis troncon.trl : surface d'eau + loi de tarage calibree
    # Q = c*h^k (le parseur historique JETAIT c et k et lisait la surface comme une
    # largeur). k_lake = c/A, beta = k. Impose HORS gradient (surcharge de lake_params).
    from pathlib import Path as _Pl
    _pt = _Pl(f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020/physitel/troncon.trl")
    _lgn = [l.strip() for l in _pt.read_text(encoding="latin-1").splitlines() if l.strip()]
    _dl = {}
    for _l in _lgn[3:]:
        _t = _l.split()
        if int(_t[1]) != 1:
            _ptr = 4 + int(_t[3])
            _dl[int(_t[0])] = (float(_t[_ptr+1]), float(_t[_ptr+2]), float(_t[_ptr+3]))
    _idxl = {int(i): j for j, i in enumerate(r["node_ids"])}
    import numpy as _npl
    _surf = _npl.full(n_nodes, _npl.nan); _c = _npl.full(n_nodes, _npl.nan); _k = _npl.full(n_nodes, _npl.nan)
    for _tid, (_s2, _c2, _k2) in _dl.items():
        if _tid in _idxl:
            _surf[_idxl[_tid]], _c[_idxl[_tid]], _k[_idxl[_tid]] = _s2, _c2, _k2
    _lacb = td.graph.is_lake.bool().cpu().numpy()
    _cv = _npl.isfinite(_surf) & _lacb & (_surf > 0)
    _cvt = torch.tensor(_cv, device=DEVICE)
    _ktl = torch.tensor(_npl.nan_to_num(_npl.where(_cv, _c / _npl.clip(_surf*1e6, 1, None), _npl.nan), nan=1e-4),
                        dtype=torch.float32, device=DEVICE)
    _btl = torch.tensor(_npl.nan_to_num(_k, nan=1.5), dtype=torch.float32, device=DEVICE)
    _olp = model.spatial_encoder.lake_params
    def _lp_trl(*a, _o=_olp, **kw):
        kk3, bb3 = _o(*a, **kw)
        return torch.where(_cvt, _ktl, kk3), torch.where(_cvt, _btl, bb3)
    model.spatial_encoder.lake_params = _lp_trl
    model.set_lake_area(torch.tensor(_npl.where(_cv, _surf, 1.0), dtype=torch.float32))
    print(f"[etl] lacs troncon.trl imposes : {int(_cv.sum())}/{int(_lacb.sum())} | "
          f"k_lake med {float(_npl.nanmedian(_npl.where(_cv, _c/_npl.clip(_surf*1e6,1,None), _npl.nan))):.2e} /s")
if "ETL_WARM_FROM" in os.environ:
    model.load(os.environ["ETL_WARM_FROM"])
    print(f"[etl] départ à chaud depuis {os.path.basename(os.environ['ETL_WARM_FROM'])}")
if os.environ.get("ETL_CAPACITE", "0") == "1":
    # TEST DE CAPACITÉ (question d'Essi, 3 août) : « si les paramètres ne s'ajustent pas
    # à la météo, il y a un problème ». On mesure le PLAFOND du modèle, pas sa
    # généralisation : tout est gelé sauf les codes latents additifs, qui sont un décalage
    # LIBRE par nœud sur les 37 paramètres, et la régularisation est coupée. C'est du
    # surajustement volontaire sur la période d'entraînement. Si le KGE d'ENTRAÎNEMENT
    # plafonne malgré cette liberté totale, la limite est dans la donnée ou la structure,
    # pas dans l'optimisation ni dans le réseau spatial.
    if not getattr(model.spatial_encoder, "use_latent_codes", False):
        raise SystemExit("[etl] ETL_CAPACITE exige use_latent_codes=true (mode additif)")
    _libre = []
    for _n, _p in model.named_parameters():
        _p.requires_grad = ("latent_codes" in _n)
        if _p.requires_grad:
            _libre.append((_n, _p.numel()))
    print(f"[etl] CAPACITÉ : {sum(k for _, k in _libre):,} params libres ({[n for n, _ in _libre]}), "
          f"tout le reste GELÉ, régularisation latente coupée")
print(f"[etl] modèle {sum(p.numel() for p in model.parameters()):,} params | etp_channel=6 (demande apprise × K_c NeRF, init 1.0)")

_lake_lr = float(os.environ.get("ETL_LAKE_LR", "50"))
# ── VALIDATION CROISÉE SPATIALE (ETL_FOLD="k/K") ────────────────────────────
# Retire les jauges du pli k de l'ENTRAÎNEMENT (leur q_obs passe à NaN, donc elles ne
# pèsent ni dans la perte ni dans la validation) et rapporte le tenu de côté SUR ELLES.
# Mesure ce que le champ spatial produit là où il n'a JAMAIS vu de débit — question
# devenue centrale le 2026-08-11, la physique ancrée battant la physique apprise de
# 0.134 sur les bassins jaugés. Découpage déterministe (indices modulo K).
_FOLD = os.environ.get("ETL_FOLD")
_fold_test = None
if _FOLD:
    _k, _K = (int(x) for x in _FOLD.split("/"))
    _ns = int(td.q_obs.shape[1])
    _fold_test = [i for i in range(_ns) if i % _K == _k]
    _qo_full = td.q_obs.clone()   # COPIE INTACTE pour l'évaluation : masquer q_obs
    # retire les jauges de l'entraînement ET les rend inévaluables si on ne garde pas
    # l'original (erreur commise au premier essai : le pli ne rapportait rien).
    _qo = td.q_obs.clone()
    _qo[:, _fold_test] = float("nan")
    td = _dc_replace(td, q_obs=_qo)
    r["train_data"] = td
    print(f"[etl] PLI SPATIAL {_k}/{_K} : {len(_fold_test)}/{_ns} jauges RETIRÉES de "
          f"l'entraînement (indices {_fold_test})")

tconf = TrainingConfig(
    lake_lr_mult=_lake_lr,
    n_epochs=N_EPOCHS,
    lr=float(os.environ.get("ETL_LR", tcfg.get("lr", 5e-4))),
    chunk_steps=int(tcfg.get("chunk_steps", 45)),
    tbptt_steps=int(tcfg.get("tbptt_steps", 365)),
    grad_clip=float(tcfg.get("clip_grad_norm", 1.0)),
    w_prior=0.0 if os.environ.get("ETL_CAPACITE", "0") == "1" else float(tcfg.get("w_prior", 0.005)),
    w_latent_reg=0.0 if os.environ.get("ETL_CAPACITE", "0") == "1" else float(tcfg.get("w_latent_reg", 1e-3)),
    best_metric="kge_median",
    # autopilot du TOML : LR plateau + garde-fou régression (sans lui, GASP/MONT-etl
    # divergeaient après le pic epoch ~7-12, val -0.15 non rattrapée — bug 2026-07-22)
    autopilot=bool(tcfg.get("autopilot", True)),
    autopilot_grace_epochs=int(tcfg.get("autopilot_grace_epochs", 8)),
    autopilot_lr_patience=int(tcfg.get("autopilot_lr_patience", 6)),
    autopilot_lr_factor=float(tcfg.get("autopilot_lr_factor", 0.5)),
    autopilot_lr_min=float(tcfg.get("autopilot_lr_min", 1e-5)),
    autopilot_beta_threshold=float(tcfg.get("autopilot_beta_threshold", 0.10)),
    autopilot_restart_regression=float(tcfg.get("autopilot_restart_regression", 0.05)),
    autopilot_restart_max=int(tcfg.get("autopilot_restart_max", 3)),
    val_every=1,
)
tr = Trainer(model=model, loss_fn=r["loss_fn"], train_data=td, val_data=vd,
             config=tconf, run_name=f"{REG}-etl", checkpoint_path=CKPT)
tr.fit()

# ── held-out 2022-2024 (best checkpoint) ─────────────────────────────────────
# ETL_EPOCHS=0 ne sauvegarde aucun point de reprise : on évalue alors le modèle EN
# MÉMOIRE. C'est le contrôle le moins cher qui soit — il mesure ce que vaut une
# initialisation AVANT le moindre pas de gradient, et il aurait évité plusieurs
# entraînements de 4 h lancés sur une initialisation jamais vérifiée.
if os.path.exists(CKPT):
    model.load(CKPT)
    # Le point de reprise RESTAURE krec_raw : sans ce rappel, ETL_KREC n'a aucun effet
    # en évaluation pure et un balayage de recharge mesure onze fois la même chose
    # (mesuré le 2026-08-19). Quand la recharge est GELÉE, elle vaut la valeur choisie
    # de bout en bout, y compris après le chargement.
    if os.environ.get("ETL_KREC_GEL", "0") == "1" and "ETL_KREC" in os.environ:
        import math as _mk2
        _kv2 = float(os.environ["ETL_KREC"])
        _lo2, _hi2 = model.vertical_column._krec_bounds
        _x2 = min(max((_kv2 - _lo2) / (_hi2 - _lo2), 1e-6), 1 - 1e-6)
        with torch.no_grad():
            model.vertical_column.krec_raw.copy_(torch.tensor(_mk2.log(_x2 / (1 - _x2))))
        print(f"[etl] krec ré-imposé après chargement : {_kv2:.1e} (gelé)")
else:
    print(f"[etl] pas de point de reprise ({N_EPOCHS} époque(s)) : évaluation du modèle EN MÉMOIRE")
model.eval()
with torch.no_grad():
    Q, _ = model.simulate(forcing=f7, initial_state=HydroState.zeros(n_nodes, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
times = r["times"]
# Le tenu de côté suit le découpage : par défaut ce qui suit la validation
# (2022-2024), sinon ETL_HELDOUT="debut,fin". Voir joint_data : la fenêtre historique
# est un juge biaisé (été 30 % plus humide que la période de sélection).
_HO = os.environ.get("ETL_HELDOUT", "2022-01-01,2024-12-31").split(",")
sl = (times >= _HO[0].strip()) & (times <= _HO[1].strip())
print(f"[etl] tenu de côté : {_HO[0].strip()} .. {_HO[1].strip()}")
slt = torch.tensor(sl.values if hasattr(sl, "values") else sl, device=DEVICE)
Qs = Q[slt][:, td.station_idx].cpu()
t0 = td.train_slice.start
_qo_eval = _qo_full if _fold_test is not None else td.q_obs
qo_test = _qo_eval[np.flatnonzero(sl)[0] - t0 : np.flatnonzero(sl)[-1] - t0 + 1].cpu()
ks = []
for s in range(Qs.shape[1]):
    v = ~torch.isnan(qo_test[:, s]) & ~torch.isnan(Qs[:, s])
    if v.sum() < 60: continue
    ks.append(float(kge_fn(qo_test[v, s], Qs[v, s])))
ks = np.array(ks)
print(f"\n[etl] HELD-OUT 2022-2024 {REG}: n={len(ks)} | médian {np.median(ks):.4f} | mean {ks.mean():.4f}")

# BIAIS MENSUEL dans le protocole de REFERENCE. Le score seul ne dit pas OU le modele
# se trompe, et les rapports mensuels vivaient jusqu'ici dans des scripts de diagnostic
# qui ne reproduisaient pas le pilote. Fevrier est le plus gros ecart connu du champion
# (0.688 de l'observe, quand Hydrotel fait l'erreur inverse a 1.235).
import pandas as _pdm
_mo = _pdm.DatetimeIndex(times[np.flatnonzero(sl)[0]:np.flatnonzero(sl)[-1] + 1]).month
_qs_np = Qs.numpy(); _qo_np = qo_test.numpy()
_rap = []
for _m in range(1, 13):
    _k = _mo == _m
    if _k.sum() < 10:
        _rap.append(float("nan")); continue
    _num = np.nansum(_qs_np[_k], 0); _den = np.clip(np.nansum(_qo_np[_k], 0), 1e-9, None)
    _rap.append(float(np.nanmedian(_num / _den)))
print("[etl] simule/observe par mois : " + " ".join(
    f"{_m:02d}={_r:.3f}" for _m, _r in zip(range(1, 13), _rap)))

# DECOMPOSITION du KGE : correlation, biais de volume, rapport de variabilite.
# Un score agrege ne dit pas OU l'on perd. Ajoute le 2026-08-20 : le biais mensuel a
# montre que nous et Hydrotel nous trompons en sens OPPOSES tout en etant separes de
# 0.042, donc l'ecart n'est pas dans le volume mensuel mais ailleurs.
def _composantes(_o, _s):
    _r, _b, _g = [], [], []
    for _k in range(_s.shape[1]):
        _v = np.isfinite(_o[:, _k]) & np.isfinite(_s[:, _k])
        if _v.sum() < 60: continue
        _oo, _ss = _o[_v, _k], _s[_v, _k]
        if _oo.std() < 1e-9 or _ss.std() < 1e-9 or _oo.mean() <= 0 or _ss.mean() <= 0: continue
        _r.append(np.corrcoef(_oo, _ss)[0, 1])
        _b.append(_ss.mean() / _oo.mean())
        _g.append((_ss.std() / _ss.mean()) / (_oo.std() / _oo.mean()))
    return np.median(_r), np.median(_b), np.median(_g)
_cr, _cb, _cg = _composantes(_qo_np, _qs_np)
print(f"[etl] composantes medianes : r={_cr:.4f} beta={_cb:.4f} gamma={_cg:.4f}")

# COMPARAISON AU MEMBRE HYDROTEL, memes stations, memes jours, meme protocole.
# ETL_CMP_HYDROTEL=MG24HK. Sert a repondre a UNE question : un ecart mensuel est-il
# NOTRE defaut ou un defaut PARTAGE (donc du forcage) ? Sans ce cote a cote, on repare
# a l'aveugle. Ajoute le 2026-08-20 apres constat d'un retard d'un mois sur la crue.
if os.environ.get("ETL_CMP_HYDROTEL"):
    import xarray as _xrh
    from meandre.data.hydrotel_calib import appariement_provincial as _appm
    _mb = os.environ["ETL_CMP_HYDROTEL"]
    try:
        _z = _xrh.open_zarr("C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data/"
                            f"06_posttraitement/posttraitement_{_mb}.zarr")
        _cols = _appm(REG, [int(r["node_ids"][int(_i)]) for _i in td.station_idx.cpu().numpy()],
                      np.asarray(_z["troncon_id"].values).astype(str))
        _t0h, _t1h = str(times[np.flatnonzero(sl)[0]])[:10], str(times[np.flatnonzero(sl)[-1]])[:10]
        _qh = _z["Dis"].sel(time=slice(_t0h, _t1h)).transpose("time", "troncon_idx").values[
            :, [_c if _c is not None else 0 for _c in _cols]]
        _z.close()
        _nT = min(len(_qo_np), len(_qh))
        _raph, _kh = [], []
        for _m in range(1, 13):
            _k = _mo[:_nT] == _m
            if _k.sum() < 10:
                _raph.append(float("nan")); continue
            _raph.append(float(np.nanmedian(np.nansum(_qh[:_nT][_k], 0)
                                            / np.clip(np.nansum(_qo_np[:_nT][_k], 0), 1e-9, None))))
        for _s in range(_qh.shape[1]):
            _v = ~np.isnan(_qo_np[:_nT, _s]) & ~np.isnan(_qh[:_nT, _s])
            if _v.sum() < 60: continue
            _kh.append(float(kge_fn(torch.tensor(_qo_np[:_nT, _s][_v]), torch.tensor(_qh[:_nT, _s][_v]))))
        print(f"[etl] Hydrotel {_mb} : KGE median {np.median(_kh):.4f}")
        print(f"[etl] Hydrotel {_mb} par mois : " + " ".join(
            f"{_m:02d}={_rr:.3f}" for _m, _rr in zip(range(1, 13), _raph)))
        _hr, _hb, _hg = _composantes(_qo_np[:_nT], _qh[:_nT])
        print(f"[etl] Hydrotel {_mb} composantes : r={_hr:.4f} beta={_hb:.4f} gamma={_hg:.4f}")
        print(f"[etl] ecart NOUS-EUX composantes : r={_cr - _hr:+.4f} beta={_cb - _hb:+.4f} gamma={_cg - _hg:+.4f}")
        print("[etl] ecart NOUS-EUX par mois  : " + " ".join(
            f"{_m:02d}={_a - _b:+.3f}" for _m, (_a, _b) in enumerate(zip(_rap, _raph), 1)))
    except Exception as _e:
        print(f"[etl] comparaison Hydrotel impossible : {type(_e).__name__}: {_e}")
if _fold_test is not None:
    kf = []
    for _s in _fold_test:
        v = ~torch.isnan(qo_test[:, _s]) & ~torch.isnan(Qs[:, _s])
        if v.sum() < 60:
            continue
        kf.append(float(kge_fn(qo_test[v, _s], Qs[v, _s])))
    if kf:
        kf = np.array(kf)
        print(f"[etl] PLI {_FOLD} - jauges JAMAIS VUES : n={len(kf)} | "
              f"median {np.median(kf):.4f} | mean {kf.mean():.4f}")
    _vus = [i for i in range(qo_test.shape[1]) if i not in _fold_test]
    kv = []
    for _s in _vus:
        v = ~torch.isnan(qo_test[:, _s]) & ~torch.isnan(Qs[:, _s])
        if v.sum() >= 60:
            kv.append(float(kge_fn(qo_test[v, _s], Qs[v, _s])))
    if kv:
        print(f"[etl] PLI {_FOLD} - jauges VUES a l'entrainement : n={len(kv)} | "
              f"median {np.median(kv):.4f}")
print("[etl] DONE")
