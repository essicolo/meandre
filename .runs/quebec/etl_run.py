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
from meandre.utils import paths as _paths

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
ETB = f"{_paths.DATA_ROOT}/quebec/checkpoints-etbench"

cfg = tomllib.load(open(BASE_CFG, "rb"))
lcfg = dict(cfg["loss"]); tcfg = cfg["training"]; mcfg = cfg["model"]
if "ETL_WSNOW" in os.environ:
    # seuils de fonte appris contre MOD10 (fonte à 0 jusqu'à Tmax+5.5 au banc freshet
    # = 2 semaines de retard ; la donnée entre par la loss, leçon pilote4b/4c)
    lcfg["w_snow"] = float(os.environ["ETL_WSNOW"])
    print(f"[etl] w_snow = {lcfg['w_snow']} (fonte supervisée MOD10)")
    if lcfg["w_snow"] > 0:
        # R24 (2026-08-21). Cette contrainte n'avait JAMAIS ete active : `with_forcing`
        # perdait `swe_obs` (dette #14). En la reparant on la rend effective, et l'audit
        # dit qu'elle pousse DANS LE MAUVAIS SENS : le modele voit +0.47 de fraction de
        # couverture de plus que MODIS en mars et +0.45 en avril, a 38 ecarts-times, donc
        # le terme lui demanderait d'ENLEVER de la neige et de fondre plus tot. Or GRACE
        # veut l'inverse (la reserve doit tenir jusqu'en mai) et CanSWE mesure un pic de
        # 238 mm quand le modele en a 121. MODIS mesure une reflectance et sous-estime la
        # neige sous couvert : OUTV est boise a 74 %, dont 32 % de coniferes.
        print("[etl] ATTENTION w_snow > 0 : contrainte MODIS neige effective depuis le "
              "correctif de with_forcing (dette #14).")
        print("      R24 : elle tire vers MOINS de neige en mars-avril, contre GRACE et "
              "CanSWE. Mettre ETL_WSNOW=0 tant que la cible n'est pas CanSWE (masse) "
              "plutot que MOD10 (couverture).")
if "ETL_WTWS" in os.environ:
    # GRACE (anomalie de stockage total). Actif par défaut à 0.2 via le fichier de
    # config ; on l'expose pour pouvoir mesurer ce qu'il apporte OU coûte, sa ligne de
    # base ayant été corrigée le 2026-08-10 (elle était calculée par tronçon de séquence).
    lcfg["w_tws"] = float(os.environ["ETL_WTWS"])
    print(f"[etl] w_tws override = {lcfg['w_tws']} (GRACE)")
if "ETL_WSWE" in os.environ:
    # CanSWE : masse du manteau mesuree au sol (R24). Cible distincte de MOD10, qui ne
    # mesure qu'une couverture. Le pic simule vaut 121 mm sur OUTV contre 238 mesures.
    lcfg["w_swe_mass"] = float(os.environ["ETL_WSWE"])
    print(f"[etl] w_swe_mass = {lcfg['w_swe_mass']} (masse du manteau, CanSWE)")
if "ETL_WTWSCLIM" in os.environ:
    # Biais saisonnier GRACE par mois calendaire (R23). Premiere valeur jamais essayee
    # a 0.05 dans les configs : ce levier existe pour la balayer.
    lcfg["w_tws_clim"] = float(os.environ["ETL_WTWSCLIM"])
    print(f"[etl] w_tws_clim override = {lcfg['w_tws_clim']} (biais saisonnier GRACE)")
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
# ETL_ETI=1 : fonte au bilan radiatif reel (Enhanced Temperature Index, valide en
# juin sur SLSO). Le canal 6, qui porte la demande ET apprise, est REMPLACE par FB
# (sw_in W/m2, cache forcing-<reg>-swin.nc de build_swin_region.py) : sous
# ETL_ETP=linacre la demande est ignoree de toute facon (etp_channel=None), le canal
# est donc libre -- la colonne ETI lit sw_channel=6. Motif (R42) : l'amplitude
# saisonniere est un SUBSTITUT du cycle radiatif, faux en climat maritime ; la
# radiation reelle dissout ce scalaire comme le bulbe humide a dissous le seuil.
_ETI = os.environ.get("ETL_ETI", "0") == "1"
if _ETI:
    if os.environ.get("ETL_ETP", "appris") != "linacre":
        raise SystemExit("ETL_ETI=1 exige ETL_ETP=linacre (le canal 6 doit etre libre)")
    import xarray as _xsw
    _swf = f"{_paths.DATA_ROOT}/quebec/forcing-{REG}-swin.nc"
    if not os.path.exists(_swf):
        raise SystemExit(f"ETL_ETI=1 : {_swf} absent (lancer build_swin_region.py {REG.upper()})")
    _dsw = _xsw.open_dataset(_swf)
    _swv = torch.tensor(_dsw["forcing"].values[:, :, 0], dtype=F.dtype, device=F.device)
    _dsw.close()
    assert _swv.shape == F.shape[:2], (_swv.shape, F.shape)
    f7 = torch.cat([F[:, :, :6], _swv[:, :, None]], dim=2)
    print(f"[etl] ETI : canal 6 = FB reel (moy {float(_swv.mean()):.0f} W/m2) ; "
          f"la demande apprise n'est PAS embarquee")
else:
    f7 = torch.cat([F[:, :, :6], demand[:, :, None]], dim=2)

# ETL_SANS_PRELEV=1 : met les prelevements et rejets a ZERO. C'est le protocole de
# RENATURALISATION en miniature, et le seul moyen de comparer proprement avec/sans
# dans le code du jour -- comparer a un chiffre historique serait non apparie, le code
# ayant change entre-temps (conservation de la masse, graine, etc.).
# Rappel : pour que la renaturalisation ait un sens, le modele doit avoir ete CALE
# AVEC les prelevements. Sinon ses parametres les ont absorbes et les mettre a zero ne
# renaturalise rien.
_SANS_PRELEV = os.environ.get("ETL_SANS_PRELEV", "0") == "1"


def _sans_prelev(d):
    """Retourne une copie de d dont les prelevements sont nuls."""
    from meandre.routing.withdrawals import WithdrawalData
    return _dc_replace(d, withdrawals=WithdrawalData.zeros_like(d.withdrawals))


def with_forcing(d):
    # `swe_obs` MANQUAIT ICI du 2026-07-21 au 2026-08-21 (dette #14). Le defaut du
    # dataclass est None et `_need_snow` du trainer exige un tenseur, donc la perte
    # neige ne s'est jamais evaluee dans le pilote quebecois alors qu'il annonce
    # `w_snow = 0.3` a chaque run. Toute reconstruction manuelle d'un dataclass est un
    # piege de ce genre : preferer `dataclasses.replace`, qui ne peut rien oublier.
    return _dc_replace(d, forcing=f7)


td, vd = with_forcing(td), with_forcing(vd)
if _SANS_PRELEV:
    td, vd = _sans_prelev(td), _sans_prelev(vd)
    print("[etl] PRELEVEMENTS ET REJETS MIS A ZERO (renaturalisation)")
else:
    _wsum = float(td.withdrawals.net.abs().sum()) if hasattr(td.withdrawals, "net") else None
    print(f"[etl] prelevements et rejets ACTIFS"
          + (f" (somme des |termes| = {_wsum:.0f})" if _wsum is not None else ""))

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
    melt_mode=("eti" if os.environ.get("ETL_ETI", "0") == "1" else "degree_day"),
    # Phase probabiliste (ETL_QUANTILE=1) : tete quantile K=6, offsets monotones
    # depuis la mediane = Q_sim -- le KGE du socle est PRESERVE par construction.
    # Pipeline etabli sur SLSO (memoire pipeline_final) : backbone GELE, w_quantile,
    # best_metric nll. Ici on le porte sur le pilote quebecois pour la 1.0.
    use_quantile_head=(os.environ.get("ETL_QUANTILE", "0") == "1"),
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
_PLATB = f"{_paths.PLATFORMS_ROOT}"
_PROJ_M = f"{_PLATB}/{_MEMBRE}/{REG.upper()}_{_MEMBRE}_2020"
if _MEMBRE != "LN24HA":
    os.environ.setdefault("ETL_MELT_DIR", _PROJ_M)
    print(f"[etl] plateforme d'ancrage : {_MEMBRE} ({_PROJ_M.split('/')[-1]})")

# ETL_SEUIL_VALEUR impose un seuil EXPLICITE (en °C) au lieu de celui du projet.
# Motif (2026-08-20) : le profil mensuel du champion montre decembre en EXCES (1.207)
# et avril en DEFICIT (0.729), signature d'eau relachee en debut d'hiver au lieu d'etre
# stockee jusqu'a la crue. Le seuil du projet est a -2.2168 °C, donc tout ce qui est
# au-dessus compte comme PLUIE : un seuil aussi bas fabrique de la pluie en decembre.
if os.environ.get("ETL_SUBLIM", "0") == "1":
    # Sublimation Kuzmin (R32) : 15-40 mm/hiver attendus en foret boreale. Sortie
    # atmospherique exposee a diag.sublimation, comptee par ETL_BILAN.
    model.vertical_column.sublimation_mode = "kuzmin"
    print("[etl] SUBLIMATION du manteau : Kuzmin (u2, e_a du forcage)")
if "ETL_WET_CPROD" in os.environ:
    # Allonge la constante de vidange du reservoir de milieu humide (c_prod, en jours
    # au-dessus du volume normal ; defaut SWAT 10 j). LE candidat rapide-lent de R39 :
    # retenir la crue quelques SEMAINES -- le souterrain rend trop tard, le milieu
    # humide est deja conservatif et sur le chemin de la crue. L'attribut est lu par
    # la colonne AU MOMENT ou elle construit le bloc wetland (le dict n'existe pas
    # encore ici : premiere version plantee sur _static None, corrigee).
    model.vertical_column.c_prod_override = float(os.environ["ETL_WET_CPROD"])
    print(f"[etl] milieu humide : c_prod impose a "
          f"{float(os.environ['ETL_WET_CPROD']):.0f} jours (defaut 10)")
if "ETL_GW_POWER" in os.environ:
    # Nappe non lineaire Q = q_ref*(S/100)^b (R38) : "q_ref,b", ex. "2.6,2.2" --
    # jeu qui reproduit les DEUX residences mesurees (37 j plein, 111 j bas).
    _qr, _b = (float(x) for x in os.environ["ETL_GW_POWER"].split(","))
    model.vertical_column.gw_power_law = (_qr, _b)
    print(f"[etl] nappe NON LINEAIRE : q_ref {_qr} mm/j a 100 mm, exposant {_b}")
if os.environ.get("ETL_ETI", "0") == "1" and ("ETL_ETI_TF" in os.environ or "ETL_ETI_SRF" in os.environ):
    # Re-echelle des coefficients ETI. Les valeurs litterature (Pellicciotti 2005 :
    # tf 1.2 mm/degre/j, srf 9.4e-3 mm/j par W/m2) viennent de glaciers alpins d'ete ;
    # aux conditions de crue boreale (T ~5, SW ~250, albedo ~0.5) elles donnent
    # ~7 mm/j quand le degre-jour CALE d'Hydrotel en libere ~25 (taux 8.5 x indice_rad
    # x (1-albedo)). Le run eti-outv l'a paye : accumulation parfaite SANS sinusoide
    # (1.02/0.97/1.04) mais mai a 4.10 -- le manteau ne part plus, et deux scalaires
    # ne remontent pas d'un facteur 3.5 en dix epoques. On initialise donc a
    # l'equivalence energetique et on laisse l'apprentissage AFFINER.
    import math as _me
    with torch.no_grad():
        if "ETL_ETI_TF" in os.environ:
            _tf = float(os.environ["ETL_ETI_TF"])
            model.vertical_column.sp_tf.copy_(torch.tensor(_me.log(_me.expm1(_tf))))
        if "ETL_ETI_SRF" in os.environ:
            _srf = float(os.environ["ETL_ETI_SRF"])
            model.vertical_column.sp_srf.copy_(torch.tensor(_me.log(_me.expm1(_srf))))
    print(f"[etl] ETI re-echelle : tf {float(torch.nn.functional.softplus(model.vertical_column.sp_tf))*1000:.2f} mm/C/j | "
          f"srf {float(torch.nn.functional.softplus(model.vertical_column.sp_srf))*1000:.4f} mm/j/(W/m2)")
if "ETL_KREC_PRIOR" in os.environ:
    # Deplace la CIBLE d'ancrage du champ krec (banc G4-G6 : la phase GRACE se repare
    # a krec ~5e-5..1e-4 avec la vanne non lineaire, pas a 2e-5). La moyenne du champ
    # reste ancree, la variation spatiale reste libre.
    _t = getattr(model.spatial_encoder, "_prior_targets", None) or {}
    _t["krec"] = float(os.environ["ETL_KREC_PRIOR"])
    model.spatial_encoder._prior_targets = _t
    print(f"[etl] cible du prior krec -> {_t['krec']:.1e} m/h")
if "ETL_L3_EXP" in os.environ:
    # Drainage non lineaire de L3 (R37) : meme plafond a saturation, coupure en
    # dessous -> L3 respire, la recharge devient saisonniere. n=1 == fidele.
    model.vertical_column.l3_drain_exp = float(os.environ["ETL_L3_EXP"])
    print(f"[etl] drainage L3 NON LINEAIRE : exposant {float(os.environ['ETL_L3_EXP'])}")
if "ETL_SEUIL_TWB" in os.environ:
    # Partage pluie-neige au BULBE HUMIDE (generalisation du seuil, remarque d'Essi
    # sur R35) : un seuil unique en Twb remplace le seuil AIR par region. Sur nos 6
    # regions, le Twb equivalent au +0.3 air d'OUTV tient dans [-1.0, -0.7].
    model.vertical_column.split_mode = "wet_bulb"
    model.vertical_column.t_neige_seuil = float(os.environ["ETL_SEUIL_TWB"])
    print(f"[etl] partage pluie/neige au BULBE HUMIDE, seuil Twb "
          f"{float(os.environ['ETL_SEUIL_TWB']):+.2f} degres (Stull 2011, e_a du forcage)")
if "ETL_MELT_SAISON" in os.environ:
    # Modulation saisonniere du facteur de fonte (R32). Un degre-jour constant absorbe
    # le cycle annuel de radiation ; cale sur la crue, il fond trop en novembre-decembre
    # -- le deficit mesure (accumulation 58 % de CanSWE, fonte de coeur d'hiver juste).
    # s(j) = 1 + amp*sin(2*pi*(j-81)/365), moyenne annuelle 1. amp=0 = clone fidele.
    _msa = float(os.environ["ETL_MELT_SAISON"])
    model.vertical_column.melt_seasonal_amp = _msa
    print(f"[etl] fonte SAISONNIERE : amplitude {_msa} "
          f"(decembre x{1-_msa:.2f}, juin x{1+_msa:.2f}, moyenne annuelle inchangee)")
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
    from meandre.data.hydrotel_calib import imposed_retention_curve
    _plh = os.environ.get("ETL_MELT_DIR") or         _PROJ_M
    _cs = _lcs(_plh, r["node_ids"], 0.15, device=DEVICE)
    _cib = {"K_sat_1": _cs["ks1"].float() * 24, "K_sat_2": _cs["ks2"].float() * 24,
            "K_sat_3": _cs["ks3"].float() * 24, "porosity_1": _cs["thetas1"].float(),
            "porosity_2": _cs["thetas2"].float(), "porosity_3": _cs["thetas3"].float(),
            "Z2": _cs["z2"].float(), "Z3": _cs["z3"].float()}
    print(f"[etl] départ sur le champ Hydrotel ({os.path.basename(_plh)}) : ajustement du NeRF")
    model.spatial_encoder.fit_to_field(
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
            # La règle est dans imposed_retention_curve() pour que les diagnostics
            # appliquent EXACTEMENT la même (ils en divergeaient, cf. sa docstring).
            _courbe = imposed_retention_curve(_cs, _aq_actif)
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
        if False and "krec" in _cs:   # krec vient du CHAMP depuis 2026-08-20
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
    _cf = _pd.read_parquet(f"{_paths.DATA_ROOT}/quebec/champ_kgw_QC.parquet")
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
    _cf = _pd.read_parquet(f"{_paths.DATA_ROOT}/quebec/champ_freshet_QC.parquet")
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
# krec LIBRE veut dire krec APPRIS PAR LE CHAMP, donc ancre comme les autres sorties
# (Essi, 2026-08-22 : « krec devrait etre dans le nerf »). Sans ancrage il n'a que le
# debit pour juge, et le debit seul le pousse a zero (R11). Le prior tient la MOYENNE du
# champ en espace log et laisse la variation spatiale libre -- meme traitement que k_gw.
# Inutile quand krec est impose : la sortie du NeRF n'est alors pas utilisee, le calage
# etant fusionne par-dessus dans la colonne.
if os.environ.get("ETL_KREC_LIBRE", "0") == "1" and "ETL_KREC_GEL" not in os.environ:
    model.spatial_encoder.prior_on_krec = True
    print("[etl] krec APPRIS par le champ, moyenne ancree par le prior physique "
          "(cible 2e-5 m/h, ~34 % de debit de base ; le calage Hydrotel donne 1.3e-7)")

if "ETL_KREC" in os.environ:
    import math as _mk
    _kv = float(os.environ["ETL_KREC"])
    model.spatial_encoder.set_uniform_krec(_kv)
    print(f"[etl] krec pose UNIFORME -> {_kv:.0e} m/h (mode degrade : annule la "
          f"variation spatiale du champ, reserve aux bancs)")
    # ETL_KREC_GEL=1 : la recharge est un LIVRABLE, pas un bouton de calage. Le débit
    # seul la pousse aux extrêmes (banc du 2026-08-19 : à 5e-5 la nappe fournit 69 % du
    # débit et le KGE tombe à 0.589 ; à 1e-4 il devient négatif). Quand on la pose à une
    # valeur choisie pour des raisons PHYSIQUES, il faut la geler, sinon l'apprentissage
    # la déplace et on ne sait plus ce qu'on a mesuré.
    if os.environ.get("ETL_KREC_GEL", "0") == "1":
        model.spatial_encoder.freeze_krec()
        print(f"[etl] krec GELÉ à {_kv:.1e} (sortie exclue de l'apprentissage)")
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
    _plat = f"{_paths.PLATFORMS_ROOT}/LN24HA"
    _pl = os.environ.get("ETL_MELT_DIR") or f"{_plat}/{REG.upper()}_LN24HA_2020"
    try:
        from meandre.data.hydrotel_calib import (load_occupation_sol, load_milieux_humides,
                                                 load_phenologie)
        _lc = load_occupation_sol(_pl, r["node_ids"], device=DEVICE)
        _mh = load_milieux_humides(_pl, r["node_ids"], device=DEVICE)
        # ETL_SANS_MH=1 : DIAGNOSTIC seulement. Le couplage du milieu humide au troncon
        # est desequilibre (2026-08-20) : la colonne retire prod x wet_fr de la
        # production mais le reservoir ne recoit que prod x (wet_fr - wetsa/hru), et il
        # se voit crediter en echange la precipitation directe sur sa propre surface,
        # eau que le sol a deja traitee. La difference (apport - prod) x wetsa/hru est
        # creee ou detruite. Le reservoir lui-meme conserve EXACTEMENT sa masse (teste
        # sur 4 regimes) : c'est le couplage, porte fidelement de bv3c2.cpp, qui ne
        # ferme pas. Cet interrupteur mesure sa contribution a l'erreur de bilan.
        # Couplage conservatif par DEFAUT depuis le 2026-08-20. ETL_MH_FIDELE=1
        # restitue la formulation d'Hydrotel, qui perd 1.38 % de la precipitation, pour
        # les comparaisons module par module avec le binaire C++.
        if os.environ.get("ETL_MH_FIDELE", "0") == "1":
            model.vertical_column.mh_conservatif = False
            print("[etl] milieu humide FIDELE a Hydrotel (perd 1.38 % de la precipitation)")
        if os.environ.get("ETL_SANS_MH", "0") == "1":
            _mh = {}
            print("[etl] MILIEUX HUMIDES DESACTIVES (diagnostic de bilan)")
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
    _rw = _pdl.read_parquet(f"{_paths.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
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
    from recipe import set_lake_area_from_hydrolakes
    set_lake_area_from_hydrolakes(model, REG, td.territorial.get_physical("area_km2_local"), n_nodes)
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
    _rwp = _pdp.read_parquet(f"{_paths.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
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
    _pt = _Pl(f"{_paths.PLATFORMS_ROOT}/LN24HA/{REG.upper()}_LN24HA_2020/physitel/troncon.trl")
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
# CE QUI EST IMPRIME DOIT ETRE CE QUI TOURNE. La ligne annoncait `etp_channel=6
# (demande apprise)` EN DUR, donc y compris sous ETL_ETP=linacre|mcguinness qui posent
# `etp_channel = None` : tous les journaux du chantier quebecois affirmaient que la
# demande MLP etait active alors qu'elle etait ignoree par la colonne. Corrige le
# 2026-08-21 -- on lit l'attribut. Meme famille que la dette #3.
_ch = model.vertical_column.etp_channel
print(f"[etl] modèle {sum(p.numel() for p in model.parameters()):,} params | "
      + (f"etp_channel={_ch} (demande apprise × K_c NeRF)" if _ch is not None
         else f"ETP = {model.vertical_column.et_mode} (demande apprise CHARGEE MAIS IGNOREE)"))

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
    # la cle TOML s'appelle `grad_clip` ; `clip_grad_norm` (lu jusqu'au 2026-08-22)
    # n'existe dans aucune config -- benin car la valeur egale le defaut (1.0),
    # mais c'est le meme motif que la dette #16, attrape par le test de famille.
    grad_clip=float(tcfg.get("grad_clip", 1.0)),
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
    # Dette #16 (2026-08-22) : cette cle MANQUAIT dans l'enumeration, donc patience
    # restait au defaut 0 et l'arret anticipe ne s'est JAMAIS declenche dans le pilote
    # quebecois -- les 53 configs portant `patience = 8` n'y changeaient rien, et le run
    # aux-A a fait 10 epoques d'effondrement complet sans s'arreter. Troisieme instance
    # du motif « construction par enumeration » (dettes #14, #15).
    patience=int(tcfg.get("patience", 0)),
    val_every=1,
)
# CE QUI CONTRAINT REELLEMENT LE MODELE, lu dans l'objet de perte et dans les donnees,
# jamais dans la config (dette #15 : une ligne codee en dur a fait croire pendant des
# semaines que la demande ET apprise etait active alors que la colonne l'ignorait ;
# dette #14 : `w_snow = 0.3` etait annonce alors que la cible avait ete perdue en route).
# On imprime le poids ET la presence de la cible, cote a cote.
_lf = r["loss_fn"]
print("[etl] contraintes auxiliaires effectives :")
for _nom, _poids, _cible in (
        ("MODIS ET (MOD16)", getattr(_lf, "w_et", 0.0), td.et_obs),
        ("  mode ET", getattr(_lf, "et_mode", "?"), None),
        ("MODIS couverture nivale", getattr(_lf, "w_snow", 0.0), td.swe_obs),
        ("CanSWE masse du manteau", getattr(_lf, "w_swe_mass", 0.0), td.swe_mass_obs),
        ("GRACE TWS mensuel", getattr(_lf, "w_tws", 0.0), td.tws_obs),
        ("GRACE biais saisonnier", getattr(_lf, "w_tws_clim", 0.0), td.tws_obs)):
    if _nom == "  mode ET":
        print(f"        {_nom:<26s} {_poids}")
        continue
    _etat = ("ACTIVE" if (_poids > 0 and _cible is not None)
             else "eteinte" if _poids == 0 else "POIDS SANS CIBLE")
    print(f"        {_nom:<26s} poids {_poids:<6g} cible "
          f"{'presente' if _cible is not None else 'ABSENTE':<9s} -> {_etat}")

if os.environ.get("ETL_QUANTILE", "0") == "1":
    # GEL DU SOCLE : tout sauf la tete quantile. La mediane reste exactement le Q_sim
    # du point de reprise charge par ETL_WARM_FROM ; seule l'enveloppe s'apprend.
    _libres = 0
    for _nom, _p in model.named_parameters():
        if "quantile_head" in _nom:
            _p.requires_grad = True; _libres += _p.numel()
        else:
            _p.requires_grad = False
    r["loss_fn"].w_quantile = 1.0
    # le debit ne doit plus tirer le socle : tous les poids de debit a zero
    for _k in ("w_nse", "w_kge", "w_mse", "w_log_mse", "w_pbias", "w_peak",
               "w_et", "w_tws", "w_tws_clim", "w_snow"):
        if hasattr(r["loss_fn"], _k):
            setattr(r["loss_fn"], _k, 0.0)
    tconf.best_metric = "nll"
    tconf.w_prior = 0.0
    print(f"[etl] PHASE QUANTILE : socle gele, {_libres:,} parametres libres "
          f"(tete K=6), best_metric nll, pertes de debit a zero")

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
        _kv2 = float(os.environ["ETL_KREC"])
        model.spatial_encoder.set_uniform_krec(_kv2)
        print(f"[etl] krec ré-imposé après chargement : {_kv2:.1e} (gelé, uniforme)")
else:
    print(f"[etl] pas de point de reprise ({N_EPOCHS} époque(s)) : évaluation du modèle EN MÉMOIRE")
model.eval()
# ETL_CMP_NEIGE=1 demande aussi les diagnostics pour comparer le manteau simule aux
# releves CanSWE. La comparaison vit ICI, dans le pilote, et non dans un script annexe :
# les diagnostics qui vivaient a cote ont fini par mesurer un autre modele que le
# champion (fevrier annonce a 0.688 alors qu'il vaut 0.896).
_VEUT_NEIGE = os.environ.get("ETL_CMP_NEIGE", "0") == "1"
with torch.no_grad():
    if _VEUT_NEIGE:
        Q, _, _DIAG = model.simulate(
            forcing=f7, initial_state=HydroState.zeros(n_nodes, device=DEVICE),
            graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
            withdrawals=td.withdrawals, day_of_year=td.day_of_year, return_diagnostics=True)
    else:
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

# SCORE SUR LES JOURS REELLEMENT MESURES (R19, 2026-08-21).
# Le CEHQ publie a cote de chaque debit une remarque, et deux de ses codes disent que la
# valeur n'est pas une lecture de courbe de tarage : `E` (estimee) et `R` (corrigee pour
# effet de refoulement, donc sous glace). Mesure sur les 16 stations d'OUTV en tenue de
# cote : janvier 85.4 %, fevrier 87.3 %, mars 60.8 %, decembre 47.0 %, avril 9.5 %, zero
# de mai a octobre -- 21.3 % de la periode entiere. Une part de ce qu'on appelle l'erreur
# du modele en hiver est donc un desaccord avec la reconstruction d'un hydrologue.
# Ce bloc rejoue le MEME score en ne gardant que les jours mesures. L'ecart entre les
# deux chiffres est la part du classement qui repose sur des valeurs reconstruites.
# Enveloppe : un diagnostic ne doit JAMAIS couter l'evaluation qui le precede.
try:
    import duckdb as _dqf
    import pandas as _pdm   # aussi utilise par les blocs suivants
    _cf = _dqf.connect(_paths.data_path("quebec", f"{REG}.duckdb"), read_only=True)
    if "reconstructed" in [c[0] for c in _cf.execute("DESCRIBE observations").fetchall()]:
        _fl = _cf.execute(
            "SELECT station_id, date, reconstructed FROM observations "
            "WHERE reconstructed IS NOT NULL").df()
        _cf.close()
        _sids = r.get("station_ids")
        if _sids and len(_sids) == qo_test.shape[1] and len(_fl):
            _fl["station_id"] = _fl.station_id.astype(str)
            _piv = _fl.pivot_table(index="date", columns="station_id",
                                   values="reconstructed", aggfunc="first")
            _jours = _pdm.DatetimeIndex(
                times[np.flatnonzero(sl)[0]:np.flatnonzero(sl)[-1] + 1]).normalize()
            _piv.index = _pdm.DatetimeIndex(_piv.index).normalize()
            _rec = _piv.reindex(index=_jours, columns=_sids).to_numpy()
            _mes = (_rec == False)   # noqa: E712 -- NaN (pas de drapeau) exclu aussi
            _ks2, _ksh = [], []
            for _s in range(Qs.shape[1]):
                _v = (~torch.isnan(qo_test[:, _s]) & ~torch.isnan(Qs[:, _s])).numpy()
                _vm = _v & _mes[:, _s]
                if _v.sum() >= 60:
                    _ks2.append(float(kge_fn(qo_test[_v, _s], Qs[_v, _s])))
                if _vm.sum() >= 60:
                    _ksh.append(float(kge_fn(qo_test[_vm, _s], Qs[_vm, _s])))
            if _ksh:
                print(f"[etl] score sur les jours MESURES seulement (drapeaux CEHQ) : "
                      f"n={len(_ksh)} | median {np.median(_ksh):.4f} "
                      f"(contre {np.median(_ks2):.4f} tous jours confondus, "
                      f"ecart {np.median(_ksh)-np.median(_ks2):+.4f})")
                print(f"        jours reconstruits ecartes : "
                      f"{100*(1-_mes.sum()/max(np.isfinite(_rec.astype(float)).sum(),1)):.1f} % "
                      f"des jours drapeautes de la tenue de cote")
    else:
        _cf.close()
        print("[etl] drapeaux CEHQ absents de la base "
              "(lancer .runs/quebec/ingest_cehq_flags.py)")
except Exception as _efl:
    print(f"[etl] score sur jours mesures : impossible ({type(_efl).__name__}: {_efl})")

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
# ── AUDIT DE FERMETURE DU BILAN D'EAU (ETL_BILAN=1) ──────────────────────────
# Le depot teste la conservation de masse du ROUTAGE mais PAS celle de la colonne
# verticale. Or le deficit de debit est de -6.5 % par an, dont 86 % en avril, et les
# ecarts mensuels NE SE COMPENSENT PAS : il faut chercher une PERTE. Regle de decision
# ecrite AVANT la mesure : erreur < 0.1 % de la precipitation = ferme (bruit numerique),
# > 1 % = fuite reelle, entre les deux = suspect. Dette #42 du registre.
if os.environ.get("ETL_BILAN", "0") == "1":
    if not _VEUT_NEIGE:
        print("[etl] bilan : ETL_BILAN=1 exige ETL_CMP_NEIGE=1 (diagnostics requis)")
    else:
        _st = model.vertical_column._static["soil"]
        def _mmv(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        _z1, _z2, _z3 = _mmv(_st["z1"]), _mmv(_st["z2"]), _mmv(_st["z3"])
        _P_tot = f7[:, :, 0].cpu().numpy()
        _etr = _DIAG.etr.cpu().numpy()
        _lat = _DIAG.lateral_mm.cpu().numpy()
        # ATTENTION : _DIAG.wetland est un champ MORT en mode hydrotel (recopie tel quel
        # d'un pas a l'autre). Le vrai stock du milieu humide est _DIAG.wet_vol, expose
        # le 2026-08-20 ; son evaporation est _DIAG.etr_mh. Sans ces deux termes, le
        # bilan lisait une fuite de 1.97 % correlee a 0.48 avec la fraction de MH.
        _mh_st = (_DIAG.wet_vol.cpu().numpy() if _DIAG.wet_vol is not None else 0.0)
        _mh_ev = (_DIAG.etr_mh.cpu().numpy() if _DIAG.etr_mh is not None else None)
        _stock = (_DIAG.swe.cpu().numpy()
                  + 1000.0 * (_DIAG.theta1.cpu().numpy() * _z1
                              + _DIAG.theta2.cpu().numpy() * _z2
                              + _DIAG.theta3.cpu().numpy() * _z3)
                  + _DIAG.canopy.cpu().numpy() + _mh_st
                  + _DIAG.s_gw.cpu().numpy())
        _dS = _stock[-1] - _stock[0]
        _ent = _P_tot.sum(axis=0)
        _evmh = (_mh_ev.sum(axis=0) if _mh_ev is not None else np.zeros_like(_etr[0]))
        # sublimation (opt-in) : sortie atmospherique au meme titre que l'ETR
        _sub = (_DIAG.sublimation.cpu().numpy().sum(axis=0)
                if getattr(_DIAG, "sublimation", None) is not None else 0.0)
        _sor = _etr.sum(axis=0) + _lat.sum(axis=0) + _evmh + _sub
        _err = _ent - _sor - _dS
        _rel = _err / np.clip(_ent, 1e-9, None)
        print("[etl] BILAN D'EAU de la colonne, cumul sur toute la simulation (mm) :")
        print(f"        precipitation      {np.nanmean(_ent):10.0f}")
        print(f"        ETR                {np.nanmean(_etr.sum(axis=0)):10.0f}")
        print(f"        production         {np.nanmean(_lat.sum(axis=0)):10.0f}")
        print(f"        evap. milieu humide{np.nanmean(_evmh):10.0f}")
        print(f"        variation de stock {np.nanmean(_dS):10.0f}")
        print(f"        ERREUR             {np.nanmean(_err):10.0f}  "
              f"({100*np.nanmean(_rel):+.2f} % de la precipitation)")
        print(f"        par noeud : q10 {100*np.nanpercentile(_rel,10):+.2f} % | "
              f"med {100*np.nanmedian(_rel):+.2f} % | q90 {100*np.nanpercentile(_rel,90):+.2f} %")
        _abs = abs(float(np.nanmean(_rel)))
        # LA FUITE SUIT-ELLE LES FRACTIONS D'OCCUPATION ? Prediction : prod_surf pondere
        # lruis par fsa, leau par fse (eau libre) et lprec par fsi (impermeable), avec
        # leau = clamp(apport - etp, 0). L'ETR declaree au bilan est celle du noeud
        # ENTIER. Si les fractions ne somment pas a 1, ou si le plancher mord sur l'eau
        # libre, de l'eau se cree ou disparait. La fuite devrait alors suivre f_water
        # et f_urban.
        try:
            import pandas as _pdb
            _rwb = _pdb.read_parquet(f"{_paths.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
            _rwb = _rwb[_rwb.region == REG]
            if len(_rwb) == len(_rel):
                for _col in ("f_water", "f_urban", "f_wetland", "f_forest"):
                    _x = _rwb[_col].values
                    _ok = np.isfinite(_x) & np.isfinite(_rel)
                    _c = float(np.corrcoef(_x[_ok], _rel[_ok])[0, 1])
                    print(f"        correlation fuite / {_col:10s} : {_c:+.3f}")
                _q = _pdb.qcut(_rwb["f_water"].values, 4, duplicates="drop")
                print(f"        fuite par quartile de f_water :")
                for _b in sorted(set(_q)):
                    _k = (_q == _b)
                    print(f"          {str(_b):>18s} n={_k.sum():5d} "
                          f"{100*np.nanmean(_rel[_k]):+6.2f} %")
            else:
                print(f"        (fractions indisponibles : {len(_rwb)} vs {len(_rel)} noeuds)")
        except Exception as _eb:
            print(f"        (correlation impossible : {type(_eb).__name__}: {_eb})")
        _verdict = ("FERME (bruit numerique)" if _abs < 0.001 else
                    "FUITE REELLE" if _abs > 0.01 else "SUSPECT")
        print(f"        VERDICT : {_verdict}  (regle posee avant la mesure : "
              f"<0.1 % ferme, >1 % fuite)")

# ── CYCLE SAISONNIER DES STOCKS (ETL_STOCKS=1) ───────────────────────────────
# L'audit ci-dessus ferme le bilan sur le CUMUL : rien ne se perd sur la duree de la
# simulation. Il ne dit pas OU l'eau attend entre-temps. Or le defaut du champion est
# un DEPLACEMENT dans l'annee (avril a 0.729, decembre a 1.207) alors que le printemps
# n'est pas limite par l'apport : l'ecoulement d'avril-mai vaut 0.31 a 0.45 de la neige
# plus la pluie (R20). Le modele ne RETIENT donc pas en debut d'hiver ce qu'il devrait
# relacher au printemps ; reste a savoir lequel de ses reservoirs se vide. On sort la
# climatologie mensuelle de chaque stock et de chaque flux, en mm, moyennee sur les
# noeuds -- une question de FORME, sans echelle, donc lisible malgre R19 (de decembre a
# mars la cible observee est une interpolation a la main, pas une mesure).
if os.environ.get("ETL_STOCKS", "0") == "1":
    if not _VEUT_NEIGE:
        print("[etl] stocks : ETL_STOCKS=1 exige ETL_CMP_NEIGE=1 (diagnostics requis)")
    else:
        _sts = model.vertical_column._static["soil"]
        def _np1(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        _sz1, _sz2, _sz3 = _np1(_sts["z1"]), _np1(_sts["z2"]), _np1(_sts["z3"])
        _mois_tot = _pdm.DatetimeIndex(times).month.to_numpy()
        _stocks = {
            "manteau": _DIAG.swe.cpu().numpy(),
            "canopee": _DIAG.canopy.cpu().numpy(),
            "sol L1": 1000.0 * _DIAG.theta1.cpu().numpy() * _sz1,
            "sol L2": 1000.0 * _DIAG.theta2.cpu().numpy() * _sz2,
            "sol L3": 1000.0 * _DIAG.theta3.cpu().numpy() * _sz3,
            "milieu humide": (_DIAG.wet_vol.cpu().numpy()
                              if _DIAG.wet_vol is not None else None),
            "nappe": _DIAG.s_gw.cpu().numpy(),
        }
        _stocks = {k: v for k, v in _stocks.items() if v is not None}
        _flux = {
            "precipitation": f7[:, :, 0].cpu().numpy(),
            "apport au sol": _DIAG.snowmelt.cpu().numpy(),   # MAL NOMME : apport total
            "ETR": _DIAG.etr.cpu().numpy(),
            "production": _DIAG.lateral_mm.cpu().numpy(),
        }
        # PARTITION DE LA PRODUCTION (2026-08-21). La couche profonde n'a qu'UN exutoire
        # dans bv3c2 : `q3 = krec * z3 * theta3`, qui alimente directement prod_base. Avec
        # le krec uniforme du pilote (5e-5 m/h), z3 = 2.65 m et theta a saturation 0.433,
        # ce debit plafonne a 1.38 mm/j, soit ~503 mm/an -- du meme ordre que la production
        # TOTALE (549 mm/an). Deux lectures opposees en decoulent : ou bien L3 est un tuyau
        # sature qui ne stocke rien, ou bien elle travaille a pleine capacite et porte
        # l'essentiel de l'ecoulement. La partition tranche, le calcul ne le peut pas.
        for _cle, _nom in (("prod_surf", "  dont surface"), ("prod_hypo", "  dont hypoderm."),
                           ("prod_base", "  dont base")):
            _v = getattr(_DIAG, _cle, None)
            if _v is not None:
                _flux[_nom] = _v.cpu().numpy()
        if _DIAG.etr_mh is not None:
            _flux["evap. milieu humide"] = _DIAG.etr_mh.cpu().numpy()
        _n = len(_mois_tot)
        print("\n[etl] STOCKS, niveau moyen par mois (mm sur le bassin, moyenne des noeuds) :")
        print("        " + f"{'stock':>16s}" + "".join(f"{_m:>7d}" for _m in range(1, 13)))
        _cyc = {}
        for _nom, _v in _stocks.items():
            _c = np.array([np.nanmean(_v[:_n][_mois_tot == _m]) for _m in range(1, 13)])
            _cyc[_nom] = _c
            print("        " + f"{_nom:>16s}" + "".join(f"{_x:7.0f}" for _x in _c))
        _tot = np.sum(list(_cyc.values()), axis=0)
        print("        " + f"{'TOTAL':>16s}" + "".join(f"{_x:7.0f}" for _x in _tot))
        print("\n[etl] FLUX, lame moyenne par mois (mm/mois, moyenne des noeuds) :")
        print("        " + f"{'flux':>16s}" + "".join(f"{_m:>7d}" for _m in range(1, 13)))
        for _nom, _v in _flux.items():
            _c = np.array([np.nansum(_v[:_n][_mois_tot == _m])
                           / max((_mois_tot == _m).sum(), 1) * 30.4
                           / max(_v.shape[1], 1) for _m in range(1, 13)])
            print("        " + f"{_nom:>16s}" + "".join(f"{_x:7.1f}" for _x in _c))
        print("\n[etl] VARIATION decembre -> avril, par stock (mm ; negatif = le")
        print("      reservoir se VIDE pendant l'hiver au lieu de se remplir) :")
        for _nom, _c in sorted(_cyc.items(), key=lambda kv: kv[1][3] - kv[1][11]):
            print(f"        {_nom:>16s} : dec {_c[11]:8.0f} -> avr {_c[3]:8.0f}"
                  f"   {_c[3]-_c[11]:+8.0f}")
        print(f"        {'TOTAL':>16s} : dec {_tot[11]:8.0f} -> avr {_tot[3]:8.0f}"
              f"   {_tot[3]-_tot[11]:+8.0f}")

# ── AUDIT DES CONTRAINTES AUXILIAIRES (ETL_AUX=1) ────────────────────────────
# Motif (R23, 2026-08-21) : GRACE etait branche depuis toujours (w_tws=0.2 dans la
# config de base) et ne contraignait RIEN, parce que la perte juge des mois INDIVIDUELS
# a sigma=25 mm -- l'incertitude d'UNE observation -- alors que l'erreur est un biais
# saisonnier SYSTEMATIQUE moyenne sur vingt ans. Lu a la bonne echelle de bruit, le
# meme residu passe de 1.05 a 4.8 ecarts-types. Avant de toucher a quoi que ce soit, on
# pose la MEME question aux trois auxiliaires : quelle part du residu est un biais
# saisonnier repetable, et la perte le voit-elle ?
#
# Decomposition, pour chaque auxiliaire : residu quotidien moyen-bassin r(t), sa
# CLIMATOLOGIE mensuelle (partie systematique, celle qui ne s'efface pas en moyennant
# les annees) et la dispersion interannuelle autour d'elle (le bruit). La significativite
# du biais est systematique / (dispersion / sqrt(n_annees)). Un auxiliaire dont le biais
# est significatif mais dont la perte est deja satisfaite est une contrainte gaspillee.
if os.environ.get("ETL_AUX", "0") == "1":
    if not _VEUT_NEIGE:
        print("[etl] aux : ETL_AUX=1 exige ETL_CMP_NEIGE=1 (diagnostics requis)")
    else:
        _idx = _pdm.DatetimeIndex(times)
        _an_t, _mo_t = _idx.year.to_numpy(), _idx.month.to_numpy()

        def _audit(nom, sim, obs, unite, poids, sigma_perte, note=""):
            """sim, obs : (T,) moyenne-bassin alignees sur `times`, NaN admis."""
            _v = np.isfinite(sim) & np.isfinite(obs)
            if _v.sum() < 200:
                print(f"\n  {nom} : indisponible ({int(_v.sum())} pas valides)"); return
            _r = np.where(_v, sim - obs, np.nan)
            # climatologie mensuelle du residu = partie SYSTEMATIQUE
            _sys = np.array([np.nanmean(_r[_mo_t == _m]) for _m in range(1, 13)])
            # dispersion interannuelle autour de cette climatologie = BRUIT
            _moy_am, _nan = [], []
            for _m in range(1, 13):
                _va = [np.nanmean(_r[(_mo_t == _m) & (_an_t == _a)])
                       for _a in np.unique(_an_t)]
                _va = np.array([x for x in _va if np.isfinite(x)])
                _moy_am.append(_va.std() if len(_va) > 2 else np.nan)
                _nan.append(len(_va))
            _disp, _nann = np.array(_moy_am), np.array(_nan, float)
            _rms_sys = float(np.sqrt(np.nanmean(_sys ** 2)))
            _rms_tot = float(np.sqrt(np.nanmean(_r ** 2)))
            _sig_clim = _disp / np.sqrt(np.clip(_nann, 1, None))
            _z = np.abs(_sys) / np.clip(_sig_clim, 1e-9, None)
            print(f"\n  ── {nom} ({unite}) | poids actif {poids}"
                  + (f" | sigma de la perte {sigma_perte}" if sigma_perte else " | MSE brute, sans sigma")
                  + (f"\n     {note}" if note else ""))
            print("       mois " + "".join(f"{_m:>7d}" for _m in range(1, 13)))
            print("     biais  " + "".join(f"{_x:7.2f}" for _x in _sys))
            print("     ecart-t" + "".join(f"{_x:7.1f}" for _x in _z))
            print(f"     residu total {_rms_tot:.3f} | part SYSTEMATIQUE {_rms_sys:.3f} "
                  f"({100*_rms_sys/max(_rms_tot,1e-9):.0f} %) | reste = bruit")
            print(f"     biais saisonnier a {np.nanmax(_z):.1f} ecarts-types au pire mois "
                  f"(mois {int(np.nanargmax(_z))+1})")
            if sigma_perte:
                _lu = _rms_tot / float(sigma_perte)
                print(f"     LA PERTE LIT : {_lu:.2f} ecart-type"
                      + ("  -> DEJA SATISFAITE, ne pousse plus" if _lu < 1.5 else ""))

        # DEMANDE OU OFFRE ? (remarque d'Essi, 2026-08-21). Un residu d'ET ne dit pas
        # a lui seul d'ou il vient : la phenologie et la formule d'ETP fixent la DEMANDE,
        # l'humidite du sol fixe l'OFFRE, et le forcage porte les deux. Le rapport
        # ETR/ETP tranche : proche de 1, l'evaporation est bornee par la demande et le
        # sol ne joue pas ; nettement sous 1, c'est le sol qui retient. On imprime aussi
        # la saturation par couche, parce qu'un sol qui ne descend jamais sous ~0.9 de
        # sa porosite ne peut pas etre limitant, quel que soit le rapport.
        _etr_m = _DIAG.etr.cpu().numpy()
        _etp_m = _DIAG.etp.cpu().numpy() if _DIAG.etp is not None else None
        _sat = model.vertical_column._static["soil"]
        def _n3(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        print("\n[etl] DEMANDE OU OFFRE : rapport ETR/ETP et saturation du sol, par mois")
        print("      (ETR/ETP proche de 1 = borne par la DEMANDE, le sol ne retient pas ;")
        print("       saturation = theta / porosite, un sol au-dessus de ~0.9 n'est jamais limitant)")
        _por = _n3(model.spatial_encoder(td.node_coords, td.territorial.to_tensor()).porosity_1) \
            if hasattr(model, "spatial_encoder") else None
        print("        " + f"{'':>14s}" + "".join(f"{_m:>7d}" for _m in range(1, 13)))
        if _etp_m is not None:
            _r = [np.nansum(_etr_m[_mo_t == _m]) / max(np.nansum(_etp_m[_mo_t == _m]), 1e-9)
                  for _m in range(1, 13)]
            print("        " + f"{'ETR/ETP':>14s}" + "".join(f"{_x:7.2f}" for _x in _r))
        for _nom, _th, _zz in (("sat. L1", _DIAG.theta1, "1"), ("sat. L2", _DIAG.theta2, "2"),
                               ("sat. L3", _DIAG.theta3, "3")):
            _t = _th.cpu().numpy()
            _p = _n3(getattr(model.spatial_encoder(td.node_coords, td.territorial.to_tensor()),
                             f"porosity_{_zz}"))
            _s2 = _t / np.clip(_p, 1e-9, None)
            print("        " + f"{_nom:>14s}"
                  + "".join(f"{np.nanmean(_s2[_mo_t == _m]):7.2f}" for _m in range(1, 13)))
            # UNE MOYENNE ARRONDIE A 1.00 NE DIT PAS SI LE SOL EST *EPINGLE* A SATURATION.
            # La difference compte : un reservoir a 0.95 respire encore un peu, un
            # reservoir epingle a la porosite est un tuyau et n'a aucune capacite. On
            # imprime donc la valeur absolue et la part des cas colles a la borne.
            print("        " + f"{'':>14s}   theta med {np.nanmedian(_t):.4f} | "
                  f"porosite med {np.nanmedian(_p):.4f} | "
                  f"epingle (>=0.999) {100*np.nanmean(_s2 >= 0.999):.1f} % "
                  f"des couples noeud-jour")

        print("\n[etl] AUDIT DES CONTRAINTES AUXILIAIRES")
        print("      biais = climatologie mensuelle du residu simule-observe (partie qui")
        print("      ne s'efface PAS en moyennant les annees) ; ecart-t = sa significativite")
        print("      contre la dispersion interannuelle, dispersion/sqrt(n annees).")
        # ET : MOD16, moyenne 8 jours des deux cotes (meme appariement que la perte)
        # MOYENNE-BASSIN QUI IGNORE LES MANQUANTS. `.mean(dim=1)` propage : MODIS a des
        # trous PAR NOEUD, donc un seul noeud absent suffit a rendre NaN la moyenne du
        # jour entier, et l'audit lisait 0 pas valide sur 9000.
        def _moy_bassin(x):
            a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            return np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=1)

        def _cale(serie_t0):
            """place une serie qui demarre a t0 sur l'axe complet `times`."""
            out = np.full(len(times), np.nan)
            out[t0:t0 + len(serie_t0)] = serie_t0
            return out

        _eo = getattr(td, "et_obs", None)
        if _eo is not None:
            _audit("MODIS ET (MOD16)", _moy_bassin(_DIAG.etr), _cale(_moy_bassin(_eo)),
                   "mm/j", lcfg.get("w_et", 0.0), None,
                   "w_et=0 dans la recette du champion : contrainte DEBRANCHEE")
        else:
            print("\n  MODIS ET : non charge")
        # Neige : fraction de couverture, meme transformation que la perte.
        # LU DIRECTEMENT EN BASE : `with_forcing` (l.152) reconstruit TrainingData sans
        # recopier `swe_obs`, donc td.swe_obs est None dans tout le pilote quebecois et
        # `_need_snow` du trainer est toujours faux. On mesure la contrainte telle
        # qu'elle SERAIT, sans rien changer au comportement d'entrainement.
        _so = getattr(td, "swe_obs", None)
        if _so is None:
            try:
                from meandre.data.basin_cache import BasinCache as _BC
                _so = _BC(_paths.data_path("quebec", f"{REG}.duckdb")).load_modis_snow(
                    times[0], times[-1], device="cpu")
                if _so is not None:
                    print("\n  (neige relue en base : td.swe_obs est None, voir with_forcing l.152)")
            except Exception as _es_:
                print(f"\n  MODIS neige : relecture impossible ({type(_es_).__name__}: {_es_})")
                _so = None
        if _so is not None:
            _ss = _moy_bassin(1.0 - torch.exp(-_DIAG.swe / 15.0))
            _sb = _moy_bassin(_so)
            _audit("MODIS couverture nivale", _ss,
                   _sb if len(_sb) == len(_ss) else _cale(_sb),
                   "fraction 0-1", lcfg.get("w_snow", 0.0), None,
                   "JAMAIS ACTIVE : with_forcing perd swe_obs, donc _need_snow est faux")
        else:
            print("\n  MODIS neige : non charge")
        # GRACE : deja etabli par R23, on le repasse au meme moule pour comparaison
        _to = getattr(td, "tws_obs", None)
        if _to is not None:
            _sts2 = model.vertical_column._static["soil"]
            def _n2(x):
                return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            _stw = (_DIAG.swe.cpu().numpy() + _DIAG.canopy.cpu().numpy()
                    + 1000.0 * (_DIAG.theta1.cpu().numpy() * _n2(_sts2["z1"])
                                + _DIAG.theta2.cpu().numpy() * _n2(_sts2["z2"])
                                + _DIAG.theta3.cpu().numpy() * _n2(_sts2["z3"]))
                    + (_DIAG.wet_vol.cpu().numpy() if _DIAG.wet_vol is not None else 0.0)
                    + _DIAG.s_gw.cpu().numpy()).mean(axis=1)
            _tob = np.full_like(_stw, np.nan)
            _tmp = _to.cpu().numpy()
            _tob[t0:t0 + len(_tmp)] = _tmp
            # centrage long terme des DEUX cotes, comme tws_anomaly_loss
            _vv = np.isfinite(_stw) & np.isfinite(_tob)
            _audit("GRACE TWS", _stw - np.nanmean(_stw[_vv]), _tob - np.nanmean(_tob[_vv]),
                   "mm", lcfg.get("w_tws", 0.0), 25.0,
                   "R23 : sigma=25 est l'incertitude d'UN mois, pas de la climatologie")
        else:
            print("\n  GRACE : non charge")

# ECART EN VOLUME, mois par mois. Un RAPPORT ne dit rien du volume qu'il represente :
# 27 % du volume d'avril, mois de crue, pese bien plus que 20 % de celui de decembre.
# Sans cette table on croit voir un deplacement d'eau la ou il y a une perte nette
# (confusion commise le 2026-08-20 : "il relache l'eau d'avril en decembre").
_vol_o = np.zeros(12); _vol_s = np.zeros(12)
for _m in range(1, 13):
    _k = _mo == _m
    if _k.sum() < 10:
        continue
    _vo = np.nansum(_qo_np[_k]); _vs = np.nansum(_qs_np[_k])
    _vol_o[_m - 1], _vol_s[_m - 1] = _vo, _vs
_tot_o = _vol_o.sum()
print("[etl] ecart en VOLUME par mois (somme sur stations et annees, m3/s-jours) :")
print(f"        {'mois':>5s} {'observe':>10s} {'simule':>10s} {'ecart':>10s} {'% du total annuel':>18s}")
for _m in range(12):
    if _vol_o[_m] == 0:
        continue
    _e = _vol_s[_m] - _vol_o[_m]
    print(f"        {_m+1:5d} {_vol_o[_m]:10.0f} {_vol_s[_m]:10.0f} {_e:+10.0f} "
          f"{100*_e/max(_tot_o,1e-9):17.1f} %")
_en = _vol_s.sum() - _tot_o
print(f"        {'ANNEE':>5s} {_tot_o:10.0f} {_vol_s.sum():10.0f} {_en:+10.0f} "
      f"{100*_en/max(_tot_o,1e-9):17.1f} %")
print("[etl] lecture : si les ecarts mensuels se COMPENSENT (total ~0), l'eau est")
print("      deplacee dans le temps. S'ils s'additionnent, il en manque vraiment.")
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

# ── NEIGE MESUREE : le manteau simule contre les releves CanSWE ──────────────
# Question tranchee ici : la crue printaniere arrive avec un mois de retard (avril
# 0.729, mai 1.07-1.42). Est-ce que le manteau FOND TROP TARD, ou n'a-t-il jamais eu
# la BONNE MASSE ? Le debit ne peut pas repondre, il est deja la variable ajustee.
# CanSWE est une MESURE : aucune circularite. Contrainte de TENDANCE et de TIMING,
# jamais de NIVEAU (un site est souvent en clairiere, le troncon porte sa foret).
if _VEUT_NEIGE:
    from meandre.data.basin_cache import BasinCache as _BC
    from joint_data import _paths as _jp
    _db = _jp(REG)[0]
    _mes, _sit = _BC(_db).load_canswe(_HO[0].strip(), _HO[1].strip())
    if _mes is None or _mes.empty:
        print("[etl] neige : aucune mesure CanSWE dans la fenetre")
    else:
        _swe = _DIAG.swe.cpu().numpy()                      # (T, n_noeuds), mm
        _tt = _pdm.DatetimeIndex(times)
        _pos = {d: k for k, d in enumerate(_tt.normalize())}
        _m = _mes.copy()
        _m["t"] = _m["date"].map(lambda x: _pos.get(_pdm.Timestamp(x).normalize()))
        _m = _m[_m["t"].notna()]
        _m["t"] = _m["t"].astype(int)
        _m["sim"] = _swe[_m["t"].values, _m["node_idx"].values]
        _m = _m[np.isfinite(_m["sim"]) & np.isfinite(_m["swe_mm"])]
        print(f"[etl] neige : {len(_m):,} couples simule/mesure sur "
              f"{_m.swe_station_id.nunique()} sites")
        _m["mois"] = _pdm.DatetimeIndex(_m["date"]).month
        print("[etl] neige, simule/mesure par mois :", " ".join(
            f"{_mo:02d}={_g.sim.sum()/max(_g.swe_mm.sum(), 1e-9):.2f}"
            for _mo, _g in _m.groupby("mois") if len(_g) >= 20))
        # MASSE : rapport des pics annuels. TIMING : date de disparition (< 10 mm).
        _m["an"] = _pdm.DatetimeIndex(_m["date"]).year
        print(f"{'annee':>6s} {'pic mes':>8s} {'pic sim':>8s} {'ratio':>6s} "
              f"{'dispar. mes':>12s} {'dispar. sim':>12s} {'ecart':>6s}")
        for _an, _g in _m.groupby("an"):
            _j = _pdm.DatetimeIndex(_g["date"]).dayofyear
            # PIC : mediane des pics PAR SITE, et non maximum global. Comparer deux
            # maxima revient a comparer des sites et des jours differents ; la premiere
            # version de ce bloc le faisait (2026-08-20), et une variante encore pire
            # comparait un maximum de site a une precipitation MOYENNE de domaine, ce
            # qui suggerait a tort un forcage insuffisant.
            _pp = _g.groupby("swe_station_id").agg(mes=("swe_mm", "max"), sim=("sim", "max"))
            _pp = _pp[(_pp.mes > 0) & (_pp.sim >= 0)]
            _pm, _ps = float(_pp.mes.median()), float(_pp.sim.median())
            _hiv = _g[(_j >= 60) & (_j <= 180)]
            if _hiv.empty:
                continue
            _jm = _pdm.DatetimeIndex(_hiv["date"]).dayofyear
            _vm = _hiv.groupby(_jm)["swe_mm"].mean()
            _vs = _hiv.groupby(_jm)["sim"].mean()
            def _fin(serie):
                _apres = serie[serie.index >= serie.idxmax()]
                _sous = _apres[_apres < 10.0]
                return int(_sous.index[0]) if len(_sous) else -1
            _dm, _ds = _fin(_vm), _fin(_vs)
            print(f"{_an:6d} {_pm:8.0f} {_ps:8.0f} {_ps/max(_pm,1e-9):6.2f} "
                  f"{_dm:12d} {_ds:12d} {_ds-_dm if _dm>0 and _ds>0 else 0:+6d}")
        print("[etl] lecture : ratio de pic < 1 = masse manquante ; ecart de "
              "disparition > 0 = fonte TARDIVE (en jours juliens)")
        # NOS PERTES HIVERNALES SONT-ELLES EXCESSIVES ? Reference MESUREE sur CanSWE
        # seul : un manteau REEL perd 24.6 % de ce qu'il accumule entre decembre et
        # fevrier (15 719 intervalles d'OUTV). On imprime ici la meme quantite pour le
        # modele. Sans cette reference, "notre manteau est trop leger" ne se juge pas.
        # On somme les BAISSES de SWE, exactement comme la reference CanSWE, et sur la
        # MEME fenetre que le reste de l'evaluation. Ne PAS utiliser _DIAG.snowmelt :
        # ce champ est mal nomme, il contient l'apport TOTAL au sol (donc la pluie hors
        # saison nivale), et le lire comme une fonte donnait 1741 % du pic.
        # APPARIEMENT STRICT : on differencie le simule AUX MEMES DATES que les releves,
        # site par site, exactement comme la mesure. Differencier la serie quotidienne
        # complete donnait 46.7 % contre 24.6 % mesures, mais un echantillonnage plus
        # fin capte plus d'oscillations : le confondant valait a lui seul un facteur
        # proche de 2. Ici il est elimine, il ne reste que point contre moyenne de
        # troncon (2026-08-20).
        _gs = _ps = _gm = _pm2 = 0.0
        _nint = 0
        for _sid, _g in _m.sort_values("date").groupby("swe_station_id"):
            _dt = _pdm.DatetimeIndex(_g["date"])
            _vs, _vm = _g["sim"].values, _g["swe_mm"].values
            for _k in range(1, len(_g)):
                _nj = (_dt[_k] - _dt[_k - 1]).days
                if not (1 <= _nj <= 31) or _dt[_k].month not in (12, 1, 2):
                    continue
                _ds, _dm = float(_vs[_k] - _vs[_k - 1]), float(_vm[_k] - _vm[_k - 1])
                _gs += max(_ds, 0.0); _ps += max(-_ds, 0.0)
                _gm += max(_dm, 0.0); _pm2 += max(-_dm, 0.0)
                _nint += 1
        # OU PASSE LA NEIGE ENTRE LA CHUTE ET LE MANTEAU ? Les deux mesures precedentes
        # encadrent cette etape sans la couvrir : l'apport est ample (la regle fournit
        # 1.14 a 1.24 fois l'accumulation mesuree) et la perte APRES accumulation est
        # normale (25.2 % contre 22.9 %), mais le manteau simule n'accumule que 59 % du
        # mesure. On cumule donc, sur LES MEMES intervalles, la neige que la regle
        # produit au noeud du site (2026-08-20).
        _neige_reg = 0.0
        _sr = 3.0                                  # marge : jours de l'intervalle
        for _sid, _g in _m.sort_values("date").groupby("swe_station_id"):
            _dt = _pdm.DatetimeIndex(_g["date"]); _nd0 = int(_g["node_idx"].iloc[0])
            for _k in range(1, len(_g)):
                _nj = (_dt[_k] - _dt[_k - 1]).days
                if not (1 <= _nj <= 31) or _dt[_k].month not in (12, 1, 2):
                    continue
                _i0 = _pos.get(_pdm.Timestamp(_dt[_k - 1]).normalize())
                _i1 = _pos.get(_pdm.Timestamp(_dt[_k]).normalize())
                if _i0 is None or _i1 is None or _i1 <= _i0:
                    continue
                _tn = f7[_i0 + 1:_i1 + 1, _nd0, 1].cpu().numpy()
                _tx = f7[_i0 + 1:_i1 + 1, _nd0, 2].cpu().numpy()
                _pp = f7[_i0 + 1:_i1 + 1, _nd0, 0].cpu().numpy()
                _sseuil = float(model.vertical_column.t_neige_seuil)
                _r = np.clip((_tx - _sseuil) / (_tx - _tn + 1e-6), 0.0, 1.0)
                _r = np.where(_tx < _sseuil, 0.0, _r)
                _r = np.where(_tn >= _sseuil, 1.0, _r)
                _neige_reg += float(np.nansum(_pp * (1.0 - _r)))
        if _nint:
            print(f"[etl] neige tombee selon la regle sur ces memes intervalles : "
                  f"{_neige_reg:.0f} mm")

        if _nint:
            print(f"[etl] neige, pertes dec-fev sur {_nint} intervalles APPARIES "
                  f"(memes dates, memes sites) :")
            print(f"        simule : hausses {_gs:7.0f} mm | baisses {_ps:7.0f} mm | "
                  f"perte {_ps / max(_gs, 1e-9):5.1%}")
            print(f"        mesure : hausses {_gm:7.0f} mm | baisses {_pm2:7.0f} mm | "
                  f"perte {_pm2 / max(_gm, 1e-9):5.1%}")
        print("[etl] reference MESUREE (CanSWE seul) : un manteau reel perd 24.6 % de "
              "ce qu'il accumule de decembre a fevrier")

        # LE DEFICIT DE MASSE EST-IL REEL, OU EST-CE LA REPRESENTATIVITE DU SITE ?
        # Un site nivometrique est souvent en clairiere alors que le troncon porte sa
        # foret, et la canopee intercepte : la litterature donne 20-40 % d'ecart en
        # foret boreale, soit exactement l'ampleur mesuree. Si le deficit SUIT la
        # fraction forestiere, c'est l'interception et il n'y a rien a corriger ;
        # s'il est UNIFORME, la masse manque vraiment.
        # f_forest est une colonne NORMALISEE du tenseur territorial ; get_physical()
        # ne rend que les quelques colonnes de-normalisees. On lit donc la valeur BRUTE
        # dans la base. Et on imprime l'echec au lieu de l'avaler : la premiere version
        # de ce bloc ne s'est jamais executee, sans un mot (2026-08-20).
        _ff = None
        try:
            import duckdb as _dd
            _cf = _dd.connect(_db, read_only=True)
            _ff = _cf.execute(
                "SELECT f_forest FROM territorial ORDER BY node_idx").df()["f_forest"].values
            # La colonne de la base est NORMALISEE (elle va de -3.7 a 1.0 sur OUTV),
            # comme mean_elevation_m. Les quartiles restent valides -- une
            # normalisation est croissante -- mais les bornes affichees seraient
            # illisibles. On revient donc aux fractions REELLES du parquet provincial.
            import pandas as _pdf
            _rwf = _pdf.read_parquet(f"{_paths.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
            _rwf = _rwf[_rwf.region == REG]
            if len(_rwf) == len(_ff):
                _ff = _rwf["f_forest"].values
            _cf.close()
        except Exception as _e:
            print(f"[etl] neige : fraction forestiere indisponible ({type(_e).__name__}: {_e})")
        if _ff is not None:
            _m["f_foret"] = _ff[_m["node_idx"].values]
            _hiver = _m[_pdm.DatetimeIndex(_m["date"]).month.isin([1, 2, 3])]
            _q = _pdm.qcut(_hiver["f_foret"], 4, duplicates="drop")
            print(f"{'fraction forestiere':>22s} {'n':>6s} {'sim/mes':>8s}")
            for _b, _g in _hiver.groupby(_q, observed=True):
                print(f"{str(_b):>22s} {len(_g):6d} "
                      f"{_g.sim.sum()/max(_g.swe_mm.sum(), 1e-9):8.2f}")
            print("[etl] lecture : ratio qui DECROIT avec la foret = interception "
                  "(attendu, rien a corriger) ; ratio PLAT = masse reellement manquante")

# COMPARAISON AU MEMBRE HYDROTEL, memes stations, memes jours, meme protocole.
# ETL_CMP_HYDROTEL=MG24HK. Sert a repondre a UNE question : un ecart mensuel est-il
# NOTRE defaut ou un defaut PARTAGE (donc du forcage) ? Sans ce cote a cote, on repare
# a l'aveugle. Ajoute le 2026-08-20 apres constat d'un retard d'un mois sur la crue.
if os.environ.get("ETL_CMP_HYDROTEL"):
    import xarray as _xrh
    from meandre.data.hydrotel_calib import appariement_provincial as _appm
    _mb = os.environ["ETL_CMP_HYDROTEL"]
    try:
        _z = _xrh.open_zarr(f"{_paths.RQH_ROOT}/"
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
