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

# GRAINE. Le pilote REGIONAL en pose une depuis le 2026-08-20 ; le pilote provincial
# n'en posait AUCUNE, et personne ne l'avait vu. Consequence mesuree le 2026-08-28 :
# deux runs de la MEME recette (meme chaine de configuration, memes 100 323 parametres)
# donnent kge_med 0.3481 et 0.3566 a l'epoque 0, puis 0.4378 et 0.2618 a l'epoque 1.
# Les poids du champ etaient tires au hasard a chaque lancement, donc AUCUNE des
# comparaisons provinciales dites appariees de la journee ne l'etait reellement : elles
# comparaient deux initialisations differentes. Le registre etablit deja cette lecon
# pour le pilote regional, ou le nuage vaut ~0.025 ; sur la province il est bien plus
# large. ETL_SEED change le tirage quand on veut MESURER la dispersion.
import random as _rnd
import numpy as _npseed
import torch as _tseed
_GRAINE = int(os.environ.get("ETL_SEED", "1234"))
_rnd.seed(_GRAINE); _npseed.random.seed(_GRAINE)
_tseed.manual_seed(_GRAINE); _tseed.cuda.manual_seed_all(_GRAINE)
print(f"[province] graine = {_GRAINE}", flush=True)

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
# Recette gen1 : w_et 0.4 (et non le 1.0 du TOML de base), MODIS pour la FORME.
lcfg["w_et"] = float(os.environ.get("PROV_WET", "0.4"))
# PROV_AUX : degonfle EN BLOC les trois contraintes auxiliaires. Defaut 1.0, donc rien
# ne change sans le demander. MESURE du 2026-08-28 sur le banc provincial : le prior
# pese 5045 % du total de la perte de debit, le GRACE climatologique 3548 % et le GRACE
# mensuel 1959 %. Ce ne sont pas trois problemes mais un seul -- des contraintes qui
# optimisent A LA PLACE de l'hydrogramme -- et toutes les metriques se degradent de
# facon monotone des l'epoque 1. On les traite donc EN BLOC : la question posee est
# binaire, la balance de la perte est-elle la cause de la divergence. Si oui, repartir
# entre les trois est un banc court et facile ; si non, les suspects restants sont la
# montee du taux d'apprentissage vers 5e-4, que le registre designe depuis quatre runs.
# PROV_SANS_PRELEV=1 : couper les prelevements, en ENTRAINEMENT comme en evaluation.
# Diagnostic du 2026-08-29. Le run provincial du 26 aout rendait 0.6193 de KGE median
# tenu de cote ; ceux du 28 aout rendent 0.43 a nombre d'epoques egal, et l'effondrement
# epargne OUTV tout en frappant mont, sagu et slso. Or la reingestion des prelevements
# du 26 aout (8e02622) a multiplie la couverture par deux et demi, et ces plateformes
# sont precisement celles qui portent le plus de sites : slso 1087, slno 1017, sagu 234,
# quand OUTV en porte peu. Ce levier teste la correspondance au lieu de la supposer.
_SANS_PRELEV = os.environ.get("PROV_SANS_PRELEV", "0") == "1"
_AUX = float(os.environ.get("PROV_AUX", "1.0"))
if _AUX != 1.0:
    for _k in ("w_tws", "w_tws_clim"):
        lcfg[_k] = float(lcfg.get(_k, 0.0)) * _AUX
    print(f"[province] contraintes auxiliaires degonflees x{_AUX:g} : "
          f"w_tws={lcfg['w_tws']:.4g} w_tws_clim={lcfg['w_tws_clim']:.4g}", flush=True)

print(f"[province] plateformes fondues : {PLATEFORMES} | device {DEVICE}", flush=True)
dom = load_domain(PLATEFORMES, lcfg, device=DEVICE)

# ── FONTE ETI (PROV_ETI=1, R51-R52) ──────────────────────────────────────────
# Le degre-jour fond des qu'un redoux passe, sans verifier que l'energie disponible le
# justifie : en decembre, jours courts et rayonnement faible, il produit une fonte
# fantome. Mesure sur la MASSE CanSWE a SAGU : 0.77 en decembre et 0.60 en avril, contre
# 0.86 et 0.89 pour l'ETI aux coefficients de LITTERATURE, sans aucun calage. L'ETI avait
# ete rejete le 24 aout (R44) mais sur le DEBIT SEUL, qui ne peut pas voir un decalage de
# calendrier de deux a trois mois puisqu'il se compense en partie dans le bilan annuel.
# Le bon juge pour une question de calendrier est un calendrier.
ETI = os.environ.get("PROV_ETI", "0") == "1"

# ── VERROU DE FONTE APPRIS (PROV_CANOPEE=1, R56) ─────────────────────────────
# Hydrotel place son freshet avec un seuil de fonte PAR CLASSE cale contre le debit
# (+3.35 degres sous conifere sur sagu/outv/abit : 0.7 % des jours de decembre a mars
# le franchissent, contre 4.1 % au seuil physique). On garde ses TAUX ancres et on
# rend ce seuil au NeRF, sous forme de deux retards non negatifs empiles qui imposent
# conifere >= feuillu >= decouvert.
CANOPEE = os.environ.get("PROV_CANOPEE", "0") == "1"
N_FORCAGE = 6
if ETI:
    import xarray as _xr
    from meandre.utils import paths as _p
    _sw = []
    for _nom, (_a, _b) in dom["slices"].items():
        _f = _p.data_path("quebec", f"forcing-{_nom}-swin.nc")
        if not os.path.exists(_f):
            raise SystemExit(f"PROV_ETI=1 mais pas de cache SW_in pour {_nom} : "
                             f"lancer .runs/quebec/build_swin_region.py {_nom.upper()}")
        _d = _xr.open_dataset(_f)
        _sw.append(torch.tensor(_d["forcing"].values[:, :, 0], dtype=torch.float32))
        _d.close()
    _SW = torch.cat(_sw, dim=1).to(DEVICE)
    _F6 = dom["train_data"].forcing[:]
    _F7 = torch.cat([_F6, _SW[:, :, None]], dim=2)
    for _k in ("train_data", "val_data"):
        dom[_k].forcing = _F7
    N_FORCAGE = 7
    print(f"[province] fonte ETI : forcage etendu a 7 canaux (SW_in), "
          f"tf et srf aux valeurs de litterature", flush=True)

model = HydroModel(
    n_nodes=dom["n_nodes"],
    n_territorial=dom["territorial"].n_features,
    n_forcing=N_FORCAGE,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=bool(mcfg.get("use_frost_rankinen", True)),
    column_theta_init_frac=float(mcfg.get("column_theta_init_frac", 0.9)),
    param_mode="nerf", column_mode="hydrotel",
    # recette gen1 : ETP Linacre calee du projet. PROV_SANS_LINACRE bascule le MODE en
    # meme temps que les parametres, sinon la colonne exige un set_linacre_params qui
    # n'a pas eu lieu et l'assertion tombe au premier pas.
    et_mode=("mcguinness" if os.environ.get("PROV_SANS_LINACRE", "0") == "1"
             else "linacre"),
    use_temperature=False,
    # z_n desactive : un effet aleatoire par troncon sur 25 656 troncons ajouterait
    # autant de parametres libres que de noeuds sans contrainte spatiale, et c'est
    # exactement ce que le champ continu doit porter a sa place.
    use_latent_codes=False,
    spatial_melt=bool(mcfg.get("spatial_melt", True)),
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=bool(mcfg.get("predict_lake_params", True)),
    # PROV_COMPILE=0 : desactive torch.compile sur le sol. Necessaire ici, ou la
    # compilation Triton echoue (`ptxas failed`, NoTritonConfigsError) et emporte le run
    # entier -- constate le 2026-08-26 sur le temoin de la borne. C'est probablement
    # aussi ce qui expliquait la variante sans depart sol a 40 min contre 16 : une
    # compilation qui echoue retombe sur un chemin lent sans le dire.
    compile_soil=(os.environ.get("PROV_COMPILE", "1") == "1"
                  and bool(mcfg.get("compile_soil", True))),
    use_aquifer=True,
    melt_mode=("eti" if ETI else "degree_day"),
    canopy_melt_lag=CANOPEE,
).to(DEVICE)

# Occupation du sol, milieux humides et fonte calée, livrés par plateforme et fondus
# par le chargeur. Sans eux la physique reçoit 0 % de forêt et 0 % d'eau libre, et
# OUTV perd 0.27 de KGE (mesure du 2026-08-10). joint.py ne les posait pas.
# PROV_SANS_* : interrupteurs de DIAGNOSTIC seulement. L'alignement sur la recette
# gen1 a fait passer l'epoch de 10 a 36 minutes sur le meme domaine ; ils servent a
# designer lequel des trois ajouts porte ce cout. Aucun n'a vocation a rester pose.
if dom.get("land_cover") and os.environ.get("PROV_SANS_MH", "0") != "1":
    model.vertical_column.set_land_cover(dom["land_cover"])
elif os.environ.get("PROV_SANS_MH") == "1":
    _lc = {k: v for k, v in (dom.get("land_cover") or {}).items()
           if not k.startswith(("wet", "frac"))}
    model.vertical_column.set_land_cover(_lc)
    print("[province] DIAGNOSTIC : milieux humides retires de l'occupation", flush=True)
if dom.get("melt_params"):
    model.vertical_column.set_melt_params(dom["melt_params"])
if dom.get("phenology"):
    model.vertical_column.set_phenology(dom["phenology"])
if dom.get("linacre") and os.environ.get("PROV_SANS_LINACRE", "0") != "1":
    model.vertical_column.set_linacre_params(*dom["linacre"])
    model.vertical_column.etp_channel = None
if dom.get("kgw") is not None:
    # Champ k_gw provincial (GP sur les recessions de 127 stations) : il FIXE le niveau
    # local mesure, le NeRF module autour. Meme montage que dans etl_run.
    _kv = dom["kgw"]
    _o_se = model.spatial_encoder.forward
    def _se_kgw(*a, _o=_o_se, _k=_kv, **kw):
        sp = _o(*a, **kw); sp.k_gw = _k
        return sp
    model.spatial_encoder.forward = _se_kgw

# DEPART SUR LE CHAMP HYDROTEL (recette gen1, ETL_INIT_HYDROTEL=sauf_ks). Le champ est
# ajuste par regression sur les valeurs calibrees par noeud puis laisse LIBRE : ce n'est
# pas un ancrage, c'est un point de depart. La courbe de retention (b, psis) est en
# revanche imposee par noeud, K_sat, porosites et epaisseurs restant appris.
_soil = None if os.environ.get("PROV_SANS_SOL", "0") == "1" else dom.get("soil")

# DRAINAGE SOUTERRAIN AGRICOLE (PROV_DRAIN=1, O12). Opt-in, defaut inactif : le
# processus est ecrit et teste, il n'est pas encore MESURE. Espacement et profondeur
# aux valeurs courantes du drainage quebecois (15 m, 1 m) ; la part de terres cultivees
# effectivement drainee reste le parametre le plus incertain, d'ou son exposition.
if os.environ.get("PROV_DRAIN", "0") == "1":
    model.vertical_column.drainage_agricole = dict(
        espacement_m=float(os.environ.get("PROV_DRAIN_L", "15")),
        profondeur_m=float(os.environ.get("PROV_DRAIN_Z", "1.0")),
        part_cultive=float(os.environ.get("PROV_DRAIN_F", "0.6")))
    print(f"[province] drainage agricole ACTIF : {model.vertical_column.drainage_agricole}",
          flush=True)

_lp = dict(cfg.get("literature_prior") or {})
_lp["K_sat_1"] = 0.04; _lp["K_c"] = 1.0; _lp["k_gw"] = 0.07; _lp.setdefault("krec", 5e-5)
model.spatial_encoder.init_from_literature(_lp)
if _soil:
    from meandre.data.hydrotel_calib import imposed_retention_curve
    _cib = {"K_sat_1": _soil["ks1"].float() * 24, "K_sat_2": _soil["ks2"].float() * 24,
            "K_sat_3": _soil["ks3"].float() * 24, "porosity_1": _soil["thetas1"].float(),
            "porosity_2": _soil["thetas2"].float(), "porosity_3": _soil["thetas3"].float(),
            "Z2": _soil["z2"].float(), "Z3": _soil["z3"].float()}
    model.spatial_encoder.fit_to_field(
        dom["node_coords"], dom["territorial"].data, _cib,
        n_iter=int(os.environ.get("PROV_INIT_ITER", "2000")))
    # krec LIBRE (ETL_KREC_LIBRE de gen1) : la courbe d'Hydrotel porte une recharge
    # quasi nulle, un robinet ferme en amont de notre aquifere.
    model.vertical_column.set_calibrated_soil(imposed_retention_curve(_soil, True))
    _pt0 = getattr(model.spatial_encoder, "_prior_targets", None) or {}
    for _k, _v in _cib.items():
        _pt0[_k] = float(_v.median())
    model.spatial_encoder._prior_targets = _pt0
    print(f"[province] depart sur le champ Hydrotel : K_sat_1 median "
          f"{_pt0['K_sat_1']:.4f} m/j, cibles du prior realignees", flush=True)
model.vertical_column.split_mode = "wet_bulb"
model.vertical_column.t_neige_seuil = float(os.environ.get("PROV_TWB", "-0.8"))
# La modulation saisonniere est une BEQUILLE du degre-jour : elle amplifie la fonte de
# fin mars pour compenser son absence de terme radiatif. Sous ETI, le rayonnement est
# explicite et la bequille ferait double emploi. Elle reste donc reservee au degre-jour.
if not ETI:
    model.vertical_column.melt_seasonal_amp = float(os.environ.get("PROV_AMP", "0.5"))
model.spatial_encoder.prior_on_krec = True
_t = getattr(model.spatial_encoder, "_prior_targets", None) or {}
_t["krec"] = 2e-5; _t["k_gw"] = 0.0273
model.spatial_encoder._prior_targets = _t
print(f"[province] recette 1.0 : bulbe humide {model.vertical_column.t_neige_seuil:+.2f}, "
      f"fonte {'ETI (radiation reelle)' if ETI else 'degre-jour, saison ' + str(getattr(model.vertical_column, 'melt_seasonal_amp', None))}, krec ancre 2e-5, "
      f"k_gw 0.0273 | {sum(p.numel() for p in model.parameters()):,} parametres", flush=True)
if "PROV_WARM" in os.environ:
    model.load(os.environ["PROV_WARM"])
    print(f"[province] warm-start depuis {os.environ['PROV_WARM']}", flush=True)

tcfg_obj = TrainingConfig(
    lr=float(os.environ.get("PROV_LR", tcfg.get("lr", 3e-4))),
    n_epochs=N_EPOCHS,
    chunk_steps=int(os.environ.get("PROV_CHUNK", tcfg.get("chunk_steps", 45))),
    # tbptt REGLE SUR LE CHUNK par defaut. Le defaut de 365 detachait le graphe tous
    # les quatre chunks de 90 : la memoire GPU montait de 4878 a 7932 Mo au fil de
    # l'epoch et le run provincial est mort en depassement apres 11 h 30. Sur un bassin
    # la meme relation existait sans mordre, douze fois moins de troncons tenant dans
    # la marge.
    tbptt_steps=int(os.environ.get(
        "PROV_TBPTT", os.environ.get("PROV_CHUNK", tcfg.get("chunk_steps", 45)))),
    grad_clip=float(tcfg.get("grad_clip", 1.0)),
    # w_prior : la flotte gen1 le portait (terme dominant de sa perte, ~4400 % du
    # total selon sa propre decomposition). On le REPRODUIT pour que la comparaison
    # porte sur l'architecture et non sur la recette ; son reequilibrage est une
    # question ouverte, tracee au registre (dette #19).
    # PROV_PRIOR : levier de reequilibrage, defaut = la valeur gen1 pour ne rien
    # changer en silence. MESURE du 2026-08-28, banc provincial : a 0.005 le terme
    # prior pese 5045 % du total de la perte de debit, tws_clim 3548 % et tws 1959 %.
    # L'optimiseur ramene donc le champ vers ses cibles de litterature et satisfait
    # GRACE, en SACRIFIANT l'hydrogramme -- toutes les metriques se degradent de facon
    # monotone des l'epoque 1 (val_kge 0.7149 -> 0.5226 en cinq epoques, perte
    # d'entrainement 4.57 -> 7.05). Le garde-fou de divergence ne voit rien : il
    # declenche sur un pic a 3x la moyenne mobile, pas sur une derive lente.
    w_prior=float(os.environ.get("PROV_PRIOR",
                                 tcfg.get("w_prior", 0.005) * _AUX)),
    # PROV_HUBER : borne les z-scores des contraintes auxiliaires. Defaut 3.0 ici,
    # contre 0 (desactive) dans le trainer pour ne rien changer aux runs anterieurs.
    # Sans elle, la province a vu sa perte d'entrainement passer de 4.4 a 48.9 en huit
    # epochs pendant que le debit se degradait : GRACE optimisait a la place du debit,
    # et la mediane provinciale tombait de 0.6193 a 0.4518 (2026-08-26).
    aux_huber_delta=float(os.environ.get("PROV_HUBER", "3.0")),
    # PROV_TWS_FORME=1 : GRACE en forme et non en niveau. Defaut INACTIF tant que le
    # diagnostic de stockage n'a pas montre que l'amplitude est bien le probleme --
    # activer un remede avant d'avoir le diagnostic, c'est ce qu'on reproche a un
    # calage.
    tws_shape_only=(os.environ.get("PROV_TWS_FORME", "0") == "1"),
    best_metric=tcfg.get("best_metric", "kge_median"),
    patience=int(tcfg.get("patience", 0)),
)
# Couper les prelevements AVANT le trainer, pas seulement a l'evaluation : la question
# est de savoir si le modele apprend a compenser une donnee fausse, ce qui ne se voit
# pas en inference sur des poids deja formes.
def _prelev_nul_pre(w):
    import copy as _cp
    z = _cp.copy(w)
    for _a in ("_vals", "_vals_gw"):
        if hasattr(z, _a):
            setattr(z, _a, getattr(z, _a) * 0.0)
    if hasattr(z, "net") and getattr(z, "net", None) is not None:
        z.net = z.net * 0.0
    return z

if _SANS_PRELEV:
    from dataclasses import replace as _rep
    _z = _prelev_nul_pre(dom["train_data"].withdrawals)
    dom["train_data"] = _rep(dom["train_data"], withdrawals=_z)
    dom["val_data"] = _rep(dom["val_data"], withdrawals=_z)
    print("[province] PRELEVEMENTS COUPES (diagnostic)", flush=True)
trainer = Trainer(model, dom["loss_fn"], train_data=dom["train_data"],
                  val_data=dom["val_data"], config=tcfg_obj, checkpoint_path=CKPT)
trainer.fit()

# ── tenue de cote 2022-2024, rapportee par plateforme ────────────────────────
print(f"\n[province] HELD-OUT 2022-2024", flush=True)
if os.path.exists(CKPT):
    model.load(CKPT)
model.eval()
td = dom["train_data"]
# COUPER LES PRELEVEMENTS N'EST PAS LES METTRE A None. `model.simulate` appelle
# `withdrawals.gw_withdrawal(t)` sans garde : passer None leve une AttributeError
# apres la simulation complete, donc apres avoir paye tout le calcul (mesure le
# 2026-08-29 sur le banc de bissection, qui est mort a la seconde passe). On neutralise
# donc les VALEURS en gardant l'objet et son interface.
def _prelev_nul(w):
    import copy as _cp
    z = _cp.copy(w)
    for _a in ("_vals", "_vals_gw"):
        if hasattr(z, _a):
            setattr(z, _a, getattr(z, _a) * 0.0)
    if hasattr(z, "net") and getattr(z, "net", None) is not None:
        z.net = z.net * 0.0
    return z

_PRELEV = _prelev_nul(td.withdrawals) if _SANS_PRELEV else td.withdrawals
with torch.no_grad():
    Q, _ = model.simulate(forcing=td.forcing[:],
                          initial_state=HydroState.zeros(dom["n_nodes"], device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords,
                          territorial=td.territorial, withdrawals=_PRELEV,
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
    # MARQUEUR MACHINE, distinct de la prose. Le chargeur du domaine imprime lui
    # aussi des lignes contenant 'mediane provinciale' quand un champ manque et
    # qu'il pose la valeur mediane de la province a la place. Un lecteur automatique
    # qui cherchait cette chaine rendait donc 0.6, une valeur de remplissage de
    # milieu humide, au lieu du score -- trouve le 2026-08-28, avant que la file de
    # fin de semaine ne prenne toutes ses decisions dessus. Une sortie destinee a
    # etre relue par un programme doit avoir un marqueur qui n'appartient qu'a elle.
    print(f"PROVMED {np.median(tous):.4f}", flush=True)
# ── DUMP PAR TRONCON, managed ET naturalise (PROV_DUMP=<prefixe>) ────────────
# Demande d'Essi (2026-08-28) : rapport provincial statique et couches feuillage --
# hydrogrammes jauges et non jauges, resumes de KGE, cartes d'impact RELATIF des
# prelevements et rejets, cartes des parametres du champ.
#
# POURQUOI LE PILOTE ET PAS UN SCRIPT ANNEXE. Dette #6 du registre : un point de reprise
# ne definit pas un modele. Occupation du sol, milieux humides, phenologie, ancrages de
# fonte et seuil pluie-neige sont poses A L'EXECUTION. Les diagnostics annexes qui
# reconstruisaient la recette a la main ont deja mesure un AUTRE modele que le champion
# -- trois fois rien qu'aujourd'hui sur l'etape 0 des barrages. Le cache est donc produit
# ici, ou le runtime est celui du run, et le rapport comme la carte le lisent tel quel.
#
# DEUX PASSES. La passe naturalisee (prelevements a zero, meme etat initial) est ce qui
# rend l'impact RELATIF calculable : sans elle on ne peut afficher qu'un volume preleve,
# pas sa part du debit. C'est la difference entre une carte de pression et une carte
# d'impact.
_DUMP = os.environ.get("PROV_DUMP")
if _DUMP:
    import pandas as _pdm
    _t = _pdm.DatetimeIndex(dom["times"])
    _mois = _t.month.to_numpy()

    def _ecrire(chemin, Qd, wnet):
        _q = Qd.cpu().numpy()
        _qm = np.stack([_q[_mois == m].mean(axis=0) for m in range(1, 13)])
        with torch.no_grad():
            _sp = model.spatial_encoder(td.node_coords, dom["territorial"].data)
        _champs = {f"param_{k}": getattr(_sp, k).detach().cpu().numpy()
                   for k in _sp.__dataclass_fields__
                   if torch.is_tensor(getattr(_sp, k))
                   and getattr(_sp, k).shape[:1] == (dom["n_nodes"],)}
        np.savez_compressed(chemin,
                            q_mensuel=_qm.astype(np.float32),
                            q_annuel=_q.mean(axis=0).astype(np.float32),
                            coords=td.node_coords.cpu().numpy(),
                            prelev_net_abs=wnet.astype(np.float32),
                            **{k: v.astype(np.float32) for k, v in _champs.items()})
        print(f"[dump] {chemin} : {len(_champs)} champs, {Qd.shape[0]} pas", flush=True)

    _w = td.withdrawals
    _wabs = (_w.net.abs().sum(dim=0).cpu().numpy()
             if hasattr(_w, "net") and _w.net is not None
             else np.zeros(dom["n_nodes"], dtype=np.float32))
    _ecrire(f"{_DUMP}-avec.npz", Q, _wabs)

    with torch.no_grad():
        Qn, _ = model.simulate(forcing=td.forcing[:],
                               initial_state=HydroState.zeros(dom["n_nodes"], device=DEVICE),
                               graph=td.graph, node_coords=td.node_coords,
                               territorial=dom["territorial"], withdrawals=None,
                               day_of_year=td.day_of_year)
    _ecrire(f"{_DUMP}-sans.npz", Qn, np.zeros(dom["n_nodes"], dtype=np.float32))

    # Les series par station, pour les hydrogrammes du rapport. On garde les JAUGES
    # (observe et simule) et un echantillon de troncons NON JAUGES, puisque le point du
    # modele distribue est justement de rendre un hydrogramme la ou il n'y a pas de jauge.
    _idx = td.station_idx.cpu().numpy()
    _nonj = np.setdiff1d(np.arange(dom["n_nodes"]), _idx)
    _gros = _nonj[np.argsort(-Q.mean(dim=0).cpu().numpy()[_nonj])][:200]
    np.savez_compressed(f"{_DUMP}-q.npz",
                        temps=_t.values.astype("datetime64[D]"),
                        station_ids=np.array(dom["station_ids"], dtype=object),
                        station_node=_idx.astype(np.int32),
                        q_stations=Q[:, _idx].cpu().numpy().astype(np.float32),
                        q_obs=dom["val_data"].q_obs.cpu().numpy().astype(np.float32),
                        nonjauge_node=_gros.astype(np.int32),
                        q_nonjauge=Q[:, _gros].cpu().numpy().astype(np.float32),
                        allow_pickle=True)
    print(f"[dump] series : {len(_idx)} jauges, {len(_gros)} troncons non jauges", flush=True)

os._exit(0)
