"""Le champ thermique apprend-il de la PHYSIQUE, ou compense-t-il autre chose ?

Controle annonce avant de lire le moindre score (2026-08-27). Trois degres de liberte
ajoutes au gel peuvent ameliorer un ajustement pour deux raisons opposees : parce qu'ils
representent la variabilite thermique reelle des sols, ou parce qu'ils servent de levier
libre pour absorber une erreur qui n'a rien a voir. La difference se voit dans les
VALEURS, pas dans la perte.

Trois questions, dans l'ordre :
  1. les valeurs restent-elles dans les plages de la litterature, ou filent-elles aux
     bornes (signe d'un parametre qui compense) ;
  2. varient-elles dans l'espace, ou le champ est-il plat (auquel cas les trois sorties
     ne servent qu'a decaler une constante) ;
  3. correlent-elles avec ce qui les gouverne physiquement -- la texture pour la
     conductivite et la capacite, le couvert pour l'amortissement nival.

    python .runs/quebec/audit_gel.py <checkpoint.pt> [plateformes...]
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

CKPT = sys.argv[1]
PLATEFORMES = [a.lower() for a in sys.argv[2:]] or ["gasp", "mont"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
dom = load_domain(PLATEFORMES, dict(cfg["loss"]), device=DEV)

m = HydroModel(n_nodes=dom["n_nodes"], n_territorial=dom["territorial"].n_features,
               n_forcing=6, use_temporal=False, use_residual=False,
               use_travel_time_attn=False, param_mode="nerf", column_mode="hydrotel",
               et_mode="linacre", use_latent_codes=False, spatial_melt=True,
               routing_mode=cfg["model"].get("routing_mode", "operator-lagged"),
               predict_lake_params=True, compile_soil=False, use_aquifer=True).to(DEV)
# CKPT="init" : le champ TEL QU'INITIALISE, sans aucun entrainement. Controle
# indispensable avant de lire les correlations : init_from_literature fixe le BIAIS des
# sorties, mais si les POIDS reliant les attributs a ces sorties ne sont pas nuls, le
# champ produit deja une variation spatiale correlee aux attributs par des poids
# aleatoires. On mesurerait alors du bruit d'initialisation en croyant lire un
# apprentissage -- d'autant plus probable que la dispersion apres deux epochs n'est que
# de trois pour cent.
# Le point de depart REEL du run entraine est init_from_literature, pas l'initialisation
# brute du reseau : sans cet appel, le controle comparait a un autre modele et sortait au
# milieu des bornes (1.35 / 1.73e6 / 3.23) au lieu des valeurs du C++ (2026-08-27).
_lp = dict(cfg.get("literature_prior") or {})
_lp["K_sat_1"] = 0.04; _lp["K_c"] = 1.0; _lp["k_gw"] = 0.07; _lp.setdefault("krec", 5e-5)
m.spatial_encoder.init_from_literature(_lp)
if CKPT != "init":
    m.load(CKPT)
else:
    print("[audit] champ NON ENTRAINE (init_from_literature seule)")
with torch.no_grad():
    sp = m.spatial_encoder(dom["node_coords"], dom["territorial"].to_tensor())

DEFAUT = {"diff_gel": 1.6e-7, "fs_neige": 2.35}
BORNES = {"diff_gel": (4e-8, 8e-7), "fs_neige": (0.5, 6.0)}
print(f"\n{'propriete':10s} {'defaut':>9s} {'q10':>9s} {'mediane':>9s} {'q90':>9s} "
      f"{'CV':>6s}  {'aux bornes':>10s}")
for k, (lo, hi) in BORNES.items():
    v = getattr(sp, k).detach().cpu().numpy()
    marge = 0.02 * (hi - lo)
    colle = float(((v < lo + marge) | (v > hi - marge)).mean()) * 100
    print(f"{k:10s} {DEFAUT[k]:9.3g} {np.quantile(v, .1):9.3g} {np.median(v):9.3g} "
          f"{np.quantile(v, .9):9.3g} {v.std() / abs(v.mean()):6.3f} {colle:9.1f} %")

# correlation avec ce qui les gouverne physiquement
cols = dom["territorial"].columns
X = dom["territorial"].data.detach().cpu().numpy()
print("\ncorrelation avec les attributs qui les gouvernent")
for k, attrs in (("diff_gel", ("f_sand", "f_clay", "f_wetland")),
                 ("fs_neige", ("f_forest", "f_agriculture", "mean_elevation_m"))):
    v = getattr(sp, k).detach().cpu().numpy()
    bouts = []
    for a in attrs:
        if a in cols:
            r = float(np.corrcoef(v, X[:, cols.index(a)])[0, 1])
            bouts.append(f"{a} {r:+.2f}")
    print(f"  {k:10s} " + " | ".join(bouts))
# Le controle thermodynamique conductivite/capacite n'a plus d'objet : il n'y a plus
# qu'une sortie. C'est lui qui a designe la redondance (correlation -0.920), et sa
# disparition est le signe que le defaut est ferme a la source plutot que surveille.

print("""
LECTURE. Un champ plat (CV proche de zero) veut dire que les trois sorties n'ont servi
qu'a deplacer une constante : le gain viendrait d'un decalage global, pas d'une physique
spatiale. Une forte proportion collee aux bornes veut dire que le champ pousse un levier
jusqu'a sa limite pour compenser autre chose. Et des correlations nulles avec la texture
diraient que la conductivite apprise n'a rien a voir avec le sol qu'elle est censee
decrire -- le cas ou le degre de liberte est utile au score et faux physiquement.""")
