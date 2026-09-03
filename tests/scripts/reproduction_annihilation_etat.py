"""REPRODUCTION du defaut d'aneantissement de l'etat aux frontieres de bloc (R64).

Ce fichier etait pose sous tests/test_chunk_state_annihilation.py. Il est ECRIT COMME UN
SCRIPT (code et assertions au niveau du module) et pose torch.set_default_dtype(float64)
globalement : pytest le collectait, l'executait a l'import, et la double precision fuyait
vers seize tests de routage et d'encodeur temporel qui echouaient alors. Il est deplace
sous tests/scripts/, qui n'est pas collecte, et reste lancable a la main.

La NON-REGRESSION est couverte par tests/test_vertical/test_poursuite_etat.py, qui
verifie les deux chemins : sans poursuite l'etat est refabrique, avec poursuite le
manteau et le profil de gel survivent.

  python tests/scripts/reproduction_annihilation_etat.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params
from meandre.spatial.territorial import TerritorialFeatures
from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams
from hydrotel_clone.frost import n_intervalles
from hydrotel_clone.snow import DegreJourModifie

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

N = 3
occ = dict(feuillus=102 / 1754, ouverts=118 / 1754, humides=24 / 1754,
           urbain=1111 / 1754, routes=280 / 1754, eau=119 / 1754)
psnow, psoil, petr = build_static_params(
    N, lat=45.3, slope=0.026, orientation=7, texture="sandy_loam",
    z=(0.1, 0.4, 1.0), occupation=occ)

col = HydrotelColumn(et_mode="mcguinness", use_frost=True)
col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))

T = lambda x: torch.full((N,), float(x))

# ---- Phase 1 : 45 jours d'hiver continu (un chunk), état riche au bout ----
st = col.init_state(N, theta_init=(0.36, 0.36, 0.36))
for i in range(45):
    P = T(4.0)                     # neige tous les jours
    tn = T(-12.0); tx = T(-4.0)     # froid franc
    prod, st, diag = col(P, tn, tx, T(8.0), T(1.5), T(0.8), float((i % 365) + 1), st)

def _swe_total(snow):
    """SWE total (mm) : somme des stocks des 3 classes (stock en MÈTRES, unité Hydrotel)."""
    tot = 0.0
    for c in DegreJourModifie.CLASSES:
        tot += float(snow[c][0].mean()) * 1000.0
    return tot

swe_fin = _swe_total(st.snow)
gel_fin = float(st.frost_profile.abs().mean())
theta_fin = [float(st.theta1.mean()), float(st.theta2.mean()), float(st.theta3.mean())]
print(f"[chunk 1] SWE accumulé : {swe_fin:.2f} mm ; |profil gel| : {gel_fin:.3f} ; theta : {theta_fin}")
assert swe_fin > 50.0, "45 jours de neige froide doivent construire un manteau"

# ---- Phase 2 : le trainer passe `st` comme initial_state au chunk 2. ----
# model.simulate() fait : setup_simulate(sp, terr, coords, state) qui recrée _aux.
# On reproduit EXACTEMENT ce chemin avec un SpatialParams synthétique.
net = SpatialFieldNetwork(n_territorial=17)
coords = torch.tensor([[-74.0, 45.3 + 0.01 * i] for i in range(N)], dtype=torch.float64)
terr = TerritorialFeatures.zeros(n_nodes=N, n_features=17)
terr.physical["area_km2_physical"] = torch.ones(N) * 10.0
terr.physical["slope_fraction"] = torch.ones(N) * 0.026
sp = net(coords, terr.to_tensor())

col.setup_simulate(sp, terr, coords, st)   # <- appelé par simulate() à chaque chunk

swe_apres = _swe_total(col._aux.snow)
gel_apres = float(col._aux.frost_profile.abs().mean())
print(f"[chunk 2 setup_simulate] SWE : {swe_fin:.2f} -> {swe_apres:.2f} mm ; |gel| : {gel_fin:.3f} -> {gel_apres:.3f}")

# ---- Verdicts ----
if swe_apres == 0.0:
    print("DÉFAUT CONFIRMÉ : le manteau neigeux est EFFACÉ à la frontière de chunk.")
    print("  -> L'entraînement (chunk_steps=45) n'a jamais simulé d'hiver continu.")
else:
    print("SWE survit à setup_simulate.")
    sys.exit(0)

if gel_apres < 1e-9 and gel_fin > 0.1:
    print("DÉFAUT CONFIRMÉ : le profil de gel est RÉINITIALISÉ à la frontière de chunk.")

# La theta : model.py:374-381 la remplace par 0.9×porosité AVANT setup_simulate.
poros = float(sp.porosity_1.mean())
print(f"[model.py] theta entrante {theta_fin[0]:.3f} serait écrasée par 0.9×porosité = {0.9 * poros:.3f}")