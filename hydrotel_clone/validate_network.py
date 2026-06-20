"""Validation BOUT-EN-BOUT du routeur réseau (accumulation topologique + rivières
+ lacs) contre Hydrotel C++ sur DELISLE. Pilote avec apport_lateral (sortie
versant d'Hydrotel, déjà validée) et compare debit_aval à l'EXUTOIRE (tronçon 1).
Valide l'assemblage des solveurs déjà prouvés + la topologie + les sous-pas.

  python hydrotel_clone/validate_network.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hydrotel_clone.network_routing import route_network, delisle_network

DEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
W = DEL + "/simulation/simulation/resultat"


def rcols(name, h=2):
    L = open(f"{W}/{name}.csv", encoding="latin-1").read().splitlines()
    head = L[1].split(";")[1:]                       # ids de troncon
    ids = [int(x) for x in head]
    arr = np.array([[float(x) for x in ln.split(";")[1:]] for ln in L[h:] if len(ln.split(";")) > 1])
    return arr, ids


reaches, downstream, topo, idx_of_id = delisle_network(DEL)
print(f"réseau : {len(reaches)} tronçons, {sum(1 for r in reaches if r['type']=='lake')} lacs, "
      f"topo couvre {len(topo)}/{len(reaches)}")

aplat, ids_a = rcols("apport_lateral")
aval, ids_v = rcols("debit_aval")
T = min(len(aplat), len(aval))

# réordonne apport_lateral (colonnes = id troncon) vers l'ordre des reaches (index)
nR = len(reaches)
apl_reach = np.zeros((T, nR))
for col, tid in enumerate(ids_a):
    if tid in idx_of_id:
        apl_reach[:, idx_of_id[tid]] = aplat[:T, col]

out = route_network(reaches, downstream, topo, apl_reach)

# exutoire = tronçon id 1
oidx = idx_of_id[1]
col1 = ids_v.index(1)
sim = out[:, oidx]; ref = aval[:T, col1]
rmse = float(np.sqrt(np.nanmean((sim - ref) ** 2)))
print(f"\nEXUTOIRE (tronçon 1) : RMSE {rmse:.4f} m3/s  max abs {np.max(np.abs(sim-ref)):.4f}")
print(f"somme clone {sim.sum():.0f} vs H {ref.sum():.0f} | pic clone {sim.max():.2f} vs H {ref.max():.2f}")
denom = np.nansum((ref - ref.mean())**2)
nse = 1 - np.nansum((sim-ref)**2)/denom if denom > 0 else float('nan')
print(f"NSE clone vs Hydrotel : {nse:.4f}")
print("\njour  aval_clone  aval_H")
for t in range(100, 120):
    print(f"{t:4d}  {sim[t]:9.2f}  {ref[t]:9.2f}")
print("DONE")
