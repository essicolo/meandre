"""Validation du clone routage canal ONDE CINÉMATIQUE MODIFIÉE contre Hydrotel
C++ sur DELISLE, tronçon de TÊTE (qamont≈0, pas de topologie → isole le solveur
TransfertRiviere + sous-pas + état). Pilote avec apport_lateral, compare debit_aval.

  python hydrotel_clone/validate_routing.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hydrotel_clone.routing import route_reach_day

DEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
W = DEL + "/simulation/simulation/resultat"


def read_cols(name, nhead=2):
    L = open(f"{W}/{name}.csv", encoding="latin-1").read().splitlines()
    rows = [ln.split(";") for ln in L[nhead:] if len(ln.split(";")) > 1]
    arr = np.array([[float(x) for x in r[1:]] for r in rows])
    return arr   # (T, n_troncon)


amont = read_cols("debit_amont"); aval = read_cols("debit_aval"); aplat = read_cols("apport_lateral")
T = min(len(amont), len(aval), len(aplat))
nT = amont.shape[1]

# géométrie troncon.trl : [id, ?, n_amont, n_aval, longueur, largeur, pente, ...]
geo = {}
trl = open(f"{DEL}/physitel/troncon.trl", encoding="latin-1").read().splitlines()
for ln in trl:
    p = ln.split()
    if len(p) >= 7 and p[0].isdigit():
        tid = int(p[0])
        try:
            geo[tid] = (float(p[4]), float(p[5]), float(p[6]))   # lng, lrg, pte
        except ValueError:
            pass

# tronçon de tête : amont ~0 partout, apport_lateral significatif, géométrie connue
cand = []
for j in range(nT):
    tid = j + 1
    if tid in geo and amont[:T, j].max() < 1e-4 and aplat[:T, j].sum() > 1.0:
        cand.append((tid, aplat[:T, j].sum()))
cand.sort(key=lambda x: -x[1])
assert cand, "aucun tronçon de tête trouvé"
tid = cand[0][0]; j = tid - 1
lng, lrg, pte = geo[tid]
print(f"Tronçon de tête {tid} : lng={lng} lrg={lrg} pte={pte}  (amont max {amont[:T,j].max():.2e})")

state = {"qamont": 0.0, "qaval": 0.0, "qapportlat": 0.0}
sim = np.zeros(T)
for t in range(T):
    q, state = route_reach_day(float(aplat[t, j]), lng, lrg, pte, 0.04, state)
    sim[t] = q

ref = aval[:T, j]
rmse = float(np.sqrt(np.nanmean((sim - ref) ** 2)))
print(f"RMSE debit_aval = {rmse:.5f} m3/s   max abs = {np.max(np.abs(sim-ref)):.5f}")
print(f"somme clone {sim.sum():.2f} vs H {ref.sum():.2f} ; pic clone {sim.max():.3f} vs H {ref.max():.3f}")
print("jour  apport_lat  aval_clone  aval_H")
for t in range(100, 120):
    print(f"{t:4d}  {aplat[t,j]:9.3f}  {sim[t]:9.3f}  {ref[t]:9.3f}")
print("DONE")
