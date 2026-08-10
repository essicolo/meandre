"""BANC DE MODULES (demande d'Essi) : des CENTAINES de tests unitaires en SECONDES,
sans aucune simulation régionale et SANS dépendre d'Hydrotel.

Principe : on n'a pas besoin des sorties du binaire pour comparer deux SCHÉMAS entre
eux. On injecte une impulsion unitaire dans la géométrie RÉELLE des tronçons
(troncon.trl) et on mesure la fonction de réponse de chaque schéma : retard du pic,
atténuation, conservation de la masse, temps de vidange. Là où les réponses divergent,
la divergence est chiffrée tronçon par tronçon, en secondes.

Trois familles :
  A. RIVIÈRES : Muskingum de méandre (K bornes NeRF 4-48 h) contre le clone de l'onde
     cinématique modifiée d'Hydrotel, même géométrie, même impulsion.
  B. LACS : loi de tarage Q = c·h^k du trl (clone) contre le réservoir de méandre.
  C. CONSERVATION : chaque schéma restitue-t-il exactement le volume injecté ?

  PYTHONIOENCODING=utf-8 python .runs/quebec/banc_modules.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np, torch

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").upper()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG}_LN24HA_2020"
N_ECH = int(os.environ.get("BANC_N", "400"))
T = 60           # jours simulés par test
torch.set_grad_enabled(False)

from hydrotel_clone.network_routing_torch import _transfert_riviere_vec, _transfert_lac_vec
from meandre.routing.kinematic import MuskingumCunge

# ── géométrie réelle des tronçons ────────────────────────────────────────────
lignes = [l.strip() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines() if l.strip()]
riv, lac = [], []
for l in lignes[3:]:
    t = l.split()
    if int(t[1]) == 1:
        riv.append((max(float(t[4]), 1.0), max(float(t[5]), 0.1), max(float(t[6]), 0.0025)))
    else:
        ptr = 4 + int(t[3])
        s_ = float(t[ptr+1]) * 1e6
        if s_ > 0: lac.append((s_, float(t[ptr+2]), float(t[ptr+3])))
rng = np.random.default_rng(0)
riv = np.array(riv)[rng.choice(len(riv), min(N_ECH, len(riv)), replace=False)]
lac = np.array(lac)[rng.choice(len(lac), min(N_ECH, len(lac)), replace=False)]
print(f"[banc] {len(riv)} rivières et {len(lac)} lacs échantillonnés dans {REG} "
      f"(longueur méd {np.median(riv[:,0])/1000:.1f} km, pente méd {np.median(riv[:,2]):.4f})", flush=True)

def diagnostics(Q, vol_in):
    """Q : (T, n). Retourne (retard pic j, pic, masse rendue, demi-vidange j) par colonne."""
    Q = np.asarray(Q, float)
    ip = Q.argmax(axis=0)
    pic = Q.max(axis=0)
    masse = Q.sum(axis=0) * 86400.0 / vol_in
    demi = np.full(Q.shape[1], np.nan)
    for j in range(Q.shape[1]):
        ap = np.flatnonzero(Q[ip[j]:, j] < 0.5 * pic[j])
        if len(ap): demi[j] = ap[0]
    return ip.astype(float), pic, masse, demi

IMPULSE = np.zeros(T); IMPULSE[5] = 10.0     # 10 m3/s pendant 1 jour
VOL_IN = 10.0 * 86400.0
nR = len(riv); nL = len(lac)
lng = torch.tensor(riv[:, 0], dtype=torch.float32)
lrg = torch.tensor(riv[:, 1], dtype=torch.float32)
pte = torch.tensor(riv[:, 2], dtype=torch.float32)
man = torch.full((nR,), 0.04)
z = torch.zeros(nR)

def rep_muskingum(K_h, x=0.20, nsub=2):
    r = MuskingumCunge(dt=86400.0, n_substeps=nsub)
    K = torch.full((nR,), K_h * 3600.0); X = torch.full((nR,), x)
    Qp = torch.zeros(nR); out = np.zeros((T, nR))
    for t in range(T):
        Qp = r(Q_in=z, Q_out_prev=Qp, q_lateral=torch.full((nR,), float(IMPULSE[t])), K=K, x=X)
        out[t] = Qp.numpy()
    return out

def rep_onde(nt=4):
    qb = torch.zeros(nR); ql = torch.zeros(nR); out = np.zeros((T, nR))
    dt = 86400.0 / nt
    for t in range(T):
        ap = torch.full((nR,), float(IMPULSE[t]))
        for _ in range(nt):
            qb = _transfert_riviere_vec(dt, lng, lrg, pte, man, z, ql, qb, z, ap)
            ql = ap
        out[t] = qb.numpy()
    return out

def rep_lac(nt=4):
    ai = torch.tensor(lac[:, 0], dtype=torch.float32)
    cc = torch.tensor(lac[:, 1], dtype=torch.float32)
    kk = torch.tensor(lac[:, 2], dtype=torch.float32)
    zl = torch.zeros(nL)
    qb = torch.zeros(nL); ql = torch.zeros(nL); out = np.zeros((T, nL))
    dt = 86400.0 / nt
    for t in range(T):
        ap = torch.full((nL,), float(IMPULSE[t]))
        for _ in range(nt):
            qb = _transfert_lac_vec(dt, ai, cc, kk, zl, ql, qb, zl, ap)
            ql = ap
        out[t] = qb.numpy()
    return out

KS = (4.0, 24.0, 48.0)
MK = {k: diagnostics(rep_muskingum(k), VOL_IN) for k in KS}
MO = diagnostics(rep_onde(), VOL_IN)
ML = diagnostics(rep_lac(), VOL_IN)

def ligne(nom, d):
    print(f"  {nom:32s} {np.nanmedian(d[0]):15.1f} {np.nanmedian(d[1]):12.2f} "
          f"{np.nanmedian(d[2]):13.3f} {np.nanmedian(d[3]):17.1f}")

print(f"\n=== A. RIVIÈRES : réponse à une impulsion (n={nR}, géométrie réelle) ===")
print(f"  {'schéma':32s} {'retard pic (j)':>15s} {'pic (m³/s)':>12s} {'masse rendue':>13s} {'demi-vidange (j)':>17s}")
for k in KS: ligne(f"Muskingum méandre K={k:4.0f} h", MK[k])
ligne("clone onde cinématique Hydrotel", MO)

print(f"\n=== B. LACS : réponse à une impulsion (n={nL}, loi c·h^k réelle du trl) ===")
ligne("clone transfert_lac", ML)
kq = lac[:, 1] / lac[:, 0]
print(f"  k_lake équivalent méandre = c/A : méd {np.median(kq):.2e} /s | "
      f"{(kq < 1e-6).mean()*100:.0f} % des lacs sous le plancher 1e-6 du modèle")

print(f"\n=== C. CONSERVATION de la masse sur {T} j (1.0 = parfait) ===")
for k in KS:
    d = MK[k][2]
    print(f"  Muskingum K={k:4.0f} h : méd {np.nanmedian(d):.4f} | min {np.nanmin(d):.4f} | "
          f"perdant >5 % : {(d < 0.95).mean()*100:.1f} %")
for nom, d in [("onde cinématique ", MO[2]), ("lacs (c·h^k)     ", ML[2])]:
    print(f"  {nom} : méd {np.nanmedian(d):.4f} | min {np.nanmin(d):.4f} | "
          f"perdant >5 % : {(d < 0.95).mean()*100:.1f} %")
