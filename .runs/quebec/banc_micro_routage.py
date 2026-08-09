"""BANC MICRO-ROUTAGE (demande d'Essi) : même pile des deux côtés, entrées IDENTIQUES.

L'échelle manquante entre la colonne validée par UHRH et les régions entières : le
ROUTAGE seul, tronçon par tronçon, alimenté par l'apport latéral d'HYDROTEL LUI-MÊME
(réexécution instrumentée) — donc ni météo ni colonne dans l'équation.

- Tronçons de TÊTE (aucun amont) : aval = f(apport seul). On route l'apport d'Hydrotel
  par (a) le Muskingum de méandre (K init littérature 24 h, x 0.2) et (b) le clone de
  l'onde cinématique (géométrie troncon.trl), et on compare chacun au débit aval
  qu'Hydrotel a réellement produit.
- Tronçons-LACS : amont reconstruit exactement (somme des avals amont, tous écrits),
  routé par le clone transfert_lac (c, k, surface du trl) et par le LakeModule de
  méandre. Même comparaison.

  PYTHONIOENCODING=utf-8 python .runs/quebec/banc_micro_routage.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import numpy as np, pandas as pd, torch, xarray as xr

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"D:/meandre-data/quebec/{REG.upper()}_fidelite"
RES = Path(PROJ) / "simulation/simulation/resultat"
T0, T1 = "2022-01-01", "2024-12-31"
N_ECH = int(os.environ.get("MICRO_N", "300"))

print("[resultat] fichiers :", sorted(p.name for p in RES.glob("*.nc")), flush=True)

def ouvrir(motifs):
    for p in sorted(RES.glob("*.nc")):
        if any(m in p.name.lower() for m in motifs):
            d = xr.open_dataset(p)
            var = [v for v in d.data_vars if d[v].ndim == 2][0]
            return d, var
    raise SystemExit(f"fichier absent pour {motifs}")

d_av, v_av = ouvrir(["debit_aval"])
d_ap, v_ap = ouvrir(["apport"])
t_av = pd.to_datetime(d_av["time"].values)
m = (t_av >= T0) & (t_av <= T1)
ids = d_av["idtroncon"].values if "idtroncon" in d_av else d_av[list(d_av.coords)[0]].values
QA = d_av[v_av].values[m] if d_av[v_av].dims[0] == "time" else d_av[v_av].values[:, m].T
AP = d_ap[v_ap].values[m] if d_ap[v_ap].dims[0] == "time" else d_ap[v_ap].values[:, m].T
print(f"[hydrotel] aval {QA.shape} | apport {AP.shape}", flush=True)

# geometrie et topologie du trl
lignes = [l.strip() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines() if l.strip()]
geo = {}
for l in lignes[3:]:
    t = l.split(); tid = int(t[0]); typ = int(t[1])
    if typ == 1:
        geo[tid] = dict(riv=True, lng=max(float(t[4]), 1.0), lrg=max(float(t[5]), 0.1),
                        pte=max(float(t[6]), 0.0025), aval=int(t[-1]))
    else:
        ptr = 4 + int(t[3])
        geo[tid] = dict(riv=False, surf=float(t[ptr+1]) * 1e6, c=float(t[ptr+2]),
                        k=float(t[ptr+3]), aval=int(t[-1]))
pos = {int(i): j for j, i in enumerate(ids)}
amont = {tid: [] for tid in geo}
for tid, g in geo.items():
    if g["aval"] in amont:
        amont[g["aval"]].append(tid)
tetes_riv = [tid for tid, g in geo.items() if g["riv"] and not amont[tid] and tid in pos]
lacs = [tid for tid, g in geo.items() if not g["riv"] and tid in pos]
rng = np.random.default_rng(0)
tetes_ech = list(rng.choice(tetes_riv, size=min(N_ECH, len(tetes_riv)), replace=False))
lacs_ech = list(rng.choice(lacs, size=min(N_ECH, len(lacs)), replace=False))
print(f"[reseau] {len(tetes_riv)} têtes rivière (échantillon {len(tetes_ech)}) | "
      f"{len(lacs)} lacs (échantillon {len(lacs_ech)})", flush=True)

from hydrotel_clone.network_routing_torch import _transfert_riviere_vec, _transfert_lac_vec
from meandre.routing.kinematic import MuskingumCunge

T = QA.shape[0]
musk = MuskingumCunge(n_substeps=2)

def route_muskingum(apport, K_h=24.0, x=0.2):
    K = torch.tensor([K_h * 3600.0]); X = torch.tensor([x])
    Qp = torch.zeros(1); out = np.zeros(T)
    ap = torch.tensor(apport, dtype=torch.float32)
    for t in range(T):
        Qin = torch.zeros(1)
        Q = musk(Qin, Qp, K, X, q_lateral=ap[t:t+1])
        out[t] = float(Q); Qp = Q
    return out

def route_onde(apport, g):
    lng = torch.tensor([g["lng"]]); lrg = torch.tensor([g["lrg"]])
    pte = torch.tensor([g["pte"]]); man = torch.tensor([0.04])
    qa = torch.zeros(1); qb = torch.zeros(1); ql = torch.zeros(1)
    out = np.zeros(T)
    ap = torch.tensor(apport, dtype=torch.float32)
    for t in range(T):
        nt = 4; dt = 86400.0 / nt; qd = qb
        for _ in range(nt):
            qd = _transfert_riviere_vec(dt, lng, lrg, pte, man, qa, ql, qb, torch.zeros(1), ap[t:t+1])
            qa = torch.zeros(1); qb = qd; ql = ap[t:t+1]
        out[t] = float(qd)
    return out

def route_lac_clone(qin, apport, g):
    aire = torch.tensor([g["surf"]]); c = torch.tensor([g["c"]]); k = torch.tensor([g["k"]])
    qa = torch.zeros(1); qb = torch.zeros(1); ql = torch.zeros(1)
    out = np.zeros(T)
    qi = torch.tensor(qin, dtype=torch.float32); ap = torch.tensor(apport, dtype=torch.float32)
    for t in range(T):
        nt = 4; dt = 86400.0 / nt; qd = qb
        for _ in range(nt):
            qd = _transfert_lac_vec(dt, aire, c, k, qa, ql, qb, qi[t:t+1], ap[t:t+1])
            qa = qi[t:t+1]; qb = qd; ql = ap[t:t+1]
        out[t] = float(qd)
    return out

def score(sim, ref):
    v = np.isfinite(sim) & np.isfinite(ref)
    if v.sum() < 300 or ref[v].std() < 1e-9: return np.nan, np.nan
    return float(np.corrcoef(sim[v], ref[v])[0, 1]), float(sim[v].mean() / max(ref[v].mean(), 1e-9))

res_m, res_o = [], []
for tid in tetes_ech:
    j = pos[tid]; g = geo[tid]
    ap = AP[:, j]; ref = QA[:, j]
    r1, b1 = score(route_muskingum(ap), ref); res_m.append((r1, b1))
    r2, b2 = score(route_onde(ap, g), ref); res_o.append((r2, b2))
rm = np.array(res_m); ro = np.array(res_o)
print(f"\n=== TÊTES RIVIÈRE ({len(tetes_ech)}), entrée = apport d'Hydrotel ===")
print(f"  Muskingum méandre (K=24 h, x=0.2) : r méd {np.nanmedian(rm[:,0]):.3f} | beta méd {np.nanmedian(rm[:,1]):.3f}")
print(f"  clone onde cinématique             : r méd {np.nanmedian(ro[:,0]):.3f} | beta méd {np.nanmedian(ro[:,1]):.3f}")

res_l = []
for tid in lacs_ech:
    j = pos[tid]; g = geo[tid]
    qin = np.zeros(T)
    for u in amont[tid]:
        if u in pos: qin += QA[:, pos[u]]
    ap = AP[:, j]; ref = QA[:, j]
    r3, b3 = score(route_lac_clone(qin, ap, g), ref); res_l.append((r3, b3))
rl = np.array(res_l)
print(f"\n=== LACS ({len(lacs_ech)}), entrée = amont reconstruit + apport d'Hydrotel ===")
print(f"  clone transfert_lac (c,k,surface trl) : r méd {np.nanmedian(rl[:,0]):.3f} | beta méd {np.nanmedian(rl[:,1]):.3f}")
