"""MODIS ET et débits observés sont-ils RÉCONCILIABLES ? (question d'Essi)
Test SANS MODÈLE : sur plusieurs années, le bilan d'un bassin impose P − Q = ETR (la
variation de stock s'annule). Si MODIS dépasse P − Q, aucun modèle ne peut satisfaire
les deux, et le multi-objectif force un compromis impossible.

Pour chaque jauge : P moyen sur son bassin versant amont (forçage du run), débit observé
converti en lame d'eau (mm/an) via l'aire cumulée, et ETR MODIS moyen sur le même bassin.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/bilan_modis_vs_debit.py gasp
"""
import os, sys, collections
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch, duckdb
from joint_data import load_region

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

REG = (sys.argv[1] if len(sys.argv) > 1 else "gasp").lower()
DEVICE = "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

# aires cumulées amont
A = r["territorial"].get_physical("area_km2_local").numpy()
con = duckdb.connect(f"{_DATA_ROOT}/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
enf = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enf[int(s_)].append(int(d_))
topo = td.graph.topo_order.numpy()
Acum = A.copy()
for u in topo:
    for v in enf.get(int(u), []):
        Acum[v] += Acum[u]
# ensemble des noeuds amont de chaque jauge (remontee)
par = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    par[int(d_)].append(int(s_))
def amont(j):
    vus = set(); pile = [int(j)]
    while pile:
        u = pile.pop()
        if u in vus: continue
        vus.add(u); pile.extend(par.get(u, []))
    return np.fromiter(vus, int)

P = td.forcing[:, :, 0].numpy()                  # (T, n) mm/j
et_obs = getattr(td, "et_obs", None)
if et_obs is None:
    print("PAS de MODIS chargé sur cette région (td.et_obs absent) — arrêt.")
    sys.exit(0)
E = et_obs.numpy() if hasattr(et_obs, "numpy") else np.asarray(et_obs)
print(f"[donnees] MODIS shape {E.shape} | P shape {P.shape} | {n} noeuds")
Q = td.q_obs.numpy()                             # (T, n_stations) m3/s
sid = td.station_idx.numpy()
sel = (tt >= "2003-01-01") & (tt <= "2021-12-31")
print(f"\n=== {REG.upper()} : bilan SANS MODÈLE sur {sel.sum()/365.25:.0f} ans (mm/an) ===")
print(f"{'jauge':>6s} {'aire km2':>9s} {'P':>7s} {'Q':>7s} {'P-Q':>7s} {'MODIS':>7s} {'MODIS-(P-Q)':>12s}")
lig = []
for k, j in enumerate(sid):
    up = amont(int(j))
    p_ = np.nanmean(P[sel][:, up]) * 365.25
    q_ = Q[sel][:, k]
    v = np.isfinite(q_)
    if v.sum() < 365 * 3: continue
    qmm = np.nanmean(q_[v]) * 86400.0 * 365.25 / (Acum[int(j)] * 1e6) * 1000.0
    if E.ndim == 2 and E.shape[1] == n:
        e_ = np.nanmean(E[sel][:, up]) * 365.25
    else:
        e_ = np.nanmean(E[sel]) * 365.25
    lig.append((int(j), Acum[int(j)], p_, qmm, p_ - qmm, e_, e_ - (p_ - qmm)))
    print(f"{int(j):6d} {Acum[int(j)]:9.0f} {p_:7.0f} {qmm:7.0f} {p_-qmm:7.0f} {e_:7.0f} {e_-(p_-qmm):+12.0f}")
L = np.array(lig)
if len(L):
    print(f"\nMÉDIANES : P {np.median(L[:,2]):.0f} | Q {np.median(L[:,3]):.0f} | "
          f"P-Q (= ETR imposée par le bilan) {np.median(L[:,4]):.0f} | MODIS {np.median(L[:,5]):.0f}")
    ec = np.median(L[:, 6])
    print(f"ÉCART MÉDIAN MODIS − (P−Q) : {ec:+.0f} mm/an "
          f"({100*ec/max(np.median(L[:,4]),1):+.0f} % de l'ETR du bilan)")
    print(f"jauges où MODIS > P−Q (impossible physiquement) : {int((L[:,6]>0).sum())}/{len(L)}")
