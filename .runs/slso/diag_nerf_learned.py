"""Diagnostic SANS RUN : qu'est-ce que le NeRF a APPRIS sur le run de nuit ?
Charge le checkpoint entraine, fait tourner UNIQUEMENT l'encodeur spatial sur
les noeuds (CPU, aucune simulation), et mesure pour chaque parametre :
  - CV = ecart-type / |moyenne| sur les noeuds (collapse si ~0).
  - |corr| max avec les features territoriales (le NeRF differencie-t-il selon
    le territoire, ou crache-t-il une constante ?).
Si les params de COLONNE sont collapses et decorreles, le NeRF n'apprend pas la
colonne -> l'architecture ne fait rien sur ce bassin.

  python .runs/slso/diag_nerf_learned.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import dataclasses
import numpy as np
import torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.spatial.field_network import SpatialParams

DB = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso/data/slso.duckdb"
CKPT = sys.argv[2] if len(sys.argv) > 2 else ".runs/slso/checkpoints/best-physitel-hydrotel-overnight.pt"
print(f"DB={DB}\nCKPT={CKPT}")

cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
m = HydroModel(**ck["init_kwargs"])
missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
m.eval()
print(f"noeuds={n}  params manquants={len(missing)} inattendus={len(unexpected)}")

with torch.no_grad():
    sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())

# features territoriales (pour la correlation)
feats = h["territorial"].to_tensor().numpy()   # (n, n_feat)
fnames = getattr(h["territorial"], "feature_names", None)

# parametres consommes par la colonne Hydrotel (les autres = routage/thermie/UQ)
COL = {"K_sat_1","K_sat_2","K_sat_3","porosity_1","porosity_2","porosity_3",
       "theta_fc_1","theta_fc_2","theta_fc_3","theta_wp_1","theta_wp_2","theta_wp_3",
       "f_root_1","f_root_2","f_root_3","vg_n","Z2","Z3","K_c","k_gw","f_wetland",
       "f_vert_1","f_vert_2","f_vert_3","C_f","T_melt","rain_hours"}

def cv(x):
    mu = x.mean()
    return float(x.std() / (abs(mu) + 1e-12))

def max_abs_corr(x):
    best = 0.0; bi = -1
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return 0.0, -1
    for j in range(feats.shape[1]):
        fj = feats[:, j]
        if fj.std() < 1e-9:
            continue
        r = float(np.corrcoef(x, fj)[0, 1])
        if abs(r) > abs(best):
            best = r; bi = j
    return best, bi

rows = []
for f in dataclasses.fields(SpatialParams):
    name = f.name
    val = getattr(sp, name)
    if not torch.is_tensor(val):
        continue
    x = val.detach().numpy().astype(np.float64).ravel()
    if x.shape[0] != n:
        continue
    c = cv(x)
    r, bi = max_abs_corr(x)
    fn = (fnames[bi] if (fnames is not None and bi >= 0 and bi < len(fnames)) else (f"feat{bi}" if bi >= 0 else "-"))
    rows.append((name, float(x.mean()), c, r, fn, name in COL))

# tri : params de colonne d'abord, par CV croissant (les plus collapses en tete)
rows_col = sorted([r for r in rows if r[5]], key=lambda z: z[2])
rows_oth = sorted([r for r in rows if not r[5]], key=lambda z: z[2])

def show(title, rs):
    print(f"\n=== {title} ===")
    print(f"{'param':18s} {'moyenne':>12s} {'CV':>8s} {'corr_max':>9s}  feature")
    for name, mu, c, r, fn, _ in rs:
        flag = "  <-- COLLAPSE" if c < 0.05 else ("  (faible)" if c < 0.15 else "")
        print(f"{name:18s} {mu:12.4g} {c:8.3f} {r:9.2f}  {fn}{flag}")

show("PARAMS DE COLONNE (ce que le NeRF doit apprendre)", rows_col)
show("autres params (routage / thermie / UQ)", rows_oth)

ncol = len(rows_col)
ncollapse = sum(1 for r in rows_col if r[2] < 0.05)
nweak = sum(1 for r in rows_col if 0.05 <= r[2] < 0.15)
print(f"\nRESUME colonne : {ncollapse}/{ncol} collapses (CV<0.05), {nweak}/{ncol} faibles (CV<0.15)")
print(f"corr_max mediane (colonne) : {np.median([abs(r[3]) for r in rows_col]):.2f}")
