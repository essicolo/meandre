"""Compare les champs de parametres de deux points de reprise sur les memes noeuds.

Question (R83) : SLSO-B fait des plateaux d'ete (31,5 % des jours plats contre 2 %
observes), SLSO-A non (6,8 %), meme recette, meme graine. Quels parametres different ?
"""
import sys, os
sys.path.insert(0, ".runs/quebec"); sys.path.insert(0, ".")
import numpy as np, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils import paths as _p

REG = "slso"
d = BasinCache(f"{_p.DATA_ROOT}/quebec/{REG}.duckdb").load(device=torch.device("cpu"))
terr, coords = d["territorial"], d["node_coords"]
n = coords.shape[0]
print(f"{REG.upper()} : {n} noeuds, {terr.data.shape[1]} attributs territoriaux", flush=True)

def champs(ckpt):
    m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False,
                   use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   use_latent_codes=False, predict_lake_params=True)
    m.load(ckpt)
    m.eval()
    with torch.no_grad():
        sp = m.spatial_encoder(coords, terr.to_tensor())
    out = {}
    for k in dir(sp):
        if k.startswith("_"):
            continue
        v = getattr(sp, k)
        if torch.is_tensor(v) and v.numel() >= n:
            out[k] = v.detach().reshape(-1)[:n].double().numpy()
    return out

A = champs("D:/meandre-data/quebec/flotte/best-slso-etl-fdsA.pt")
B = champs("D:/meandre-data/quebec/flotte/best-slso-etl-fdsB.pt")
communs = sorted(set(A) & set(B))
print(f"{'parametre':22s} {'mediane A':>12s} {'mediane B':>12s} {'B/A':>8s} {'disp A':>8s} {'disp B':>8s}")
lignes = []
for k in communs:
    a, b = np.median(A[k]), np.median(B[k])
    if not np.isfinite(a) or not np.isfinite(b):
        continue
    r = b / a if abs(a) > 1e-12 else float("nan")
    lignes.append((abs(np.log10(abs(r))) if np.isfinite(r) and r > 0 else 0.0, k, a, b, r,
                   float(np.std(A[k]) / max(abs(a), 1e-12)), float(np.std(B[k]) / max(abs(b), 1e-12))))
for _, k, a, b, r, da, db in sorted(lignes, reverse=True):
    print(f"{k:22s} {a:12.4g} {b:12.4g} {r:8.3f} {da:8.2f} {db:8.2f}")
