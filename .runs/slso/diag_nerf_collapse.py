"""Le NeRF CaSR est-il collapsé (params uniformes) vs le NeRF quebec.zarr ?
Charge 2 checkpoints, sort les params spatiaux par nœud, compare le coefficient
de variation (CV = std/|mean|) à travers les nœuds, paramètre par paramètre.
CV effondré sous CaSR = NeRF qui ne différencie plus l'espace (mode d'échec connu).
  python .runs/slso/diag_nerf_collapse.py CKPT_A CKPT_B
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel

DB = ".runs/slso/data/slso.duckdb"
CKPTS = sys.argv[1:3] if len(sys.argv) >= 3 else [
    ".runs/slso/checkpoints/best-physitel-hydrotel-multiobj.pt",
    ".runs/slso/checkpoints/best-physitel-hydrotel-casr.pt"]
h = BasinCache(DB).load(device="cpu")

def params_cv(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False
    m = HydroModel(**kw); m.load_state_dict(ck["state_dict"], strict=False); m.eval()
    with torch.no_grad():
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
    out = {}
    for k, v in vars(sp).items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.numel() >= h["n_nodes"]:
            x = v.reshape(-1).float().numpy()
            mu = np.abs(np.mean(x)) + 1e-9
            out[k] = (float(np.std(x) / mu), float(np.mean(x)))
    return out

a = params_cv(CKPTS[0]); b = params_cv(CKPTS[1])
print(f"{'param':22s} {'CV_A':>9s} {'CV_B':>9s} {'ratio B/A':>10s}   (A={os.path.basename(CKPTS[0])[:30]}  B={os.path.basename(CKPTS[1])[:30]})")
keys = sorted(set(a) & set(b))
ratios = []
for k in keys:
    cva, cvb = a[k][0], b[k][0]
    rr = cvb / (cva + 1e-9)
    ratios.append(rr)
    flag = "  <-- B PLUS PLAT" if rr < 0.5 else ("  <-- B plus varie" if rr > 2 else "")
    print(f"{k:22s} {cva:9.4f} {cvb:9.4f} {rr:10.2f}{flag}")
print(f"\nratio médian CV(B)/CV(A) sur {len(keys)} params : {np.median(ratios):.2f}  "
      f"(<1 = B plus uniforme/collapsé, ~1 = équivalent)")
