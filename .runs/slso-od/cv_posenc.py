"""CV des params du NeRF pour un checkpoint donné (build avec n_coord_freqs=8).

Compare au baseline collapsé (best-rebal, valeurs établies) et à PHYSITEL.
  uv run python .runs/slso-od/cv_posenc.py [checkpoint]
"""
import os, sys
from pathlib import Path
import torch, numpy as np
os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path.cwd()))
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel

CKPT = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso-od/checkpoints/best-posenc.pt"
h = BasinCache(".runs/slso-od/data/basin.duckdb").load(device=torch.device("cpu"))
terr = h["territorial"]; co = h["node_coords"]
m = HydroModel(n_nodes=h["n_nodes"], n_territorial=terr.n_features, n_forcing=6,
    context_window=30, residual_history=14, max_travel_time=20, use_temperature=False,
    param_mode="nerf", routing_mode="operator-lagged", predict_lake_params=True,
    n_coord_freqs=8)
m.load(CKPT); m.eval()
with torch.no_grad():
    p = m.spatial_encoder(co, terr.to_tensor())

# Baselines établis (CV linéaire) : collapsé (best-rebal) et PHYSITEL.
COLLAPSE = {"K_sat_1": 0.006, "f_vert_1": 0.066, "f_vert_2": 0.094, "K_c": 0.006, "k_gw": 0.010}
PHYSITEL = {"K_sat_1": 0.217, "f_vert_1": 0.384, "f_vert_2": 0.466, "K_c": 0.057, "k_gw": 0.038}
print(f"Checkpoint: {CKPT}")
print(f"{'param':10} {'CV collapse':>12} {'CV posenc':>10} {'CV PHYSITEL':>12}")
for k in COLLAPSE:
    v = getattr(p, k).cpu().numpy()
    cv = float(v.std() / abs(v.mean()))
    print(f"{k:10} {COLLAPSE[k]:12.3f} {cv:10.3f} {PHYSITEL[k]:12.3f}")
