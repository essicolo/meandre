"""Compare la variance spatiale du NeRF entre deux checkpoints.

Pré-multiobj  : best-fvert-warmstart.pt (KGE seul, ≈0.904)
Post-multiobj : best-kendall-gal-v3-phase2-boxcox-nll.pt (KGE + ETR + GRACE)
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path.cwd()))

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.paths import run_dir_from_config, resolve_run_path

CONFIG = Path(".runs/slso/config/slso-kendall-gal-v3-phase2-boxcox-nll.toml")
BASELINE_CKPT = Path(".runs/slso/checkpoints/best-fvert-warmstart.pt")

device = torch.device("cpu")

with open(CONFIG, "rb") as f:
    cfg = tomllib.load(f)
RUN_DIR = run_dir_from_config(CONFIG)
def _p(key: str) -> Path:
    return resolve_run_path(cfg["paths"][key], RUN_DIR)

mcfg = cfg["model"]
sc = cfg.get("soil", {})
soil_bounds = {k: sc[k] for k in (
    "z2_min", "z2_max", "z3_min", "z3_max", "rain_hours_min", "rain_hours_max"
) if k in sc}
soil_z1 = sc.get("z1", 0.30)

cache = BasinCache(_p("basin_db"))
hydro = cache.load(device=device)
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
n_nodes = hydro["n_nodes"]

def _build():
    return HydroModel(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=mcfg["n_forcing"],
        context_window=mcfg["context_window"],
        residual_history=mcfg["residual_history"],
        max_travel_time=mcfg["max_travel_days"],
        use_temporal=True,
        use_residual=True,
        use_travel_time_attn=True,
        use_temperature=True,
        dropout=mcfg.get("dropout", 0.0),
        concrete_dropout=mcfg.get("concrete_dropout", False),
        concrete_init_p=mcfg.get("concrete_init_p", 0.05),
        param_mode=mcfg.get("param_mode", "nerf"),
        soil_z1=soil_z1,
        soil_bounds=soil_bounds,
    ).to(device)

ckpt_post = _p("checkpoint")

m_pre = _build(); m_pre.load(str(BASELINE_CKPT)); m_pre.eval()
m_post = _build(); m_post.load(str(ckpt_post)); m_post.eval()

with torch.no_grad():
    p_pre = m_pre.spatial_encoder(node_coords, territorial.to_tensor())
    p_post = m_post.spatial_encoder(node_coords, territorial.to_tensor())

def _stats(v):
    v = np.asarray(v, dtype=float)
    m = float(v.mean()); s = float(v.std())
    return m, s, s / abs(m) if abs(m) > 1e-12 else float("nan")

def _row(name, va, vb):
    ma, sa, cva = _stats(va); mb, sb, cvb = _stats(vb)
    return {
        "param": name,
        "mean_pre": ma, "std_pre": sa, "CV_pre": cva,
        "mean_post": mb, "std_post": sb, "CV_post": cvb,
        "std_ratio": sb / sa if sa > 1e-12 else float("nan"),
    }

rows = []
rows.append(_row("k_gw", p_pre.k_gw.cpu().numpy(), p_post.k_gw.cpu().numpy()))
rows.append(_row("K_c", p_pre.K_c.cpu().numpy(), p_post.K_c.cpu().numpy()))
for i in (1, 2, 3):
    rows.append(_row(f"K_sat_L{i}(log10)",
                     np.log10(getattr(p_pre, f"K_sat_{i}").cpu().numpy()),
                     np.log10(getattr(p_post, f"K_sat_{i}").cpu().numpy())))
for i in (1, 2, 3):
    rows.append(_row(f"f_vert_L{i}",
                     getattr(p_pre, f"f_vert_{i}").cpu().numpy(),
                     getattr(p_post, f"f_vert_{i}").cpu().numpy()))

ks_pre = np.concatenate([getattr(p_pre, f"K_sat_{i}").cpu().numpy() for i in (1, 2, 3)])
ks_post = np.concatenate([getattr(p_post, f"K_sat_{i}").cpu().numpy() for i in (1, 2, 3)])
fv_pre = np.concatenate([getattr(p_pre, f"f_vert_{i}").cpu().numpy() for i in (1, 2, 3)])
fv_post = np.concatenate([getattr(p_post, f"f_vert_{i}").cpu().numpy() for i in (1, 2, 3)])
rows.append(_row("K_sat all(log10)", np.log10(ks_pre), np.log10(ks_post)))
rows.append(_row("f_vert all", fv_pre, fv_post))

df = pd.DataFrame(rows).set_index("param")
print(f"Pré-multiobj  : {BASELINE_CKPT.name}")
print(f"Post-multiobj : {ckpt_post.name}")
print()
print(df.to_string(float_format=lambda x: f"{x:.4g}"))
