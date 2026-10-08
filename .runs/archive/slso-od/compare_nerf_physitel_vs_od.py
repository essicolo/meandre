"""Compare les champs de paramètres NeRF entre PHYSITEL (val~0.83) et OPEN-DATA
(val~0.77) pour tester l'équifinalité / le collapse.

Deux signatures :
  1. CV spatial (std/|mean|) par paramètre — faible = collapsé (le NeRF sort la
     même valeur partout, n'exploite pas l'hétérogénéité).
  2. corr(param, feature physique driver) — faible = décorrélé (le NeRF ignore
     le driver territorial, signe d'équifinalité).

  uv run python .runs/slso-od/compare_nerf_physitel_vs_od.py
"""
from __future__ import annotations
import os, sys, tomllib
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path.cwd()))

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel

device = torch.device("cpu")

CASES = {
    "PHYSITEL": (".runs/slso/config/slso-phenology-no-gru.toml",
                 ".runs/slso/checkpoints/best-phenology-no-gru.pt"),
    "OPEN-DATA": (".runs/slso-od/config/slso-od-rebal.toml",
                  ".runs/slso-od/checkpoints/best-rebal.pt"),
}

def build_and_extract(cfg_path, ckpt_path):
    cfg = tomllib.load(open(cfg_path, "rb"))
    mcfg = cfg["model"]; sc = cfg.get("soil", {}); tr = cfg["training"]
    soil_bounds = {k: sc[k] for k in ("z2_min","z2_max","z3_min","z3_max",
                  "rain_hours_min","rain_hours_max") if k in sc}
    cache = BasinCache(cfg["paths"]["basin_db"] if Path(cfg["paths"]["basin_db"]).is_absolute()
                       else str(Path(cfg_path).resolve().parent.parent / cfg["paths"]["basin_db"]))
    hydro = cache.load(device=device)
    terr = hydro["territorial"]; coords = hydro["node_coords"]
    model = HydroModel(
        n_nodes=hydro["n_nodes"], n_territorial=terr.n_features,
        n_forcing=mcfg["n_forcing"], context_window=mcfg["context_window"],
        residual_history=mcfg["residual_history"], max_travel_time=mcfg["max_travel_days"],
        use_temporal=tr.get("enable_temporal_epoch",0)<9999,
        use_residual=tr.get("enable_residual_epoch",9999)<9999,
        use_travel_time_attn=tr.get("enable_travel_epoch",9999)<9999,
        use_temperature=mcfg.get("use_temperature",True),
        dropout=mcfg.get("dropout",0.0),
        param_mode=mcfg.get("param_mode","nerf"),
        soil_z1=sc.get("z1",0.30), soil_bounds=soil_bounds,
        use_phenology_modulator=mcfg.get("use_phenology_modulator",False),
        routing_mode=mcfg.get("routing_mode","level"),
        predict_lake_params=mcfg.get("predict_lake_params",False),
    ).to(device)
    model.load(str(ckpt_path)); model.eval()
    with torch.no_grad():
        p = model.spatial_encoder(coords, terr.to_tensor())
    # territorial features as dict
    cols = terr.columns
    feat = {c: terr.data[:, i].cpu().numpy() for i, c in enumerate(cols)}
    return p, feat

def cv(v):
    v = np.asarray(v, float); m = v.mean()
    return v.std()/abs(m) if abs(m) > 1e-12 else float("nan")

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

results = {}
for name, (cfg_p, ckpt_p) in CASES.items():
    print(f"--- {name}: {Path(ckpt_p).name} ---", flush=True)
    p, feat = build_and_extract(cfg_p, ckpt_p)
    results[name] = (p, feat)

# Paramètres à comparer (présents dans les deux) + leur feature driver attendue
PARAMS = [
    ("K_sat_1", "f_sand", True),   # K_sat ↔ sable (log)
    ("K_sat_2", "f_sand", True),
    ("K_sat_3", "f_sand", True),
    ("f_vert_1", "mean_slope_pct", False),
    ("f_vert_2", "mean_slope_pct", False),
    ("f_vert_3", "mean_slope_pct", False),
    ("K_c", "f_forest", False),    # coef cultural ↔ forêt
    ("k_gw", "f_wetland", False),  # récession aquifère ↔ milieux humides
]

rows = []
for pname, fdriver, islog in PARAMS:
    row = {"param": pname, "driver": fdriver}
    for case in ("PHYSITEL", "OPEN-DATA"):
        p, feat = results[case]
        if not hasattr(p, pname):
            row[f"CV_{case[:4]}"] = float("nan"); row[f"r_{case[:4]}"] = float("nan"); continue
        v = getattr(p, pname).cpu().numpy()
        vv = np.log10(np.maximum(v, 1e-12)) if islog else v
        row[f"CV_{case[:4]}"] = cv(v)
        row[f"r_{case[:4]}"] = corr(vv, feat[fdriver]) if fdriver in feat else float("nan")
    rows.append(row)

df = pd.DataFrame(rows).set_index("param")
pd.set_option("display.width", 160)
print()
print("CV = écart-type spatial / |moyenne| (faible = collapsé)")
print("r  = corr(param, feature driver) (faible |r| = décorrélé du driver physique)")
print()
print(df.to_string(float_format=lambda x: f"{x:.3f}"))
