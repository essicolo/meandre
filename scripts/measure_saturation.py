"""Mesure la saturation des paramètres NeRF à partir d'un checkpoint.

Pour chaque paramètre sigmoid-bounded : % de nodes dont la valeur normalisée
est dans la zone saturée (sigmoid(raw) < 0.05 ou > 0.95).
Pour K_sat / k_gw (log-normal) : statistiques log et déviation vs target literature.

Usage:
    python scripts/measure_saturation.py <config.toml>
"""
from __future__ import annotations
import math
import sys
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(config_path: str):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    ckpt_path = cfg["paths"]["checkpoint"]
    basin_db = cfg["paths"]["basin_db"]

    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    from meandre.spatial.field_network import SpatialFieldNetwork

    # Build a SpatialFieldNetwork. Need to know in_features for territorial.
    # Read it from the checkpoint state_dict fc1.weight shape.
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state
    spatial_sd = {k.replace("spatial_encoder.", ""): v
                  for k, v in sd.items() if k.startswith("spatial_encoder.")}
    if not spatial_sd:
        print(f"ERROR: no spatial_encoder.* keys in state_dict. Got: {list(sd.keys())[:10]}")
        return

    fc1_w = spatial_sd["fc1.weight"]
    in_feat_total = fc1_w.shape[1]
    # coord_enc gives 4 (lon/lat with 1 pe) + 2 = depends. Use default 64.
    # Try a safe default: 4 + 2*L=2*16 = 64? Actually default is L=8 → 4 + 4*8 = 36
    # Just construct with territorial_dim = in_feat_total - encoded_coord_dim.
    # Simpler approach: use a dummy SpatialFieldNetwork and probe.
    # Build with default and let it match.

    from meandre.data.basin_cache import BasinCache
    cache = BasinCache(basin_db)
    hydro = cache.load()
    coords = hydro["node_coords"]
    terr = hydro["territorial"].to_tensor()
    n_nodes = coords.shape[0]
    print(f"Loaded basin : {n_nodes} nodes, territorial dim = {terr.shape[1]}")

    # Instantiate network and load state
    enc = SpatialFieldNetwork(n_territorial=terr.shape[1])
    enc.load_state_dict(spatial_sd, strict=False)
    enc.eval()

    with torch.no_grad():
        # Raw output (avant constraints)
        h0 = enc.coord_enc(coords)
        x = torch.cat([h0, terr], dim=-1)
        h = torch.nn.functional.silu(enc.fc1(x))
        h = torch.cat([h, x], dim=-1)
        h = torch.nn.functional.silu(enc.fc2(h))
        raw = enc.fc_out(h)
        params = enc(coords, terr)

    print(f"\nRaw output shape : {raw.shape}  (expected N_PARAMS=32)")

    # Sigmoid-bounded columns (cf. boundary_regularization)
    sig_cols = (list(range(3, 12)) + list(range(15, 24))
                + [25, 26, 27, 28, 29, 30, 31])
    sig_vals = torch.sigmoid(raw[:, sig_cols]).numpy()

    print("\n" + "=" * 70)
    print("SATURATION DES PARAMÈTRES SIGMOID-BOUNDED")
    print("=" * 70)
    n = sig_vals.shape[0] * sig_vals.shape[1]
    p_sat_low = (sig_vals < 0.05).mean()
    p_sat_high = (sig_vals > 0.95).mean()
    p_sat = p_sat_low + p_sat_high
    p_mid = ((sig_vals >= 0.2) & (sig_vals <= 0.8)).mean()
    print(f"  Total observations    : {n:,}")
    print(f"  Saturé bas (<0.05)    : {100*p_sat_low:.1f}%")
    print(f"  Saturé haut (>0.95)   : {100*p_sat_high:.1f}%")
    print(f"  Total saturé          : {100*p_sat:.1f}%")
    print(f"  Zone milieu [0.2-0.8] : {100*p_mid:.1f}%")
    print(f"  Mean |2σ-1|⁴ (penalty): {((2*sig_vals - 1)**4).mean():.4f}")

    # Per-column saturation
    print("\n  Saturation par paramètre (top 10 plus saturés) :")
    col_sat = ((sig_vals < 0.05) | (sig_vals > 0.95)).mean(axis=0)
    order = np.argsort(-col_sat)
    for i in order[:10]:
        col_idx = sig_cols[i]
        print(f"    col {col_idx:2d} : {100*col_sat[i]:5.1f}% saturé, "
              f"mean σ = {sig_vals[:, i].mean():.3f}, std = {sig_vals[:, i].std():.3f}")

    # Log-normal params (K_sat 0-2, k_gw 24)
    print("\n" + "=" * 70)
    print("PARAMÈTRES LOG-NORMAL (K_sat, k_gw)")
    print("=" * 70)
    LOG_TARGETS = {
        "K_sat_1": (params.K_sat_1, math.log(0.08)),
        "K_sat_2": (params.K_sat_2, math.log(0.04)),
        "K_sat_3": (params.K_sat_3, math.log(0.015)),
        "k_gw":    (params.k_gw,    math.log(0.02)),
    }
    for name, (t, target_log) in LOG_TARGETS.items():
        v = t.detach().numpy()
        log_v = np.log(np.clip(v, 1e-8, None))
        bias = log_v.mean() - target_log
        print(f"  {name:8s} : value mean = {v.mean():.4f}, log-bias vs target = {bias:+.2f}, "
              f"log-std = {log_v.std():.3f}")

    print("\n" + "=" * 70)
    print("SOFTMAX f_vert (partition pluie)")
    print("=" * 70)
    fv = torch.stack([params.f_vert_1, params.f_vert_2, params.f_vert_3], dim=-1).detach().numpy()
    print(f"  f_vert_1 (surface) : mean={fv[:,0].mean():.3f}, std={fv[:,0].std():.3f}")
    print(f"  f_vert_2 (interflow) : mean={fv[:,1].mean():.3f}, std={fv[:,1].std():.3f}")
    print(f"  f_vert_3 (percol)  : mean={fv[:,2].mean():.3f}, std={fv[:,2].std():.3f}")
    # Diversité (Gini-like) — sont-ils tous identiques ?
    print(f"  Diversité spatiale : range f_v1 = [{fv[:,0].min():.3f}, {fv[:,0].max():.3f}]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/measure_saturation.py <config.toml>")
        sys.exit(1)
    main(sys.argv[1])
