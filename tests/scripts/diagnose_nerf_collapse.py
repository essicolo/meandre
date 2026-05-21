"""Diagnostic de la collapse spatiale de la NeRF.

Compare variance spatiale (sur les nodes) à 3 étages :
  1. Features territoriales (input)
  2. NeRF à l'init (init_from_literature, pas de training)
  3. NeRF après training (checkpoint fourni)

Objectif : localiser où la variance se perd.
"""
from __future__ import annotations
import sys
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PARAM_NAMES = [
    "K_sat_1", "K_sat_2", "K_sat_3",
    "porosity_1", "porosity_2", "porosity_3",
    "theta_fc_1", "theta_fc_2", "theta_fc_3",
    "theta_wp_1", "theta_wp_2", "theta_wp_3",
    "f_root_1", "f_root_2", "f_root_3",
    "C_f", "T_melt", "T_snow", "alpha_T",
    "vg_n", "frost_alpha",
    "interception",
    "f_vert_1", "f_vert_3",
    "k_gw", "T_gw",
    "K_atm", "f_wetland",
    "K_musk", "x_musk",
    "f_vert_2", "rain_hours",
]


def variance_summary(arr: np.ndarray, names=None):
    """arr shape (n_nodes, n_params). Print mean/std/cv per param column."""
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    cv = np.divide(stds, np.abs(means) + 1e-8)
    rng = arr.max(axis=0) - arr.min(axis=0)
    if names is None:
        names = [f"col{i}" for i in range(arr.shape[1])]
    rows = []
    for i, n in enumerate(names):
        rows.append((n, means[i], stds[i], cv[i], rng[i]))
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print(f"  {'name':12s}  {'mean':>10s}  {'std':>10s}  {'CV%':>7s}  {'range':>10s}")
    for n, m, s, c, r in rows:
        marker = "  <-- FLAT" if c < 0.01 else ("  <-- low" if c < 0.05 else "")
        print(f"  {n:12s}  {m:10.4f}  {s:10.4f}  {100*c:7.2f}  {r:10.4f}{marker}")


def main(config_path: str):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    ckpt_path = cfg["paths"]["checkpoint"]
    basin_db = cfg["paths"]["basin_db"]

    from meandre.data.basin_cache import BasinCache
    cache = BasinCache(basin_db)
    hydro = cache.load()
    coords = hydro["node_coords"]
    terr = hydro["territorial"].to_tensor()
    n_nodes = coords.shape[0]

    print("=" * 72)
    print(f"BASIN : {n_nodes} nodes, lon range [{coords[:,0].min():.3f}, "
          f"{coords[:,0].max():.3f}], lat range [{coords[:,1].min():.3f}, "
          f"{coords[:,1].max():.3f}]")
    print("=" * 72)

    # === ÉTAGE 1 : Features territoriales ===
    terr_np = terr.numpy()
    print("\n" + "=" * 72)
    print("(1) VARIANCE DES FEATURES TERRITORIALES (input)")
    print("=" * 72)
    feat_names = [f"terr_{i}" for i in range(terr_np.shape[1])]
    # Try to get real names
    try:
        feat_obj = hydro["territorial"]
        if hasattr(feat_obj, "feature_names"):
            feat_names = list(feat_obj.feature_names)
    except Exception:
        pass
    rows = variance_summary(terr_np, feat_names)
    print_table(rows, "Features territoriales :")

    # === ÉTAGE 2 : NeRF à l'init (literature, no training) ===
    from meandre.spatial.field_network import SpatialFieldNetwork
    print("\n" + "=" * 72)
    print("(2) NeRF À L'INIT (init_from_literature, pas de training)")
    print("=" * 72)
    enc_init = SpatialFieldNetwork(n_territorial=terr.shape[1])
    if hasattr(enc_init, "init_from_literature"):
        enc_init.init_from_literature()
    enc_init.eval()
    with torch.no_grad():
        p_init = enc_init(coords, terr)

    def collect(params):
        out = []
        for name in PARAM_NAMES:
            t = getattr(params, name, None)
            if t is None: continue
            out.append((name, t.detach().cpu().numpy()))
        return out

    init_arr = np.stack([v for _, v in collect(p_init)], axis=1)
    init_names = [n for n, _ in collect(p_init)]
    rows = variance_summary(init_arr, init_names)
    print_table(rows, "Paramètres NeRF à l'init :")

    # === ÉTAGE 3 : NeRF après training ===
    print("\n" + "=" * 72)
    print(f"(3) NeRF APRÈS TRAINING (checkpoint : {ckpt_path})")
    print("=" * 72)
    if not Path(ckpt_path).exists():
        print("ERROR: checkpoint not found.")
        return
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    spatial_sd = {k.replace("spatial_encoder.", ""): v
                  for k, v in sd.items() if k.startswith("spatial_encoder.")}
    enc_post = SpatialFieldNetwork(n_territorial=terr.shape[1])
    enc_post.load_state_dict(spatial_sd, strict=False)
    enc_post.eval()
    with torch.no_grad():
        p_post = enc_post(coords, terr)
    post_arr = np.stack([v for _, v in collect(p_post)], axis=1)
    post_names = [n for n, _ in collect(p_post)]
    rows = variance_summary(post_arr, post_names)
    print_table(rows, "Paramètres NeRF après training :")

    # === ÉTAGE 4 : Comparaison init vs post ===
    print("\n" + "=" * 72)
    print("(4) COMPARAISON INIT vs POST-TRAINING — perte de variance par param")
    print("=" * 72)
    print(f"  {'name':12s}  {'std_init':>10s}  {'std_post':>10s}  "
          f"{'ratio':>7s}  diagnostic")
    for i, name in enumerate(init_names):
        s_init = init_arr[:, i].std()
        s_post = post_arr[:, i].std()
        ratio = s_post / max(s_init, 1e-10)
        if s_init < 1e-6 and s_post < 1e-6:
            diag = "INIT déjà flat"
        elif ratio < 0.1:
            diag = "COLLAPSE training"
        elif ratio < 0.5:
            diag = "shrinkage"
        elif ratio > 2.0:
            diag = "expansion"
        else:
            diag = "stable"
        print(f"  {name:12s}  {s_init:10.4f}  {s_post:10.4f}  "
              f"{ratio:7.2f}  {diag}")

    # === ÉTAGE 5 : Inspect raw fc_out outputs ===
    print("\n" + "=" * 72)
    print("(5) RAW OUTPUTS DE fc_out (avant constraints sigmoid/exp)")
    print("=" * 72)
    for label, encoder in [("init", enc_init), ("post", enc_post)]:
        with torch.no_grad():
            h0 = encoder.coord_enc(coords)
            x0 = torch.cat([h0, terr], dim=-1)
            h = torch.nn.functional.silu(encoder.fc1(x0))
            h = torch.cat([h, x0], dim=-1)
            h = torch.nn.functional.silu(encoder.fc2(h))
            raw = encoder.fc_out(h).numpy()
        print(f"\n  raw fc_out [{label}] : mean across-nodes std = {raw.std(axis=0).mean():.4f}")
        print(f"  raw fc_out [{label}] : max across-nodes std    = {raw.std(axis=0).max():.4f}")
        print(f"  raw fc_out [{label}] : min across-nodes std    = {raw.std(axis=0).min():.4f}")

    # === ÉTAGE 6 : Inspect fc1/fc2 weights magnitudes ===
    print("\n" + "=" * 72)
    print("(6) MAGNITUDES DES POIDS NeRF — fc1, fc2, fc_out")
    print("=" * 72)
    for label, encoder in [("init", enc_init), ("post", enc_post)]:
        print(f"\n  [{label}]")
        for name, p in encoder.named_parameters():
            if not any(k in name for k in ["fc1", "fc2", "fc_out"]):
                continue
            print(f"    {name:30s}  shape={tuple(p.shape)}  "
                  f"mean={p.mean().item():+.4f}  std={p.std().item():.4f}  "
                  f"max|abs|={p.abs().max().item():.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/diagnose_nerf_collapse.py <config.toml>")
        sys.exit(1)
    main(sys.argv[1])
