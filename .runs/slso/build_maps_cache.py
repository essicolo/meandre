"""Cache des sorties RÉSEAU COMPLET pour les cartes de la présentation.

cache_backbone.py ne garde que les 41 stations. Ici on fait tourner le modèle
sur les 2889 tronçons et on sauvegarde : géométrie (lon/lat, arêtes), débit moyen
par période et paramètres NeRF par tronçon. Consommé par meandre-poc.qmd.

Usage :
  python .runs/slso/build_maps_cache.py \
      --ckpt .runs/slso/checkpoints/best-phenology-modulator.pt \
      --out  .runs/slso/data/cache_maps.npz
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.spatial.field_network import SpatialParams

DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
PARAM_NAMES = [f.name for f in __import__("dataclasses").fields(SpatialParams)]


def main(ckpt_path: str, out_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    init = torch.load(ckpt_path, map_location="cpu", weights_only=False)["init_kwargs"]
    m = HydroModel(**init).to(device)
    m.load(ckpt_path)
    m.eval()

    cache = BasinCache(DB)
    h = cache.load(device=device)
    ds = xr.open_dataset(FORCING)
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    fc = torch.from_numpy(fc_all).to(device)
    doy = torch.tensor(dt_all.dayofyear.values, dtype=torch.long, device=device)
    wd = cache.load_withdrawals(date_start="2000-01-01", date_end="2024-12-31", device=device)

    print(f"Forward réseau complet ({h['n_nodes']} nœuds, {fc.shape[0]} jours)...", flush=True)
    with torch.no_grad():
        Q, _ = m.simulate(
            forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=wd, day_of_year=doy,
        )                                                                # (T, N)
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor()).to_tensor()
    Q = Q.cpu().numpy().astype(np.float32)
    sp = sp.cpu().numpy().astype(np.float32)
    print(f"Q réseau : {Q.shape}, params : {sp.shape}", flush=True)

    # Géométrie
    con = duckdb.connect(DB, read_only=True)
    nodes = con.execute("SELECT node_idx, lon, lat, is_lake FROM nodes ORDER BY node_idx").fetchdf()
    edges = con.execute("SELECT src, dst FROM edges").fetchdf()
    st = con.execute("SELECT node_idx, station_id FROM stations ORDER BY node_idx").fetchdf()
    terr = con.execute("SELECT node_idx, strahler_order, drainage_area_km2 FROM territorial ORDER BY node_idx").fetchdf()
    con.close()

    train_mask = (dt_all >= "2001-01-01") & (dt_all <= "2018-12-31")
    val_mask = (dt_all >= "2019-01-01") & (dt_all <= "2021-12-31")
    test_mask = (dt_all >= "2022-01-01") & (dt_all <= "2024-12-31")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        node_lon=nodes["lon"].values.astype(np.float32),
        node_lat=nodes["lat"].values.astype(np.float32),
        node_is_lake=nodes["is_lake"].values.astype(bool),
        strahler=terr["strahler_order"].values.astype(np.int16),
        drainage_area_km2=terr["drainage_area_km2"].values.astype(np.float32),
        edge_src=edges["src"].values.astype(np.int32),
        edge_dst=edges["dst"].values.astype(np.int32),
        station_node_idx=st["node_idx"].values.astype(np.int32),
        station_ids=np.array(st["station_id"].tolist(), dtype="U16"),
        Q_mean_train=Q[train_mask].mean(0),
        Q_mean_test=Q[test_mask].mean(0),
        spatial_params=sp,                                               # (N, 36)
        param_names=np.array(PARAM_NAMES, dtype="U24"),
    )
    print(f"Cache cartes sauvé : {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=".runs/slso/checkpoints/best-phenology-modulator.pt")
    p.add_argument("--out", default=".runs/slso/data/cache_maps.npz")
    args = p.parse_args()
    main(args.ckpt, args.out)
