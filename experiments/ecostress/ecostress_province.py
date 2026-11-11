"""ETR journalière d'ECOSTRESS (produit L3T JET, couche ETdaily, mm/j) aux tronçons du domaine provincial,
de mai à septembre 2019-2024, sans garder les images.

Pour chaque scène de jour (passage de 14 h à 21 h UTC, soit 9 h à 16 h à l'heure solaire), les couches
ETdaily, cloud et water sont téléchargées dans un dossier temporaire ; la valeur de chaque tronçon est
la moyenne des pixels de 70 m sans nuage ni eau d'une fenêtre de 7 x 7 pixels centrée sur son point,
gardée si au moins la moitié des pixels sont valides (même règle que meandre-analyse,
`ecostress_reach_daily`) ; les images sont ensuite effacées. Le domaine entier tient en quelques
centaines de mégaoctets au lieu de plusieurs centaines de gigaoctets d'images.

Sortie : un parquet par mois, colonnes reach (numéro du tronçon dans l'ordre provincial), date,
hour_utc, tile, et, valid_fraction. Reprise : un mois déjà écrit est sauté ; dans un mois en cours, les
scènes déjà traitées sont relues du fichier d'avancement.

Usage :
    python ecostress_province.py <coordonnées des tronçons .npz> <dossier de sortie> [--threads 6]
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

YEARS = range(2019, 2025)
MONTHS = range(5, 10)
HOURS_UTC = range(14, 22)
HALF_WINDOW = 3
LAYERS = ("_ETdaily.tif", "_cloud.tif", "_water.tif")


def fetch(session, url, target, attempts=4):
    for attempt in range(attempts):
        try:
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5.0 * (attempt + 1))


def extract(paths, coords):
    """Valeurs par tronçon d'une scène (DataFrame), à partir de ses trois couches."""
    k = HALF_WINDOW
    with rasterio.open(paths["_ETdaily.tif"]) as src:
        et = src.read(1).astype(np.float32)
        x, y = transform("EPSG:4326", src.crs, coords[:, 0], coords[:, 1])
        rows, cols = (np.asarray(a) for a in rasterio.transform.rowcol(src.transform, x, y))
        height, width = src.shape
    inside = np.nonzero((rows >= k) & (rows < height - k) & (cols >= k) & (cols < width - k))[0]
    if len(inside) == 0:
        return None
    for layer in ("_cloud.tif", "_water.tif"):
        if layer in paths:
            with rasterio.open(paths[layer]) as src:
                et = np.where(src.read(1) == 1, np.nan, et)
    offsets = np.arange(-k, k + 1)
    window = et[rows[inside, None, None] + offsets[None, :, None], cols[inside, None, None] + offsets[None, None, :]].reshape(len(inside), -1)
    valid = np.isfinite(window).mean(axis=1)
    keep = valid >= 0.5
    if not keep.any():
        return None
    return pd.DataFrame({"reach": inside[keep].astype(np.int32), "et": np.nanmean(window[keep], axis=1).astype(np.float32), "valid_fraction": valid[keep].astype(np.float32)})


def process(granule, session, coords, tmp):
    """(nom, valeurs) d'une scène ; valeurs vaut None sans tronçon couvert, False en cas d'échec."""
    try:
        return _process(granule, session, coords, tmp)
    except Exception as e:
        print(f"  ÉCHEC {granule['meta'].get('native-id', '')} : {e}", flush=True)
        return granule["meta"].get("native-id", ""), False


def _process(granule, session, coords, tmp):
    name = granule["meta"]["native-id"]
    tile, day, hour = re.search(r"_(\w{5})_(\d{8})T(\d{2})", name).groups()
    folder = Path(tempfile.mkdtemp(dir=tmp))
    try:
        paths = {}
        for url in granule.data_links():
            for layer in LAYERS:
                if url.endswith(layer):
                    paths[layer] = folder / url.split("/")[-1]
                    fetch(session, url, paths[layer])
        if "_ETdaily.tif" not in paths:
            return name, None
        out = extract(paths, coords)
        if out is not None:
            out["date"] = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
            out["hour_utc"] = int(hour)
            out["tile"] = tile
        return name, out
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coords")
    ap.add_argument("output")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    coords = np.load(args.coords)["coords"]
    out = Path(args.output)
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    bbox = (float(coords[:, 0].min()) - 0.1, float(coords[:, 1].min()) - 0.1, float(coords[:, 0].max()) + 0.1, float(coords[:, 1].max()) + 0.1)
    if not earthaccess.login(strategy="netrc").authenticated:
        raise SystemExit("authentification Earthdata impossible (.netrc)")
    session = earthaccess.get_requests_https_session()
    for year in YEARS:
        for month in MONTHS:
            target = out / f"ecostress-{year}-{month:02d}.parquet"
            if target.exists():
                continue
            last = pd.Period(f"{year}-{month:02d}").end_time.strftime("%Y-%m-%d")
            found = earthaccess.search_data(short_name="ECO_L3T_JET", bounding_box=bbox, temporal=(f"{year}-{month:02d}-01", last), count=-1)
            unique = {}
            for g in found:
                name = g["meta"].get("native-id", "")
                m = re.search(r"_\w{5}_\d{8}T(\d{2})", name)
                if m and int(m.group(1)) in HOURS_UTC:
                    unique[name] = g
            progress = out / f"ecostress-{year}-{month:02d}.avancement"
            parts_dir = out / f"ecostress-{year}-{month:02d}.parties"
            parts_dir.mkdir(exist_ok=True)
            done = set(progress.read_text().split()) if progress.exists() else set()
            todo = [g for n, g in unique.items() if n not in done]
            print(f"{year}-{month:02d} : {len(unique)} scènes de jour, {len(todo)} à traiter", flush=True)
            t0, n_done, failed = time.time(), 0, 0
            with ThreadPoolExecutor(args.threads) as pool, open(progress, "a") as log:
                for name, df in pool.map(lambda g: process(g, session, coords, tmp), todo):
                    if df is False:
                        failed += 1
                        continue
                    if df is not None:
                        df.to_parquet(parts_dir / f"{name}.parquet")
                    log.write(name + "\n")
                    log.flush()
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  {n_done}/{len(todo)} en {time.time() - t0:.0f} s", flush=True)
            if failed:
                print(f"{year}-{month:02d} : {failed} échecs, mois laissé ouvert (relancer pour les reprendre)", flush=True)
                continue
            parts = sorted(parts_dir.glob("*.parquet"))
            month_df = pd.concat([pd.read_parquet(p) for p in parts]) if parts else pd.DataFrame()
            month_df.to_parquet(target)
            shutil.rmtree(parts_dir)
            print(f"{year}-{month:02d} écrit : {len(month_df)} couples tronçon-jour", flush=True)


if __name__ == "__main__":
    main()
