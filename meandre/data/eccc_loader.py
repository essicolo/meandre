"""Loader stations ECCC climate-daily (MSC GeoMet OGC API) — P/Tmin/Tmax quotidiens.

Source REPRODUCTIBLE et sans authentification (remplace SIMAT/quebec.zarr à yeux de
bœuf) : api.weather.gc.ca/collections/climate-daily. Réseau climatologique canadien
complet, bien plus dense que GHCN au Québec (375 stations dans le bbox GASP).

Sert d'entrée au krigeage PyGMET (station -> nœuds), énergie reprise de CaSR.

  python -m meandre.data.eccc_loader GASP   # fetch + cache une région
"""
from __future__ import annotations
import os, sys, io, json, time, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd
from meandre.utils import paths as _paths

API = "https://api.weather.gc.ca/collections"
CACHE = f"{_paths.DATA_ROOT}/eccc"


def _get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def fetch_stations(bbox, cache_dir=CACHE):
    """Inventaire stations climato ECCC dans bbox (lon0,lat0,lon1,lat1)."""
    os.makedirs(cache_dir, exist_ok=True)
    url = f"{API}/climate-stations/items?bbox={','.join(map(str, bbox))}&limit=10000&f=json"
    d = _get(url)
    rows = []
    for f in d["features"]:
        p = f["properties"]
        rows.append({"climate_id": p.get("CLIMATE_IDENTIFIER"), "name": p.get("STATION_NAME"),
                     "lon": p.get("LONGITUDE"), "lat": p.get("LATITUDE"), "elev": p.get("ELEVATION")})
    return pd.DataFrame(rows).dropna(subset=["climate_id", "lon", "lat"])


def fetch_daily(bbox, year_start, year_end, cache_dir=CACHE):
    """Séries quotidiennes P/Tmin/Tmax dans bbox, année par année (paginé).
    Retourne DataFrame(climate_id, lon, lat, elev, date, P, Tmin, Tmax)."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{'_'.join(f'{b:.1f}' for b in bbox)}_{year_start}_{year_end}"
    fp = Path(cache_dir) / f"daily_{tag}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    allrows = []
    for year in range(year_start, year_end + 1):
        off, got = 0, 0
        while True:
            q = {"bbox": ",".join(map(str, bbox)),
                 "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T00:00:00Z",
                 "limit": 10000, "offset": off, "f": "json",
                 "properties": "CLIMATE_IDENTIFIER,LOCAL_DATE,TOTAL_PRECIPITATION,MIN_TEMPERATURE,MAX_TEMPERATURE"}
            d = _get(f"{API}/climate-daily/items?" + urllib.parse.urlencode(q))
            feats = d["features"]
            for f in feats:
                p = f["properties"]; c = f.get("geometry", {}).get("coordinates") or [None, None]
                allrows.append((p.get("CLIMATE_IDENTIFIER"), c[0], c[1], p.get("LOCAL_DATE"),
                                p.get("TOTAL_PRECIPITATION"), p.get("MIN_TEMPERATURE"), p.get("MAX_TEMPERATURE")))
            got += len(feats); off += len(feats)
            if len(feats) < 10000:
                break
        print(f"  [ECCC] {year}: {got:,} enregistrements", flush=True)
    df = pd.DataFrame(allrows, columns=["climate_id", "lon", "lat", "date", "P", "Tmin", "Tmax"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ("P", "Tmin", "Tmax", "lon", "lat"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "date"])
    df.to_parquet(fp)
    print(f"[ECCC] {len(df):,} lignes | {df.climate_id.nunique()} stations -> {fp}", flush=True)
    return df


if __name__ == "__main__":
    from meandre.data.basin_cache import BasinCache
    reg = sys.argv[1].lower()
    db = ".runs/slso/data/slso.duckdb" if reg == "slso" else f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb"
    nc = BasinCache(db).load(device="cpu")["node_coords"].numpy()
    bbox = (round(float(nc[:, 0].min()) - 0.3, 1), round(float(nc[:, 1].min()) - 0.3, 1),
            round(float(nc[:, 0].max()) + 0.3, 1), round(float(nc[:, 1].max()) + 0.3, 1))
    print(f"[{reg}] bbox {bbox}")
    st = fetch_stations(bbox)
    print(f"[{reg}] {len(st)} stations dans l'inventaire")
    df = fetch_daily(bbox, int(os.environ.get("Y0", 2000)), int(os.environ.get("Y1", 2024)))
