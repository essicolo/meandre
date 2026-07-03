"""GHCN-Daily loader (NOAA) — précipitation de stations PUBLIQUES pour SLSO.

Miroir international des stations ECCC (Canada) + US, sans authentification. Sert à
la FUSION CaSR + jauges : la jauge brute au point est plus précise que la cellule
CaSR 10 km lissée, même si CaPA l'a assimilée. Récupère le timing/cumul ponctuel.

Source : https://www.ncei.noaa.gov/pub/data/ghcn/daily/
  - ghcnd-stations.txt : inventaire (ID, lat, lon, elev, nom), largeur fixe.
  - access/{ID}.csv    : série journalière par station (PRCP en 1/10 mm).

  python -m meandre.data.ghcn_loader   # test bbox SLSO
"""
from __future__ import annotations
import os, io, logging, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
ACCESS = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"


def fetch_ghcn_stations(bbox, cache_dir="D:/meandre-data/ghcn"):
    """Inventaire des stations dans le bbox (lon_min,lat_min,lon_max,lat_max)."""
    os.makedirs(cache_dir, exist_ok=True)
    fp = Path(cache_dir) / "ghcnd-stations.txt"
    if not fp.exists() or fp.stat().st_size < 100000:
        print(f"  [GHCN] téléchargement inventaire stations…")
        urllib.request.urlretrieve(f"{BASE}/ghcnd-stations.txt", fp)
    rows = []
    for line in open(fp, encoding="utf-8", errors="ignore"):
        try:
            sid = line[0:11].strip(); lat = float(line[12:20]); lon = float(line[21:30])
            name = line[41:71].strip()
        except ValueError:
            continue
        if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
            rows.append({"station_id": sid, "lat": lat, "lon": lon, "name": name})
    return pd.DataFrame(rows)


def fetch_ghcn_precip(bbox, date_start, date_end, cache_dir="D:/meandre-data/ghcn"):
    """Précipitation journalière (mm) de toutes les stations GHCN du bbox, période
    donnée. Retourne un long df [station_id, lat, lon, date, prcp_mm] (jours mesurés)."""
    st = fetch_ghcn_stations(bbox, cache_dir)
    print(f"  [GHCN] {len(st)} stations dans le bbox SLSO")
    y0, y1 = pd.Timestamp(date_start).year, pd.Timestamp(date_end).year
    os.makedirs(Path(cache_dir) / "access", exist_ok=True)
    out = []
    for _, s in st.iterrows():
        sid = s.station_id
        fp = Path(cache_dir) / "access" / f"{sid}.csv"
        if not fp.exists():
            try:
                urllib.request.urlretrieve(f"{ACCESS}/{sid}.csv", fp)
            except Exception as e:
                logger.warning(f"[GHCN] {sid} : {e}"); continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        if "PRCP" not in df.columns or "DATE" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["DATE"], errors="coerce")
        m = (df.date >= pd.Timestamp(date_start)) & (df.date <= pd.Timestamp(date_end)) & df.PRCP.notna()
        d = df[m][["date", "PRCP"]].copy()
        if len(d) == 0:
            continue
        d["prcp_mm"] = d.PRCP / 10.0   # 1/10 mm -> mm
        d["station_id"] = sid; d["lat"] = s.lat; d["lon"] = s.lon
        out.append(d[["station_id", "lat", "lon", "date", "prcp_mm"]])
    if not out:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "date", "prcp_mm"])
    return pd.concat(out).sort_values(["station_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bbox = (-73.5, 44.0, -69.0, 48.0)   # SLSO + marge
    df = fetch_ghcn_precip(bbox, "2000-01-01", "2024-12-31")
    n_st = df.station_id.nunique() if len(df) else 0
    print(f"\n{len(df):,} obs journalières, {n_st} stations avec données")
    if len(df):
        cov = df.groupby("station_id").agg(n=("prcp_mm", "size"), lat=("lat", "first"),
                                           lon=("lon", "first")).sort_values("n", ascending=False)
        print("top stations (jours) :"); print(cov.head(10).to_string())
        print(f"\nP_moy toutes stations : {df.prcp_mm.mean()*365.25:.0f} mm/an "
              f"(vs CaSR ~1230 mm/an)")
