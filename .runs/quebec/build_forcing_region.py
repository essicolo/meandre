"""Forçage CaSR CORRIGÉ pour une région PHYSITEL (recette championne SLSO généralisée).
Canaux : P corrigé (dé-crachinage horaire 0.3 mm/h + agrégation jour-local UTC-5 +
calage volume sur le bilan d'eau RÉGIONAL lame_obs_train + ETR 450), Tmin, Tmax,
R_n (FAO-56 depuis FB/FI), u2, e_a. Interpolation bilinéaire en espace pôle tourné.
Tuiles cherchées dans D:/meandre-data/casr puis .runs/slso/data/casr (legacy).
  python .runs/quebec/build_forcing_region.py GASP [--test]
Sortie : D:/meandre-data/quebec/forcing-<reg>.nc. ENV : VOL (mm/an, sinon bilan), DRIZZLE_H.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

REG = sys.argv[1].upper()
TEST = "--test" in sys.argv
DB = f"D:/meandre-data/quebec/{REG.lower()}.duckdb"
OUT = f"D:/meandre-data/quebec/forcing-{REG.lower()}.nc"
CASR_DIRS = ["D:/meandre-data/casr", ".runs/slso/data/casr"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]
SIGMA = 5.670374e-8; ALBEDO = 0.23; EMIS = 0.95
DRIZZLE_H = float(os.environ.get("DRIZZLE_H", "0.3")); SHIFT_H = -5; ET_MM = 450.0

def casr_path(fn):
    for d in CASR_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p): return p
    raise FileNotFoundError(fn)

h = BasinCache(DB).load(device="cpu")
nc_ = h["node_coords"].numpy(); n_nodes = h["n_nodes"]

# volume cible : bilan d'eau régional (train seulement) ou VOL env ; fallback 1147
if os.environ.get("VOL"):
    VOL = float(os.environ["VOL"])
else:
    c = duckdb.connect(DB, read_only=True)
    try:
        st = c.execute("""SELECT s.station_id, s.drainage_area_km2, AVG(o.discharge) q
                          FROM stations s JOIN observations o ON s.station_id = o.station_id
                          WHERE o.date <= '2021-12-31' GROUP BY 1, 2""").fetchdf()
    finally: c.close()
    st = st.dropna()
    if len(st) >= 2:
        lame = (st.q * 31_557_600.0 / (st.drainage_area_km2 * 1e6) * 1000.0).median()
        VOL = float(lame + ET_MM)
    else:
        VOL = 1147.0
        print(f"[{REG}] pas assez de jauges — volume fallback 1147")
print(f"[{REG}] {n_nodes} nœuds | volume cible {VOL:.0f} mm/an")

# CRS pôle tourné + grille mosaïque depuis les tuiles de la région
_g = xr.open_dataset(casr_path("CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc"))["rotated_pole"].attrs
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
nrlon, nrlat = Transformer.from_crs(geo, rp, always_xy=True).transform(nc_[:, 0], nc_[:, 1])
_reftile = xr.open_dataset(casr_path("CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc"))
dr_lon = float(np.diff(_reftile.rlon.values).mean()); dr_lat = float(np.diff(_reftile.rlat.values).mean())
rlon0 = _reftile.rlon.values[0] - 526 * dr_lon; rlat0 = _reftile.rlat.values[0] - 351 * dr_lat
_reftile.close()
def blocks(vals, orig, step):
    k = sorted(set((np.round((vals - orig) / step).astype(int) - 1) // 35))
    return [f"{35*b+1}-{35*b+35}" for b in k]
RLON_BLOCKS = [f"rlon{b}" for b in blocks(nrlon, rlon0, dr_lon)]
RLAT_BLOCKS = [f"rlat{b}" for b in blocks(nrlat, rlat0, dr_lat)]
print(f"[{REG}] mosaïque {len(RLON_BLOCKS)}x{len(RLAT_BLOCKS)} : {RLON_BLOCKS} x {RLAT_BLOCKS}")
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")

def _axes():
    rlons, rlats = [], []
    for rb in RLON_BLOCKS:
        d = xr.open_dataset(casr_path(f"CaSR_v3.2_A_TT_1.5m_{rb}_{RLAT_BLOCKS[0]}_2000-2003.nc")); rlons.append(d.rlon.values); d.close()
    for ab in RLAT_BLOCKS:
        d = xr.open_dataset(casr_path(f"CaSR_v3.2_A_TT_1.5m_{RLON_BLOCKS[0]}_{ab}_2000-2003.nc")); rlats.append(d.rlat.values); d.close()
    return np.concatenate(rlons), np.concatenate(rlats)
RLON, RLAT = _axes()
assert np.all(np.diff(RLON) > 0) and np.all(np.diff(RLAT) > 0)
assert nrlon.min() >= RLON[0] and nrlon.max() <= RLON[-1] and nrlat.min() >= RLAT[0] and nrlat.max() <= RLAT[-1], "nœuds hors mosaïque"

def hourly_chunk(var, ch):
    times = None; rows = []
    for ab in RLAT_BLOCKS:
        cols = []
        for rb in RLON_BLOCKS:
            ds = xr.open_dataset(casr_path(f"CaSR_v3.2_{var}_{rb}_{ab}_{ch}.nc"))
            v = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[v].values); times = pd.to_datetime(ds.time.values); ds.close()
        rows.append(np.concatenate(cols, axis=2))
    da = xr.DataArray(np.concatenate(rows, axis=1), dims=("time", "rlat", "rlon"),
                      coords={"time": times, "rlat": RLAT, "rlon": RLON})
    return da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear"), times

def load_daily(var, agg):
    parts = []
    for ch in (CHUNKS[:1] if TEST else CHUNKS):
        samp, times = hourly_chunk(var, ch)
        parts.append(getattr(pd.DataFrame(samp.values, index=times).resample("1D"), agg)())
    out = pd.concat(parts)
    return getattr(out.groupby(out.index), agg)()

def load_p_corr():
    parts = []
    for ch in (CHUNKS[:1] if TEST else CHUNKS):
        samp, times = hourly_chunk("A_PR0_SFC", ch)
        idx_local = times + pd.Timedelta(hours=SHIFT_H)              # TIMING : jour local
        df = pd.DataFrame(samp.values * 1000.0, index=idx_local)     # mm/h
        kept = df.where(df >= DRIZZLE_H, 0.0)                        # dé-crachinage
        parts.append(kept.resample("1D").sum())
    out = pd.concat(parts)
    return out.groupby(out.index).sum()

print("P corrigé..."); P = load_p_corr()
print("Tmin/Tmax..."); Tmin = load_daily("A_TT_1.5m", "min"); Tmax = load_daily("A_TT_1.5m", "max")
print("TD..."); TD = load_daily("A_TD_1.5m", "mean")
print("FB/FI..."); FB = load_daily("P_FB_SFC", "mean"); FI = load_daily("P_FI_SFC", "mean")
print("UVC..."); UVC = load_daily("P_UVC_10m", "mean")

idx = pd.date_range("2000-01-01", "2024-12-31", freq="D")
if TEST: idx = pd.date_range("2000-01-01", "2003-12-30", freq="D")
Tmean = (Tmin + Tmax) / 2.0
e_a = 0.6108 * np.exp(17.27 * TD / (TD + 237.3))
R_n = (((1 - ALBEDO) * FB - (EMIS * SIGMA * (Tmean + 273.15)**4 - FI)) * 0.0864).clip(lower=0.0)
u2 = UVC * 0.748
Pv = P.reindex(idx).values
Pv = Pv * (VOL / (np.nanmean(Pv) * 365.25))                          # VOLUME : calage bilan
cols = [None, Tmin, Tmax, R_n, u2, e_a]
arr = [Pv] + [c.reindex(idx).values for c in cols[1:]]
F = np.stack([a.astype(np.float32) for a in arr], axis=-1)
print(f"forcing {F.shape} NaN={np.isnan(F).any()} | P {np.nanmean(Pv)*365.25:.0f} mm/an | "
      f"jours pluvieux {(Pv > 0.1).mean()*100:.0f}% | Rn {np.nanmean(arr[3]):.1f}")
if TEST:
    print("[test] OK"); sys.exit(0)
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F)},
           coords={"time": idx, "node": np.arange(n_nodes), "var": ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"]}).to_netcdf(OUT)
print(f"[ok] {OUT}")
