"""Correction CaSR ENTIÈREMENT auto-référencée (volume ET timing), depuis l'horaire.
Attaque les deux axes du diagnostic :
  TIMING  : agrégation sur le jour LOCAL (UTC-5, EST) au lieu du jour UTC, pour aligner
            la précip sur le débit observé (jour local CEHQ). Corrige le décalage de
            frontière de jour (~5h) qui misplace les orages de fin de journée.
  VOLUME  : dé-crachinage horaire (heures < seuil = crachin, retirées) puis calage du
            total sur le bilan d'eau flux-tower (1147 mm/an = ET 450 + Q 697).
Aucune dépendance à quebec.zarr. Remplace le canal P d'un forçage existant (T/Rn gardés).
Sortie : forcing-casr-corr.nc.  ENV : DRIZZLE_H (mm/h, déf 0.3), SHIFT_H (déf -5), VOL (déf 1147).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

CASR = ".runs/slso/data/casr"
DB = os.environ.get("CASR_DB", ".runs/slso/data/slso.duckdb")
BASE = os.environ.get("BASE_FORCING", ".runs/slso/data/forcing-casr-riox-intens.nc")  # T/Rn/etc gardés
OUT = ".runs/slso/data/forcing-casr-corr.nc"
DRIZZLE_H = float(os.environ.get("DRIZZLE_H", "0.3"))   # mm/h seuil crachin horaire
SHIFT_H = int(os.environ.get("SHIFT_H", "-5"))          # UTC -> EST local
VOL = float(os.environ.get("VOL", "1147.0"))            # mm/an bilan d'eau
RLON_BLOCKS = ["rlon526-560", "rlon561-595"]; RLAT_BLOCKS = ["rlat351-385", "rlat386-420"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]

h = BasinCache(DB).load(device="cpu"); nc = h["node_coords"].numpy()
_g = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}_2000-2003.nc")["rotated_pole"].attrs
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
nrlon, nrlat = Transformer.from_crs(geo, rp, always_xy=True).transform(nc[:, 0], nc[:, 1])
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")
print(f"noeuds {len(nc)} | seuil crachin {DRIZZLE_H} mm/h | shift {SHIFT_H}h (jour local) | volume {VOL} mm/an")

daily = []
for ch in CHUNKS:
    times = None; rows = []
    for ab in RLAT_BLOCKS:
        cols = []
        for rb in RLON_BLOCKS:
            ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{rb}_{ab}_{ch}.nc")
            v = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[v].values); times = pd.to_datetime(ds.time.values); ds.close()
        rows.append(np.concatenate(cols, axis=2))
    merged = np.concatenate(rows, axis=1)
    _d = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}_{ch}.nc")
    rlon0 = _d.rlon.values; rlat0 = _d.rlat.values; _d.close()
    _d2 = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[1]}_{RLAT_BLOCKS[0]}_{ch}.nc"); rlon1 = _d2.rlon.values; _d2.close()
    _d3 = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[1]}_{ch}.nc"); rlat1 = _d3.rlat.values; _d3.close()
    da = xr.DataArray(merged, dims=("time", "rlat", "rlon"),
                      coords={"time": times, "rlat": np.concatenate([rlat0, rlat1]), "rlon": np.concatenate([rlon0, rlon1])})
    samp = da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear")
    if os.environ.get("DST", "0") == "1":
        # heure avancée : UTC-4 avril-octobre (approx 2e dim. mars - 1er dim. nov.),
        # UTC-5 (SHIFT_H) le reste — aligne l'été sur l'heure locale réelle des jauges
        _off = np.where(times.month.isin(range(4, 11)), SHIFT_H + 1, SHIFT_H)
        idx_local = times + pd.to_timedelta(_off, unit="h")             # TIMING : jour local DST
    else:
        idx_local = times + pd.Timedelta(hours=SHIFT_H)                 # TIMING : jour local
    df = pd.DataFrame(samp.values * 1000.0, index=idx_local)           # mm/h
    kept = df.where(df >= DRIZZLE_H, 0.0)                              # VOLUME/distrib : dé-crachinage
    daily.append(kept.resample("1D").sum())
out = pd.concat(daily); out = out.groupby(out.index).sum()
out = out[(out.index >= pd.Timestamp("2000-01-01")) & (out.index <= pd.Timestamp("2024-12-31"))]
Pcorr = out.values
Pcorr = Pcorr * (VOL / (Pcorr.mean() * 365.25))                        # VOLUME : calage bilan
print(f"P corrigé : {Pcorr.mean()*365.25:.0f} mm/an | jours pluvieux {(Pcorr>0.1).mean()*100:.0f}%")

# remplacer le canal P (0) du forçage de base ; garder T/Rn/etc
b = xr.open_dataset(BASE); F = b["forcing"].values.copy(); VARS = list(b["var"].values.astype(str)); t = b["time"].values; b.close()
assert F.shape[0] == Pcorr.shape[0], f"{F.shape} vs {Pcorr.shape}"
F[:, :, 0] = Pcorr.astype(np.float32)
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
           coords={"time": t, "node": np.arange(F.shape[1]), "var": VARS}).to_netcdf(OUT)
print(f"[ok] {OUT} (P corrigé jour-local + dé-crachiné + volume ; T/Rn de CaSR)")
