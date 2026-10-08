"""Variante du builder CaSR avec REPROJECTION propre (pôle tourné -> géographique)
au lieu du plus-proche-voisin KDTree. CaSR est sur une grille rotated_latitude_
longitude RÉGULIÈRE en (rlon, rlat). On construit le CRS rotated-pole (rioxarray/
pyproj), on transforme les coords des nœuds DANS l'espace tourné, puis interpolation
BILINÉAIRE sur la grille régulière (pas de resampling intermédiaire). Sortie séparée
forcing-casr-riox.nc pour comparer au cache plus-proche-voisin.
  python .runs/slso/build_casr_forcing_riox.py [--test]
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

CASR = ".runs/slso/data/casr"
# DB cible + sortie paramétrables (env) : PHYSITEL par défaut, ou réseau open-data.
#   CASR_DB=.runs/slso-od/data/basin.duckdb CASR_OUT=.runs/slso-od/data/forcing-casr-riox.nc
DB = os.environ.get("CASR_DB", ".runs/slso/data/slso.duckdb")
OUT = os.environ.get("CASR_OUT", ".runs/slso/data/forcing-casr-riox.nc")
# Mosaïque 2x2 : colonnes rlon (ouest, est) x rangées rlat (sud, nord).
RLON_BLOCKS = ["rlon526-560", "rlon561-595"]
RLAT_BLOCKS = ["rlat351-385", "rlat386-420"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]
SIGMA = 5.670374e-8; ALBEDO = 0.23; EMIS = 0.95
TEST = "--test" in sys.argv

h = BasinCache(DB).load(device="cpu")
nc = h["node_coords"].numpy(); nlon, nlat = nc[:, 0], nc[:, 1]
n_nodes = len(nc)
print(f"noeuds ({DB}) : {n_nodes} -> {OUT}")

# ── CRS rotated-pole depuis les attrs CF du grid_mapping ──
_t0 = f"{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}"
_g = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{_t0}_2000-2003.nc")["rotated_pole"].attrs
rp_crs = CRS.from_cf({
    "grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0)),
})
# CRS géographique sphérique cohérent avec earth_radius CaSR (6370997 m)
geo_crs = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
to_rot = Transformer.from_crs(geo_crs, rp_crs, always_xy=True)
nrlon, nrlat = to_rot.transform(nlon, nlat)   # coords des noeuds en espace tourné (deg)
print(f"CRS rotated-pole OK | noeuds en rlon {nrlon.min():.2f}..{nrlon.max():.2f} rlat {nrlat.min():.2f}..{nrlat.max():.2f}")
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")

def _mosaic_coords():
    """Axes réguliers de la mosaïque : RLON = concat des colonnes, RLAT = concat des rangées."""
    rlons = []
    for rb in RLON_BLOCKS:
        ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{rb}_{RLAT_BLOCKS[0]}_2000-2003.nc")
        rlons.append(ds.rlon.values); ds.close()
    rlats = []
    for ab in RLAT_BLOCKS:
        ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{RLON_BLOCKS[0]}_{ab}_2000-2003.nc")
        rlats.append(ds.rlat.values); ds.close()
    rlon = np.concatenate(rlons); rlat = np.concatenate(rlats)
    assert np.all(np.diff(rlon) > 0), "rlon non monotone après concat colonnes"
    assert np.all(np.diff(rlat) > 0), "rlat non monotone après concat rangées"
    return rlon, rlat
RLON, RLAT = _mosaic_coords()
print(f"grille tournée régulière : rlon {len(RLON)} ({RLON[0]:.3f}..{RLON[-1]:.3f}) rlat {len(RLAT)} ({RLAT[0]:.3f}..{RLAT[-1]:.3f})")
# Garde-fou : tous les nœuds doivent tomber DANS la grille (sinon NaN bilinéaire).
assert nrlon.min() >= RLON[0] and nrlon.max() <= RLON[-1], f"noeuds hors rlon [{RLON[0]:.2f},{RLON[-1]:.2f}]"
assert nrlat.min() >= RLAT[0] and nrlat.max() <= RLAT[-1], f"noeuds hors rlat [{RLAT[0]:.2f},{RLAT[-1]:.2f}]"

def load_var(var, agg):
    daily_all = []
    chunks = CHUNKS[:1] if TEST else CHUNKS
    for ch in chunks:
        times = None
        rows = []   # une entrée par rangée rlat ; chaque entrée = colonnes rlon concaténées
        for ab in RLAT_BLOCKS:
            cols = []
            for rb in RLON_BLOCKS:
                ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_{var}_{rb}_{ab}_{ch}.nc")
                vname = [x for x in ds.data_vars if "CaSR" in x][0]
                cols.append(ds[vname].values); times = pd.to_datetime(ds.time.values); ds.close()
            rows.append(np.concatenate(cols, axis=2))   # concat rlon (axis=2)
        merged = np.concatenate(rows, axis=1)   # concat rlat (axis=1) -> (time, 70, 70)
        da = xr.DataArray(merged, dims=("time", "rlat", "rlon"),
                          coords={"time": times, "rlat": RLAT, "rlon": RLON})
        # interpolation BILINÉAIRE aux noeuds en espace tourné régulier
        samp = da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear")   # (time, node)
        df = pd.DataFrame(samp.values, index=times)
        daily_all.append(getattr(df.resample("1D"), agg)())
    out = pd.concat(daily_all)
    return getattr(out.groupby(out.index), agg)()

def load_intensity():
    """Durée EFFECTIVE d'orage par jour et par nœud (h), depuis le PR0 HORAIRE :
    DT_eff = pluie_journalière / max_horaire, bornée [1, 24]. Court = convectif
    intense (ruisselle), long = frontal diffus (s'infiltre). Récupère l'intensité
    sous-journalière que l'agrégation journalière jette."""
    daily_all = []
    for ch in CHUNKS:
        times = None; rows = []
        for ab in RLAT_BLOCKS:
            cols = []
            for rb in RLON_BLOCKS:
                ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{rb}_{ab}_{ch}.nc")
                vname = [x for x in ds.data_vars if "CaSR" in x][0]
                cols.append(ds[vname].values); times = pd.to_datetime(ds.time.values); ds.close()
            rows.append(np.concatenate(cols, axis=2))
        merged = np.concatenate(rows, axis=1)
        da = xr.DataArray(merged, dims=("time", "rlat", "rlon"),
                          coords={"time": times, "rlat": RLAT, "rlon": RLON})
        samp = da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear")   # (time, node) horaire, m/h
        df = pd.DataFrame(samp.values * 1000.0, index=times)              # mm/h
        dsum = df.resample("1D").sum(); dmax = df.resample("1D").max()
        dt_eff = (dsum / dmax.replace(0.0, np.nan)).clip(1.0, 24.0).fillna(24.0)
        daily_all.append(dt_eff)
    out = pd.concat(daily_all)
    return out.groupby(out.index).min()   # jour-frontière : garde la plus intense

print("P..."); P = load_var("A_PR0_SFC", "sum") * 1000.0
print("Tmin/Tmax..."); TT_min = load_var("A_TT_1.5m", "min"); TT_max = load_var("A_TT_1.5m", "max")
TT_mean = (TT_min + TT_max) / 2.0
print("TD..."); TD = load_var("A_TD_1.5m", "mean")
print("FB/FI..."); FB = load_var("P_FB_SFC", "mean"); FI = load_var("P_FI_SFC", "mean")
print("UVC..."); UVC = load_var("P_UVC_10m", "mean")

idx = P.index
e_a = 0.6108 * np.exp(17.27 * TD / (TD + 237.3))
T_K = TT_mean + 273.15
R_nl = EMIS * SIGMA * T_K**4 - FI
R_n = ((1 - ALBEDO) * FB - R_nl) * 0.0864
R_n = R_n.clip(lower=0.0)
u2 = UVC * 0.748

idx_slice = (idx >= pd.Timestamp("2000-01-01")) & (idx <= pd.Timestamp("2024-12-31"))
idx = idx[idx_slice]
# CASR_EB=1 : ajoute la radiation INCIDENTE brute FB (courte LO) + FI (grande LO),
# en W/m², comme canaux 6,7 pour la fonte ETI / le bilan d'énergie (la neige
# applique son propre albédo, ≠ le R_n net dérivé à albédo 0.23 pour l'ET).
EB = os.environ.get("CASR_EB", "0") == "1"
# CASR_INTENS=1 : ajoute la durée effective d'orage DT_eff (h) comme canal, pour le
# ruissellement hortonien sous-journalier (prec = P/DT_eff au lieu de P/24 fixe).
INTENS = os.environ.get("CASR_INTENS", "0") == "1"
DT_eff = (load_intensity() if INTENS else None)
VARS = ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"] + (["FB", "FI"] if EB else []) + (["DT_eff"] if INTENS else [])
cols = [P, TT_min, TT_max, R_n, u2, e_a] + ([FB, FI] if EB else []) + ([DT_eff] if INTENS else [])
cols = [x.reindex(idx) for x in cols]
forcing = np.stack([c.values.astype(np.float32) for c in cols], axis=-1)
print(f"forcing {forcing.shape} ({'EB 8ch' if EB else '6ch'}) NaN={np.isnan(forcing).any()} "
      f"P_an={float(np.nanmean(cols[0].values))*365:.0f}mm Rn={float(np.nanmean(R_n)):.1f} u2={float(np.nanmean(u2)):.1f}"
      + (f" FB={float(np.nanmean(FB)):.0f}W/m2" if EB else ""))
if TEST:
    print("[test] OK, pas d'écriture"); sys.exit(0)

ds = xr.Dataset({"forcing": (("time", "node", "var"), forcing)},
                coords={"time": idx.values, "node": np.arange(n_nodes), "var": VARS})
ds.to_netcdf(OUT)
print(f"[ok] cache forcage CaSR (riox bilinéaire) : {OUT}")
