"""Driver SLSO du merge CaSR x grille de stations (logique dans meandre/data/forcing_correction.py).
Ratio MENSUEL climatologique par noeud vers la reference (SIMAT/quebec.zarr aujourd'hui,
PyGMET au scale-up), volume final = bilan flux-tower. Voir reports/enquete_simat.md §5.
Sortie : forcing-casr-merge.nc.  ENV : REF, SRC, OUT, VOL (def 1147 ; 0 = volume REF), RMIN/RMAX.
  python .runs/slso/build_casr_merge.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from meandre.data.forcing_correction import monthly_ratio_merge

REF = os.environ.get("REF", ".runs/slso/data/forcing.nc")                    # SIMAT/quebec.zarr ou PyGMET
SRC = os.environ.get("SRC", "D:/meandre-data/slso/forcing-casr-corr.nc")     # base : jour-local + de-crachine
OUT = os.environ.get("OUT", "D:/meandre-data/slso/forcing-casr-merge.nc")
VOL = float(os.environ.get("VOL", "1147.0"))                                 # mm/an bilan d'eau ; 0 = volume REF
RMIN = float(os.environ.get("RMIN", "0.5")); RMAX = float(os.environ.get("RMAX", "2.0"))

def load(path):
    ds = xr.open_dataset(path)
    t = pd.to_datetime(ds["time"].values).normalize()
    F = ds["forcing"].values
    vdim = "var" if "var" in ds else "variable"
    VARS = list(ds[vdim].values.astype(str))
    ds.close()
    return pd.DatetimeIndex(t), F, VARS

tr, Fr, _ = load(REF)
ts, Fs, VARS = load(SRC)
Pr = Fr[:, :, 0].astype(np.float64); Ps = Fs[:, :, 0].astype(np.float64)
print(f"REF {os.path.basename(REF)} {Pr.mean()*365.25:.0f} mm/an | SRC {os.path.basename(SRC)} {Ps.mean()*365.25:.0f} mm/an")

Pm, ratio, factor = monthly_ratio_merge(Ps, ts, Pr, tr, bounds=(RMIN, RMAX),
                                        target_vol_mm_yr=VOL if VOL > 0 else None)
print("ratio mensuel (mediane noeuds) :", " ".join(f"{m+1}:{np.median(ratio[m]):.2f}" for m in range(12)))
print(f"noeuds satures aux bornes : {(np.isclose(ratio, RMIN) | np.isclose(ratio, RMAX)).mean()*100:.1f}%")
print(f"volume final {Pm.mean()*365.25:.0f} mm/an (rescale global x{factor:.3f}) | jours pluvieux {(Pm>0.1).mean()*100:.0f}%")

Fs[:, :, 0] = Pm.astype(np.float32)
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), Fs.astype(np.float32))},
           coords={"time": ts.values, "node": np.arange(Fs.shape[1]), "var": VARS}).to_netcdf(OUT)
print(f"[ok] {OUT}")
