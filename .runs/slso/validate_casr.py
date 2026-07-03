"""Validation propre du forçage CaSR : charge les deux caches (quebec.zarr-derived
forcing.nc vs forcing-casr.nc), confirme qu'ils diffèrent physiquement, puis lance
le MÊME modèle/intervention sur les deux et compare le débit. But : prouver que le
forçage CaSR circule réellement dans le pipeline (et non un npz périmé).
  python .runs/slso/validate_casr.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CKPT = ".runs/slso/checkpoints/best-physitel-hydrotel-multiobj.pt"
DB = ".runs/slso/data/slso.duckdb"
SPIN0, TEST1 = "2020-01-01", "2024-12-31"
FORCS = {"qb": ".runs/slso/data/forcing.nc", "casr": ".runs/slso/data/forcing-casr.nc"}
dev = "cpu"

# 1) Les deux forçages diffèrent-ils ?
for k, f in FORCS.items():
    ds = xr.open_dataset(f); t = pd.to_datetime(ds["time"].values)
    ndup = int(t.duplicated().sum())
    print(f"[{k}] {f}  T={ds.sizes['time']}  doublons_dates={ndup}  "
          f"{t.min().date()}..{t.max().date()}")
    ds.close()

cache = BasinCache(DB); h = cache.load(device=dev); n = h["n_nodes"]
ck = torch.load(CKPT, map_location=dev, weights_only=False)
kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False
m = HydroModel(**kw).to(dev); m.load_state_dict(ck["state_dict"], strict=False); m.eval()

def run(forc):
    ds = xr.open_dataset(forc); times = pd.to_datetime(ds["time"].values).normalize()
    w0 = int(np.searchsorted(times, np.datetime64(SPIN0)))
    w1 = int(np.searchsorted(times, np.datetime64(TEST1))) + 1
    ff = ds["forcing"].values[w0:w1].astype(np.float32); ds.close()
    win = times[w0:w1]
    fc = torch.from_numpy(ff)
    doy = torch.tensor(win.dayofyear.values, dtype=torch.long)
    wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=dev)
    # ksat0.3 (recette) appliquée pareil aux deux
    orig = m.spatial_encoder.forward
    def patched(*a, **k):
        sp = orig(*a, **k)
        for nm in ("K_sat_1", "K_sat_2", "K_sat_3"):
            setattr(sp, nm, getattr(sp, nm) * 0.3)
        return sp
    m.spatial_encoder.forward = patched
    with torch.no_grad():
        Q, _ = m.simulate(forcing=fc, initial_state=HydroState.default_warm(n, device=dev),
                          graph=h["graph"], node_coords=h["node_coords"],
                          territorial=h["territorial"], withdrawals=wd, day_of_year=doy)
    m.spatial_encoder.forward = orig
    # moyenne du forçage P sur la fenêtre, pour preuve de différence
    p_mean = float(ff[..., 0].mean())
    return Q.cpu().numpy(), p_mean, win

Qqb, pqb, win = run(FORCS["qb"])
Qcasr, pcasr, _ = run(FORCS["casr"])
print(f"\nP_moy fenêtre  qb={pqb:.3f}mm/j  casr={pcasr:.3f}mm/j  (diff={abs(pqb-pcasr):.3f})")
print(f"Q_moy          qb={np.nanmean(Qqb):.3f}  casr={np.nanmean(Qcasr):.3f}")
d = np.abs(Qqb - Qcasr)
print(f"|Q_qb - Q_casr|  max={np.nanmax(d):.4f}  moy={np.nanmean(d):.4f}  "
      f"identiques={bool(np.allclose(Qqb, Qcasr))}")
np.savez_compressed(".runs/slso/results/eval_qb.npz", Q=Qqb.astype(np.float32),
                    dates=win.values.astype("datetime64[D]").astype(str))
np.savez_compressed(".runs/slso/results/eval_casr.npz", Q=Qcasr.astype(np.float32),
                    dates=win.values.astype("datetime64[D]").astype(str))
print("[ok] eval_qb.npz / eval_casr.npz écrits")
