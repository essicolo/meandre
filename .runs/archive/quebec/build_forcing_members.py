"""Membres d'ensemble PyGMET -> forcages complets par membre (chaine probabiliste 1.0).

Decision d'Essi (2026-08-24) : le resultat probabiliste de la 1.0 doit etre probant,
donc l'incertitude de FORCAGE entre dans la chaine. Chaque membre PyGMET est une
realisation meteorologique plausible (champ d'erreur spatialement et temporellement
correle, conditionne aux stations). Ce script fabrique, pour chaque membre, un forcage
COMPLET au format du pilote : la base hybride partout, le membre substitue sur sa
fenetre (2021-2024 : l'annee d'etat + la tenue de cote). Le meme point de reprise 1.0
s'evalue ensuite une fois par membre (JOINT_FX_SUFFIX=-ensNNN, ETL_EPOCHS=0,
ETL_DUMP_Q), et les enveloppes se combinent hors ligne avec la tete quantile.

Seuls P, Tmin et Tmax viennent du membre (c'est ce que PyGMET simule : prcp, tmean,
trange) ; R_n, u2 et e_a restent ceux de la base -- l'incertitude d'energie n'est pas
portee par cet ensemble, et le dire est plus honnete que de la bruiter sans modele.

    .venv/Scripts/python.exe .runs/quebec/build_forcing_members.py outv
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _paths

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
ENS_DIR = f"{_paths.DATA_ROOT}/pygmet/{REG}/PyGMET_output/ensemble_outputs"
BASE = _paths.data_path("quebec", f"forcing-{REG}-hyb.nc")

h = BasinCache(_paths.data_path("quebec", f"{REG}.duckdb")).load(device="cpu")
nc_ = h["node_coords"].numpy()
lon_n, lat_n = nc_[:, 0], nc_[:, 1]

base = xr.open_dataset(BASE)
F0 = base["forcing"].values.copy()
times = pd.DatetimeIndex(base["time"].values)
base.close()

membres = sorted(glob.glob(f"{ENS_DIR}/*_Ensemble_*_*.nc"))
groupes = {}
for f in membres:
    num = f.rsplit("_", 1)[1].split(".")[0]
    groupes.setdefault(num, []).append(f)
print(f"[{REG}] {len(groupes)} membres, {len(membres)} fichiers")

for num, fichiers in sorted(groupes.items()):
    F = F0.copy()
    n_jours = 0
    for fp in sorted(fichiers):
        ds = xr.open_dataset(fp)
        tm = pd.DatetimeIndex(ds["time"].values)
        lat_g = ds["latitude"].values
        lon_g = ds["longitude"].values
        if lat_g.ndim == 2:
            lat_1d, lon_1d = lat_g[:, 0], lon_g[0, :]
        else:
            lat_1d, lon_1d = lat_g, lon_g
        # axes croissants pour l'interpolateur
        def prep(v):
            a = ds[v].values
            if lat_1d[0] > lat_1d[-1]:
                a = a[:, ::-1, :]
            return a
        la = lat_1d[::-1] if lat_1d[0] > lat_1d[-1] else lat_1d
        prcp, tmean, trange = prep("prcp"), prep("tmean"), prep("trange")
        # GARDE-FOU D'UNITE : les sorties d'ensemble PyGMET sont en espace PHYSIQUE
        # (prcp mm/j) ; si la mediane humide depasse 500 ou est negative, c'est un
        # espace transforme et on refuse plutot que d'ecrire un forcage absurde.
        w = prcp[prcp > 0.1]
        assert w.size and 0.2 < np.nanmedian(w) < 100, f"unites prcp suspectes ({np.nanmedian(w):.2f})"
        idx = times.get_indexer(tm)
        ok = idx >= 0
        pts = np.stack([np.clip(lat_n, la[0], la[-1]),
                        np.clip(lon_n, lon_1d[0], lon_1d[-1])], axis=1)
        for j, t_i in zip(np.flatnonzero(ok), idx[ok]):
            fp_p = RegularGridInterpolator((la, lon_1d), prcp[j], bounds_error=False, fill_value=None)
            fp_t = RegularGridInterpolator((la, lon_1d), tmean[j], bounds_error=False, fill_value=None)
            fp_r = RegularGridInterpolator((la, lon_1d), trange[j], bounds_error=False, fill_value=None)
            P = np.clip(np.nan_to_num(fp_p(pts), nan=0.0), 0.0, None)
            T = fp_t(pts); R = np.clip(fp_r(pts), 0.5, None)
            F[t_i, :, 0] = P
            F[t_i, :, 1] = T - R / 2.0
            F[t_i, :, 2] = T + R / 2.0
            n_jours += 1
        ds.close()
    out = _paths.data_path("quebec", f"forcing-{REG}-ens{num}.nc")
    xr.DataArray(F.astype(np.float32), dims=("time", "node", "var"),
                 coords={"time": times, "node": np.arange(F.shape[1]),
                         "var": ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"]},
                 name="forcing").to_netcdf(out)
    print(f"  membre {num} : {n_jours} jours substitues -> {os.path.basename(out)} | "
          f"P annuel fenetre {F[times.year >= 2021, :, 0].mean() * 365.25:.0f} mm "
          f"(base {F0[times.year >= 2021, :, 0].mean() * 365.25:.0f})")
print("fini")
