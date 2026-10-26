"""Forçage de DIAGNOSTIC : celui du modèle (CaSR brut) dont la pluie est remplacée par celle de GOENS,
la météo de l'ensemble Hydrotel, à la cellule la plus proche de chaque tronçon. Les autres variables
restent celles de CaSR. Sert à une passe à poids gelés ; ce n'est pas un forçage de production (le
prétraitement de la météo reste hors de meandre, décision d'Essi).

Jour GOENS : de 08 h à 08 h UTC, le décalage qui maximise la corrélation journalière avec CaSR.

    .venv/Scripts/python.exe experiments/plan_ml/forcage-pluie-goens.py slso
    .venv/Scripts/python.exe experiments/plan_ml/forcage-pluie-goens.py slso winter

Avec `winter`, seule la pluie de novembre à avril est remplacée (forcing-<reg>-goensPwinter.nc) :
sépare l'effet de l'excès hivernal de CaSR de celui de la pluie d'été.
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, ".")
from meandre.utils import paths as _p

GOENS = "D:/rqh/meteo/goens/GRILLE_PRECIP_QCMERI_GOENS00_MB0000.nc"


def main(reg, season=None):
    src = f"{_p.DATA_ROOT}/quebec/forcing-{reg}-casr-brut.nc"
    dst = f"{_p.DATA_ROOT}/quebec/forcing-{reg}-goensP{season or ''}.nc"
    f = xr.open_dataset(src)
    temps = pd.DatetimeIndex(f.time.values)
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    nodes = con.sql("select lon, lat from nodes order by node_idx").df()
    con.close()
    assert len(nodes) == f.sizes["node"]
    g = xr.open_dataset(GOENS)
    lon_g, lat_g = g.lon.values, g.lat.values
    ix = np.abs(lon_g[None, :] - nodes.lon.to_numpy()[:, None]).argmin(1)
    iy = np.abs(lat_g[None, :] - nodes.lat.to_numpy()[:, None]).argmin(1)
    cel = ix * len(lat_g) + iy
    P = np.full((len(temps), len(nodes)), np.nan, dtype=np.float32)
    pos = {d: k for k, d in enumerate(temps.normalize())}
    for m in pd.period_range(temps[0], temps[-1], freq="M"):
        b = g.pr.sel(time=slice(f"{m.start_time:%Y-%m-%d}T08:00", f"{(m.end_time + pd.Timedelta(days=1)):%Y-%m-%d}T07:00"))
        v = np.nan_to_num(b.values.reshape(-1, b.sizes["time"])[cel])
        jours = pd.DatetimeIndex(b.time.values - np.timedelta64(8, "h")).normalize()
        jour = pd.DataFrame(v.T, index=jours).groupby(level=0).sum()
        for d, row in jour.iterrows():
            k = pos.get(d)
            if k is not None:
                P[k] = row.to_numpy()
    manque = np.isnan(P).any(axis=1)
    if season == "winter":
        manque |= ~temps.month.isin([11, 12, 1, 2, 3, 4])
    print(f"{reg} : {len(nodes)} tronçons, {len(temps)} jours, {int(manque.sum())} jours sans GOENS (gardent CaSR)")
    forcing = f["forcing"].values.copy()
    ip = list(f["var"].values).index("P")
    casr_annuel = forcing[:, :, ip].mean() * 365.25
    forcing[~manque, :, ip] = P[~manque]
    print(f"pluie moyenne annuelle du territoire : CaSR {casr_annuel:.0f} mm, forçage de diagnostic {forcing[:, :, ip].mean() * 365.25:.0f} mm")
    out = f.copy()
    out["forcing"] = (f["forcing"].dims, forcing)
    out.to_netcdf(dst)
    print(f"écrit : {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
