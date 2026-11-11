"""Forçage dont le rayonnement net, la pression de vapeur et le vent sont reconstruits à partir de la
précipitation et des températures seules (FAO-56, données manquantes), à partir d'un forçage CaSR.

Sert à mesurer si meandre, entraîné sur les six variables de CaSR, tolère des scénarios qui ne
fournissent que P, Tmin et Tmax (ESPO-G6 d'Ouranos) :
- rayonnement solaire de Hargreaves, Rs = 0,16 √(Tmax − Tmin) Ra (FAO-56 éq. 50, site intérieur) ;
- rayonnement net FAO-56 (éq. 38 à 40) avec albédo 0,23 et e_a reconstruit ;
- pression de vapeur à saturation de Tmin (FAO-56 éq. 48, point de rosée pris égal à Tmin) ;
- vent : climatologie journalière du forçage d'origine (moyenne par tronçon et jour de l'année).

Usage : python reconstruct_forcing_fao.py <forçage d'origine .nc> <sortie .nc> <coordonnées .npz ou base .duckdb>
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

SIGMA_MJ = 4.903e-9


def es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def main():
    src, out, coords_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ds = xr.open_dataset(src).load()
    f = ds.forcing.values.copy()
    names = list(ds["var"].values)
    tmin, tmax = f[..., names.index("Tmin")], f[..., names.index("Tmax")]
    if coords_path.endswith(".duckdb"):
        from meandre.data.basin_cache import BasinCache
        lat = BasinCache(coords_path).load(device="cpu")["node_coords"].numpy()[:, 1]
    else:
        lat = np.load(coords_path)["coords"][:, 1]
    days = pd.DatetimeIndex(ds.time.values)
    J = days.dayofyear.to_numpy()[:, None]
    phi = np.deg2rad(lat)[None, :]
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    Rs = 0.16 * np.sqrt(np.clip(tmax - tmin, 0, None)) * Ra
    Rso = 0.75 * Ra
    ea = es(tmin)
    Rnl = SIGMA_MJ * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(Rs / np.maximum(Rso, 1e-6), 0, 1) - 0.35)
    Rn = np.clip(0.77 * Rs - Rnl, 0, None)
    u2 = f[..., names.index("u2")]
    clim = pd.DataFrame(u2).groupby(J[:, 0]).transform("mean").to_numpy()
    if "--calibrate" in sys.argv:
        # Facteur par tronçon et par mois qui rend à la reconstruction la moyenne du forçage d'origine ;
        # ajusté sur la période historique, il se réapplique tel quel à un scénario.
        months = days.month.to_numpy()
        for k, rec in (("R_n", Rn), ("e_a", ea)):
            orig = f[..., names.index(k)]
            for m in range(1, 13):
                s = months == m
                rec[s] *= orig[s].mean(axis=0) / np.maximum(rec[s].mean(axis=0), 1e-6)
    g = f.copy()
    g[..., names.index("R_n")] = Rn
    g[..., names.index("e_a")] = ea
    g[..., names.index("u2")] = clim
    ds["forcing"].values[:] = g.astype(np.float32)
    ds.to_netcdf(out)
    for k, a, b in (("R_n", f[..., names.index("R_n")], Rn), ("e_a", f[..., names.index("e_a")], ea)):
        print(f"{k} : moyenne {a.mean():.3f} -> {b.mean():.3f}, corrélation journalière {np.corrcoef(a.ravel()[::97], b.ravel()[::97])[0, 1]:.2f}")
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
