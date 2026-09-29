"""La chute d'automne de l'indice foliaire MODIS suit-elle la part de feuillus et de cultures ?

Test spatial sans simulation. Pour chaque composite MODIS d'août et d'octobre (MOD15A2H,
500 m, 8 jours) de la période, on lit la fenêtre du sous-bassin, on rattache chaque pixel au
tronçon dont le centroïde est le plus proche, et on calcule par tronçon le rapport de l'indice
foliaire d'octobre à celui d'août. On le régresse sur les fractions brutes d'occupation de la
table `territorial` : feuillus, agricole, conifères. Si le rapport baisse avec la part de
feuillus et de cultures et pas avec celle de conifères, la fraction d'occupation d'Hydrotel
suffit comme indicateur de l'amplitude de la sénescence, et le seuil reste une constante.

    python .runs/quebec/chute_foliaire_par_troncon.py outv 040110 2011 2013
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import numpy as np
import pandas as pd

from meandre.data.open_data import _windowed_read
from meandre.utils import paths as _p


def pixels_lonlat(arr, transform, crs):
    from rasterio.warp import transform as _tr
    h, w = arr.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lon, lat = _tr(crs, "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist())
    return np.asarray(lon), np.asarray(lat)


def main(reg, station, a0, a1):
    import planetary_computer
    from pystac_client import Client
    from banc_sousbassin import extraire
    s = extraire(reg, station)
    c = s["node_coords"].cpu().numpy()
    idx = np.asarray(s["idx"])
    bbox = (float(c[:, 0].min()) - 0.03, float(c[:, 1].min()) - 0.03, float(c[:, 0].max()) + 0.03, float(c[:, 1].max()) + 0.03)
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    terr = con.execute("SELECT node_idx, f_forest_deciduous_raw, f_forest_conifer_raw, f_agriculture_raw, f_wetland_raw, area_km2_local FROM territorial").df().set_index("node_idx").loc[idx]
    con.close()
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["modis-15A2H-061"], bbox=bbox, datetime=f"{a0}-01-01/{a1}-12-31").items())
    _date = lambda x: pd.Timestamp(x.datetime or x.properties.get("start_datetime")).tz_localize(None) if pd.Timestamp(x.datetime or x.properties.get("start_datetime")).tzinfo is None else pd.Timestamp(x.datetime or x.properties.get("start_datetime")).tz_convert(None)
    # Août : composites des jours 209 à 233 ; octobre : 281 à 305. Un composite couvre 8 jours.
    fenetres = {"aout": (209, 233), "octobre": (281, 305)}
    somme = {k: np.zeros(len(idx)) for k in fenetres}
    compte = {k: np.zeros(len(idx)) for k in fenetres}
    n_pix = 0
    for it in items:
        d = _date(it)
        doy = d.dayofyear
        nom = next((k for k, (d0, d1) in fenetres.items() if d0 <= doy <= d1), None)
        if nom is None or "Lai_500m" not in it.assets:
            continue
        r = _windowed_read(it.assets["Lai_500m"].href, bbox)
        if r is None:
            continue
        arr, transform, crs, _ = r
        a = arr[0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
        a[a > 100] = np.nan
        a *= 0.1
        lon, lat = pixels_lonlat(a, transform, crs)
        ok = np.isfinite(a.ravel())
        if not ok.any():
            continue
        # Rattachement au centroïde de tronçon le plus proche, en degrés corrigés du cosinus.
        cl = np.cos(np.deg2rad(lat.mean()))
        dx = (lon[ok, None] - c[None, :, 0]) * cl
        dy = lat[ok, None] - c[None, :, 1]
        proche = np.argmin(dx * dx + dy * dy, axis=1)
        np.add.at(somme[nom], proche, a.ravel()[ok])
        np.add.at(compte[nom], proche, 1.0)
        n_pix = max(n_pix, int(ok.sum()))
    lai = {k: np.where(compte[k] > 0, somme[k] / np.maximum(compte[k], 1), np.nan) for k in fenetres}
    t = terr.copy()
    t["lai_aout"] = lai["aout"]
    t["lai_octobre"] = lai["octobre"]
    t["rapport"] = t.lai_octobre / t.lai_aout
    t["pixels"] = compte["aout"]
    t = t[(t.pixels >= 3) & np.isfinite(t.rapport)]
    t["f_chute"] = t.f_forest_deciduous_raw + t.f_agriculture_raw
    print(f"{reg} {station} : {len(t)} troncons avec au moins 3 pixels MODIS, {n_pix} pixels par composite, {a0}-{a1}")
    print(f"  indice foliaire moyen : aout {t.lai_aout.mean():.2f}, octobre {t.lai_octobre.mean():.2f}, rapport {t.rapport.mean():.2f}")
    for col in ("f_forest_deciduous_raw", "f_agriculture_raw", "f_chute", "f_forest_conifer_raw", "f_wetland_raw"):
        r = np.corrcoef(t[col], t.rapport)[0, 1]
        print(f"  correlation du rapport octobre/aout avec {col:24s} : {r:+.2f}  (fraction moyenne {t[col].mean():.2f}, etendue {t[col].min():.2f} a {t[col].max():.2f})")
    # Regression du rapport sur feuillus et agricole : rapport = a - b_feu f_feu - b_agri f_agri
    X = np.column_stack([np.ones(len(t)), t.f_forest_deciduous_raw, t.f_agriculture_raw, t.f_forest_conifer_raw])
    beta, *_ = np.linalg.lstsq(X, t.rapport.to_numpy(), rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((t.rapport - pred) ** 2) / np.sum((t.rapport - t.rapport.mean()) ** 2)
    print(f"  regression : rapport = {beta[0]:.2f} {beta[1]:+.2f} x feuillus {beta[2]:+.2f} x agricole {beta[3]:+.2f} x coniferes | R2 {r2:.2f}")
    print("  lecture : un coefficient negatif sur feuillus et agricole et nul sur coniferes valide la fraction d'occupation comme indicateur de l'amplitude.")
    f = f"{_p.DATA_ROOT}/derives/auxiliaires/chute-foliaire-{reg}-{station}-{a0}-{a1}.csv"
    t.to_csv(f)
    print(f"  ecrit : {f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
