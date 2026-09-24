"""Reconstruction d'une carte de la classe de drainage de terrain, a partir de tout.

Constat qui motive le chantier (registre, 2026-09-23) : seule la classe de drainage etablie
au terrain par le profil predit la part souterraine du debit, l'IRDA a +33 % et les
Pedo-paysages du Canada a +28 % contre le temoin ; mais l'une et l'autre ne couvrent que le
sud agricole. Les dépôts du Quaternaire du SIGEOM, ranges par la cle de permeabilite d'une
collegue, predisent a +17 % et couvrent la foret ; l'inventaire ecoforestier ne predit rien.

La carte reconstruite apprend la classe de terrain la ou elle existe et la predit ailleurs.

    Cible : rang de drainage de 1 (tres rapidement) a 7 (tres mal draine), de l'IRDA par
    polygone, des Pedo-paysages par polygone (ponderes par le pourcentage des composantes)
    et des pedons de la base nationale (points), seule verite de terrain en foret.

    Covariables, disponibles partout dans les six territoires, sur la grille de 100 m des
    plateformes : depot ecoforestier (famille genetique, rang de permeabilite, depot mince
    sur roc), classe de drainage ecoforestiere, groupe de permeabilite du SIGEOM, textures et
    carbone organique SIIGSOL.

Trois etapes, chacune reprenable :

    reconstruire_drainage.py grille mont slso ...   construit et enregistre les grilles
    reconstruire_drainage.py ajuster                 ajuste par territoire tenu de cote, juge
    reconstruire_drainage.py troncons                predit partout et agrege par troncon

Le jugement se fait sur trois epreuves independantes de l'ajustement : le territoire tenu de
cote (cellules IRDA), les polygones des Pedo-paysages, et les pedons forestiers.
"""
import os
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

import croiser_drainage as cd
import croiser_slc as cs

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SORTIE = Path(f"{_paths.DATA_ROOT}/derives/reconstruction")
SIIGSOL = f"{_paths.DATA_ROOT}/sources/siigsol"
NPDB = f"{_paths.DATA_ROOT}/sources/npdb"
RESOLUTION = float(os.environ.get("DRAINAGE_RESOLUTION_M", "100"))
NPDB_DRAINAGE = {"very rapidly drained": 1.0, "rapidly drained": 2.0, "well drained": 3.0, "moderately well drained": 4.0, "imperfectly drained": 5.0, "poorly drained": 6.0, "very poorly drained": 7.0}
COVARIABLES = ["eco_drainage", "eco_depot", "eco_famille", "eco_mince", "sigeom_rang", "sable", "limon", "argile", "corg", "twi"]
TWI = f"{_paths.DATA_ROOT}/sources/humidite-lidar"


def famille_ecoforestier(codes):
    """Famille genetique du depot ecoforestier, 1 a 9, 10 pour le roc ; nan sinon."""
    out, mince = [], []
    for c in codes:
        c = str(c).strip().upper() if isinstance(c, str) else ""
        m = 1.0 if (len(c) >= 2 and c[0] in "MR" and c[1].isdigit()) or c in ("R", "RC", "RS") else 0.0
        if len(c) >= 2 and c[0] in "MR" and c[1].isdigit():
            c = c[1:]
        if c[:1].isdigit():
            out.append(float(c[0]))
        elif c[:1] == "R":
            out.append(10.0)
        else:
            out.append(np.nan)
        mince.append(m if c else np.nan)
    return np.array(out, dtype=float), np.array(mince, dtype=float)


def raster_siigsol(nom, crs, transform, forme):
    """SIIGSOL reprojete sur la grille du territoire, nan hors donnee."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.full(forme, np.nan, dtype=np.float32)
    with rasterio.open(f"{SIIGSOL}/{nom}_fr_siigsol.tif") as src:
        reproject(rasterio.band(src, 1), out, dst_transform=transform, dst_crs=crs, src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def raster_twi(crs, emprise_4326, transform, forme):
    """Indice d'humidite topographique LiDAR (MFFP, 1 m, 2 321 feuillets), lu a distance a
    100 m par feuillet puis reprojete sur la grille ; chaque feuillet lu est mis en cache."""
    import geopandas as gpd
    import rasterio
    from concurrent.futures import ThreadPoolExecutor
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import box
    index = gpd.read_file("zip://" + f"{TWI}/index-feuillets-shp.zip" + "!URL_twi.shp")
    zone = gpd.GeoSeries([box(*emprise_4326)], crs=4326).to_crs(index.crs).iloc[0]
    feuillets = index[index.intersects(zone)]
    cache = SORTIE / "cache-twi"
    cache.mkdir(parents=True, exist_ok=True)

    def un_feuillet(ligne):
        nom = ligne["feuillet"]
        f_cache = cache / f"{nom}.npz"
        if f_cache.exists():
            z = np.load(f_cache, allow_pickle=True)
            return z["data"], Affine(*z["transform"]), str(z["crs"])
        url = "/vsicurl/" + ligne["twi_url"].rstrip("/") + f"/TWI_{nom}.tif"
        for _ in range(3):
            try:
                with rasterio.open(url) as ds:
                    fac = max(1, int(round(RESOLUTION / ds.res[0])))
                    fo = (max(1, ds.height // fac), max(1, ds.width // fac))
                    data = ds.read(1, out_shape=fo, resampling=Resampling.average, masked=True).astype(np.float32).filled(np.nan)
                    t = ds.transform * Affine.scale(ds.width / fo[1], ds.height / fo[0])
                    np.savez_compressed(f_cache, data=data, transform=np.array(t[:6]), crs=str(ds.crs))
                    return data, t, str(ds.crs)
            except rasterio.errors.RasterioIOError:
                continue
        print(f"    TWI {nom} : lecture impossible", flush=True)
        return None, None, None

    out = np.full(forme, np.nan, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for data, t, c in pool.map(un_feuillet, [r for _, r in feuillets.iterrows()]):
            if data is None:
                continue
            morceau = np.full(forme, np.nan, dtype=np.float32)
            reproject(data, morceau, src_transform=t, src_crs=c, src_nodata=np.nan, dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
            prend = np.isfinite(morceau) & ~np.isfinite(out)
            out[prend] = morceau[prend]
    print(f"    TWI : {len(feuillets)} feuillets, {np.isfinite(out).mean():.2f} de la grille", flush=True)
    return out


def completer_twi(reg):
    """Ajoute l'humidite topographique a une grille deja enregistree."""
    from rasterio.transform import Affine
    f = SORTIE / f"grille-{reg}.npz"
    z = dict(np.load(f, allow_pickle=True))
    t = Affine.from_gdal(*z["transform"])
    forme = z["ids"].shape
    import geopandas as gpd
    from rasterio.transform import array_bounds
    b = array_bounds(forme[0], forme[1], t)
    emprise_4326 = tuple(gpd.GeoSeries.from_xy([b[0], b[2]], [b[1], b[3]], crs=str(z["crs"])).to_crs(4326).total_bounds)
    z["twi"] = raster_twi(str(z["crs"]), emprise_4326, t, forme)
    np.savez_compressed(f, **z)
    print(f"{reg}: grille completee", flush=True)


def raster_slc(crs, emprise_4326, transform, forme):
    import geopandas as gpd
    poly = gpd.read_file(f"{cs.SLC}/ca_all_slc_v3r2.shp", bbox=emprise_4326)
    cmp = gpd.read_file(f"{cs.SLC}/ca_all_slc_v3r2_cmp.dbf")
    sols = cs.proprietes_des_sols()
    cm = cmp[cmp.POLY_ID.isin(poly.POLY_ID)].join(sols[["drainage"]], on="SOIL_ID")
    w = cm.PERCENT.where(cm.drainage.notna(), 0.0).astype(float)
    val = (cm.drainage.fillna(0.0) * w).groupby(cm.POLY_ID).sum() / w.groupby(cm.POLY_ID).sum().replace(0, np.nan)
    part = w.groupby(cm.POLY_ID).sum() / cm.PERCENT.astype(float).groupby(cm.POLY_ID).sum()
    poly = poly.join(pd.DataFrame({"drainage": val, "part": part}), on="POLY_ID").to_crs(crs)
    # Un polygone dont moins de la moitie des composantes portent un drainage ne le definit pas.
    poly.loc[poly.part.fillna(0) < 0.5, "drainage"] = np.nan
    return cd.rasteriser(poly, "drainage", transform, forme)


def pedons(crs):
    import geopandas as gpd
    info = pd.read_csv(f"{NPDB}/Info.csv", encoding="latin-1")
    info.columns = [c.lstrip("﻿") for c in info.columns]
    prof = pd.read_csv(f"{NPDB}/Profiles.csv", encoding="latin-1")
    prof.columns = [c.lstrip("﻿") for c in prof.columns]
    d = info.merge(prof[["PEDON_ID", "DRAINAGE"]], on="PEDON_ID", how="inner")
    d["drainage"] = d.DRAINAGE.str.strip().str.lower().map(NPDB_DRAINAGE)
    d = d[d.drainage.notna() & d.DD_LAT_N.notna()]
    return gpd.GeoDataFrame(d[["PEDON_ID", "drainage", "CAL_YEAR", "LANDUSE"]], geometry=gpd.points_from_xy(d.DD_LONG_N, d.DD_LAT_N), crs=4326).to_crs(crs)


def grille(reg):
    """Construit et enregistre la grille d'un territoire : cibles, covariables, unites."""
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    t0 = time.time()
    proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
    uh = gpd.read_file(f"{proj}/physitel/uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    crs = uh.crs
    x0, y0, x1, y1 = uh.total_bounds
    emprise_4326 = tuple(uh.to_crs(4326).total_bounds)
    ncol, nlig = int(np.ceil((x1 - x0) / RESOLUTION)), int(np.ceil((y1 - y0) / RESOLUTION))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    forme = (nlig, ncol)
    ids = features.rasterize(((g, int(u)) for g, u in zip(uh.geometry, uh.uhrh)), out_shape=forme, transform=transform, fill=0, dtype=np.int32)
    couches = {"ids": ids}
    couches["irda"] = cd.rasteriser(cd.couche_irda(crs, emprise_4326), "drainage", transform, forme)
    couches["slc"] = raster_slc(crs, emprise_4326, transform, forme)
    eco = cd.rasters_ecoforestier(crs, emprise_4326, cd.feuillets_de(emprise_4326), transform, forme, colonnes=("drainage", "depot", "famille", "mince"))
    for k in ("drainage", "depot", "famille", "mince"):
        couches[f"eco_{k}"] = eco[k]
    sig = cd.couche_sigeom(crs, emprise_4326)
    couches["sigeom_rang"] = cd.rasteriser(sig, "drainage", transform, forme)
    for nom in ("sable", "limon", "argile", "corg"):
        couches[nom] = raster_siigsol(nom, crs, transform, forme)
    couches["twi"] = raster_twi(crs, emprise_4326, transform, forme)
    # Pedons : indices de cellule et rang, pour l'epreuve forestiere.
    pe = pedons(crs)
    pe = pe[pe.geometry.within(gpd.GeoSeries([uh.union_all()], crs=crs).iloc[0])]
    rows = ((y1 - pe.geometry.y) // RESOLUTION).astype(int).to_numpy()
    cols = ((pe.geometry.x - x0) // RESOLUTION).astype(int).to_numpy()
    SORTIE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SORTIE / f"grille-{reg}.npz", transform=np.array(transform.to_gdal()), crs=str(crs), pedon_lig=rows, pedon_col=cols, pedon_rang=pe.drainage.to_numpy(), pedon_id=pe.PEDON_ID.to_numpy(), **couches)
    dans = ids > 0
    print(f"{reg}: grille {nlig} x {ncol}, {int(dans.sum())} cellules | couverture des cellules : " + ", ".join(f"{k} {np.isfinite(v[dans]).mean():.2f}" for k, v in couches.items() if k != "ids") + f" | {len(pe)} pedons | {time.time() - t0:.0f} s", flush=True)


def charger(regions):
    """Table cellule par cellule, echantillonnee, avec territoire, cibles et covariables."""
    tables = []
    for reg in regions:
        f = SORTIE / f"grille-{reg}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        dans = z["ids"] > 0
        n = int(dans.sum())
        # Un echantillon regulier de cellules par territoire, pour tenir en memoire.
        pas = max(1, n // int(os.environ.get("RECONSTRUCTION_CELLULES", "400000")))
        idx = np.flatnonzero(dans.ravel())[::pas]
        t = pd.DataFrame({k: z[k].ravel()[idx] for k in ["irda", "slc"] + COVARIABLES})
        t["ids"] = z["ids"].ravel()[idx]
        t["region"] = reg
        t["cellule"] = idx
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def modele():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, categorical_features=[COVARIABLES.index("eco_famille")], random_state=0)


def ajuster(regions):
    """Ajuste sur les cellules IRDA, juge par territoire tenu de cote, sur les Pedo-paysages
    et sur les pedons ; puis ajuste sur tout et enregistre le modele."""
    import joblib
    t = charger(regions)
    app = t[t.irda.notna()]
    print(f"{len(t)} cellules echantillonnees, {len(app)} avec une cible IRDA : " + ", ".join(f"{r} {int(n)}" for r, n in app.region.value_counts().items()), flush=True)
    X = app[COVARIABLES].to_numpy(dtype=float)
    y = app.irda.to_numpy()
    print("\nTerritoire tenu de cote, cellules IRDA : erreur absolue moyenne du modele contre celle du temoin (moyenne des autres territoires)")
    for reg in sorted(app.region.unique()):
        te, ap = app.region == reg, app.region != reg
        if te.sum() < 1000 or ap.sum() < 1000:
            continue
        m = modele().fit(X[ap], y[ap])
        pred = m.predict(X[te])
        temoin = np.full(te.sum(), y[ap].mean())
        r = np.corrcoef(pred, y[te])[0, 1]
        print(f"  {reg}: {int(te.sum())} cellules | modele {np.abs(pred - y[te]).mean():.3f} contre temoin {np.abs(temoin - y[te]).mean():.3f}, soit {100 * (1 - np.abs(pred - y[te]).mean() / np.abs(temoin - y[te]).mean()):+.0f} % | correlation {r:.2f}", flush=True)
    m = modele().fit(X, y)
    # Epreuve 2 : les Pedo-paysages, la ou l'IRDA n'existe pas.
    slc = t[t.slc.notna() & t.irda.isna()]
    if len(slc):
        pred = m.predict(slc[COVARIABLES].to_numpy(dtype=float))
        print(f"\nPedo-paysages hors IRDA, {len(slc)} cellules : erreur {np.abs(pred - slc.slc).mean():.3f} contre temoin {np.abs(y.mean() - slc.slc).mean():.3f}, correlation {np.corrcoef(pred, slc.slc)[0, 1]:.2f}")
    # Epreuve 3 : les pedons, cellule par cellule.
    lignes = []
    for reg in regions:
        f = SORTIE / f"grille-{reg}.npz"
        if not f.exists():
            continue
        # Chaque acces a une cle d'un npz decompresse la grille entiere : on charge une fois.
        z = np.load(f, allow_pickle=True)
        g = {k: z[k] for k in ["ids", "irda", "pedon_rang", "pedon_lig", "pedon_col"] + COVARIABLES}
        for i in range(len(g["pedon_rang"])):
            li, co = int(g["pedon_lig"][i]), int(g["pedon_col"][i])
            if not (0 <= li < g["ids"].shape[0] and 0 <= co < g["ids"].shape[1]):
                continue
            lignes.append({"region": reg, "rang": float(g["pedon_rang"][i]), "irda": float(g["irda"][li, co]), **{k: float(g[k][li, co]) for k in COVARIABLES}})
    pe = pd.DataFrame(lignes)
    if len(pe):
        pred = m.predict(pe[COVARIABLES].to_numpy(dtype=float))
        hors = pe.irda.isna()
        for nom, masque in (("tous", np.ones(len(pe), dtype=bool)), ("hors IRDA, donc foret", hors.to_numpy())):
            if masque.sum() >= 5:
                from scipy.stats import spearmanr
                print(f"Pedons {nom}, {int(masque.sum())} : erreur {np.abs(pred[masque] - pe.rang[masque]).mean():.2f} contre temoin {np.abs(y.mean() - pe.rang[masque]).mean():.2f}, Spearman {spearmanr(pred[masque], pe.rang[masque]).statistic:.2f}")
    joblib.dump(m, SORTIE / "modele-drainage.joblib")
    print(f"\nmodele enregistre : {SORTIE / 'modele-drainage.joblib'}")


def troncons(regions):
    """Predit la classe partout, puis agrege par unite hydrologique et par troncon."""
    import joblib
    m = joblib.load(SORTIE / "modele-drainage.joblib")
    out = []
    for reg in regions:
        f = SORTIE / f"grille-{reg}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        ids = z["ids"]
        dans = ids > 0
        X = np.stack([z[k][dans] for k in COVARIABLES], axis=1).astype(float)
        # Une cellule sans aucune covariable n'est pas predite.
        ok = np.isfinite(X).any(axis=1)
        pred = np.full(int(dans.sum()), np.nan)
        pred[ok] = m.predict(X[ok])
        # La ou la classe de terrain existe, elle prime sur la prediction.
        terrain = z["irda"][dans]
        valeur = np.where(np.isfinite(terrain), terrain, pred)
        n_uhrh = np.bincount(ids[dans])
        fini = np.isfinite(valeur)
        num = np.bincount(ids[dans][fini], weights=valeur[fini], minlength=len(n_uhrh))
        den = np.bincount(ids[dans][fini], minlength=len(n_uhrh))
        ter = np.bincount(ids[dans][np.isfinite(terrain)], minlength=len(n_uhrh))
        proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
        for t in _parse_troncon(Path(f"{proj}/physitel/troncon.trl")):
            uids = [u for u in t["uhrh_ids"] if u < len(n_uhrh) and n_uhrh[u] > 0]
            if not uids:
                continue
            n, s, d, tt = n_uhrh[uids].sum(), num[uids].sum(), den[uids].sum(), ter[uids].sum()
            out.append({"region": reg, "troncon": int(t["id"]), "aire_m2": float(n) * RESOLUTION ** 2, "drainage": float(s / d) if d > 0 else np.nan, "couv_drainage": float(d / n), "part_terrain": float(tt / n)})
        print(f"{reg}: predit", flush=True)
    d = pd.DataFrame(out)
    f = f"{_paths.DATA_ROOT}/derives/auxiliaires/drainage-reconstruit-troncons.parquet"
    d.to_parquet(f, index=False)
    print(f"{f} : {len(d)} tronçons | couverture mediane {d.couv_drainage.median():.2f} | part de terrain moyenne {d.part_terrain.mean():.2f}")


if __name__ == "__main__":
    etape, regs = sys.argv[1], [a.lower() for a in sys.argv[2:]] or ["mont", "slso", "slno", "outv", "gasp", "sagu"]
    if etape == "grille":
        for r in regs:
            grille(r)
    elif etape == "twi":
        for r in regs:
            completer_twi(r)
    elif etape == "ajuster":
        ajuster(regs)
    elif etape == "troncons":
        troncons(regs)
