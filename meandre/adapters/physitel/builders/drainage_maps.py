"""Carte hierarchique de la classe de drainage, croisee par troncon.

Reproduit la construction d'une collegue d'Essi : en chaque point, la classe de drainage de
l'IRDA quand elle existe, sinon celle de l'inventaire ecoforestier, sinon les depots de
surface du SIGEOM regroupes par permeabilite. La regle s'applique cellule par cellule sur
une grille de 100 m, dans la projection de la plateforme, puis la valeur est moyennee par
unite hydrologique et par troncon, ponderee par la surface. Une cellule que rien ne couvre
reste VIDE : le troncon porte sa fraction couverte et la valeur n'est jamais comblee.

Echelle commune, rang de 1 a 7 : tres rapidement (excessif), rapidement, bien, moderement
bien, imparfaitement, mal, tres mal draine. C'est l'echelle de l'IRDA
(`irda_properties.DRAINAGE`), et la dizaine du code ecoforestier `cl_drai` plus un.

La cle des depots du SIGEOM est celle de la collegue, cinq classes de permeabilite validees
par un geologue du Quaternaire. Elle ne nomme pas tous les codes rencontres : les tills,
alluvions, sediments marins et organiques a suffixe sont rattaches a leur famille dans
`SIGEOM_EXTENSION`, a faire valider ; le roc et les codes restants restent vides.

    python -m meandre.adapters.physitel.builders.drainage_maps mont slso slno outv gasp sagu
"""
import os
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd


from meandre.adapters.physitel.physitel_loader import _parse_reach
from meandre.utils import paths as _paths

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SOURCES = f"{_paths.DATA_ROOT}/sources"
OUTPUT = f"{_paths.DATA_ROOT}/derives/auxiliaires"
ECOFOREST = f"{SOURCES}/ecoforestier"
SIGEOM = f"{SOURCES}/sigeom-quaternaire/sigeom.gpkg"
RESOLUTION = float(os.environ.get("DRAINAGE_RESOLUTION_M", "100"))
# Ordre de preseance des sources, du premier au dernier recours.
ORDER = os.environ.get("DRAINAGE_ORDRE", "irda,ecoforestier,sigeom").split(",")
# Suffixe du parquet de sortie, pour tenir plusieurs ordres cote a cote.
SUFFIX = os.environ.get("DRAINAGE_SUFFIXE", "")

# Cle de permeabilite des depots de la collegue d'Essi, validee par un geologue du
# Quaternaire (2026-09-23), placee sur l'echelle de drainage de 1 a 7 par pas egaux.
SIGEOM_RANK = {"tres_permeable": 1.0, "permeable": 2.5, "moyennement_permeable": 4.0, "peu_permeable": 5.5, "tres_peu_permeable": 7.0}
SIGEOM_GROUP = {
    "tres_permeable": ("Ce", "LGd", "MGd", "Go", "Gx", "Gxi", "Qf", "Ed"),
    "permeable": ("LGb", "MGb", "Gs", "Ap", "A", "Qa", "Tf", "L"),
    "moyennement_permeable": ("GxT", "T", "C"),
    "peu_permeable": ("Cg",),
    "tres_peu_permeable": ("MGa", "O", "LGa"),
}
# Codes absents de la cle, rattaches a la famille de leur racine ; a faire valider.
SIGEOM_EXTENSION = {
    "tres_permeable": ("Ld", "Md"),
    "permeable": ("At", "Ax", "Ac", "Ae", "G", "Ge", "Lb", "Mb"),
    "moyennement_permeable": ("Tm", "Tc", "Tr", "Trm", "Td", "Ts", "Tb", "To"),
    "tres_peu_permeable": ("Ma", "Om", "Ot"),
}
CODE_TO_GROUP = {c: g for g, codes in SIGEOM_GROUP.items() for c in codes}
CODE_TO_GROUP.update({c: g for g, codes in SIGEOM_EXTENSION.items() for c in codes if c not in CODE_TO_GROUP})


# Depots de surface de l'inventaire ecoforestier, places sur la meme echelle par la cle de la
# collegue transposee aux familles genetiques du code MFFP : 1 glaciaire (till), 2 fluvio-
# glaciaire, 3 fluviatile, 4 lacustre, 5 marin, 6 littoral, 7 organique, 8 pente et
# alteration, 9 eolien. Le roc (R, RC, RS) et l'anthropique restent vides ; les prefixes M et R
# (depot mince sur roc) sont retires pour lire la famille.
ECOFOREST_DEPOSIT_RANK = {
    "1": 4.0,
    "2A": 1.0, "2B": 1.0, "2": 1.0,
    "3A": 2.5, "3D": 1.0, "3": 2.5,
    "4A": 2.5, "4G": 7.0, "4P": 2.5, "4": 2.5,
    "5A": 7.0, "5G": 7.0, "5L": 5.5, "5S": 2.5, "5": 7.0,
    "6": 2.5,
    "7": 7.0,
    "8A": 4.0, "8C": 4.0, "8E": 1.0, "8F": 1.0, "8G": 5.5, "8P": 5.5, "8": 4.0,
    "9": 1.0,
}


def ecoforest_deposit_rank(codes):
    """Rang de permeabilite du depot ecoforestier ; vide pour le roc, l'anthropique et l'absent."""
    out = []
    for c in codes:
        c = str(c).strip().upper() if isinstance(c, str) else ""
        if len(c) >= 2 and c[0] in "MR" and c[1].isdigit():
            c = c[1:]
        r = ECOFOREST_DEPOSIT_RANK.get(c[:2]) if len(c) >= 2 else None
        if r is None and c[:1].isdigit():
            r = ECOFOREST_DEPOSIT_RANK.get(c[:1])
        out.append(np.nan if r is None else r)
    return np.array(out, dtype=float)


def ecoforest_rank(code):
    """Dizaine du code `cl_drai` plus un ; 16 (complexe) et les codes absents restent vides."""
    v = pd.to_numeric(code, errors="coerce")
    tens = np.floor(v / 10.0)
    ok = np.isfinite(v) & (v != 16) & (tens >= 0) & (tens <= 6)
    return np.where(ok, tens + 1.0, np.nan)


def rank_layers(layers):
    """Cellule par cellule, la premiere couche qui porte une valeur l'emporte.

    Rend la valeur retenue et le rang de la couche qui l'a donnee, 1 pour la premiere ;
    0 la ou aucune ne repond.
    """
    value = np.full(layers[0].shape, np.nan, dtype=np.float32)
    source = np.zeros(layers[0].shape, dtype=np.int8)
    for k, c in enumerate(layers, start=1):
        vide = ~np.isfinite(value)
        takes = vide & np.isfinite(c)
        value = np.where(takes, c, value)
        source = np.where(takes, k, source)
    return value, source


def rasterize_layer(gdf, column, transform, shape_value):
    """Rasterise une valeur flottante par polygone ; les cellules sans polygone valent nan."""
    from rasterio import features
    g = gdf[gdf[column].notna()]
    if g.empty:
        return np.full(shape_value, np.nan, dtype=np.float32)
    shapes = ((geom, float(val)) for geom, val in zip(g.geometry, g[column]))
    return features.rasterize(shapes, out_shape=shape_value, transform=transform, fill=np.nan, dtype=np.float32)


def irda_layer(crs, extent):
    import geopandas as gpd
    from meandre.adapters.physitel.builders.irda_properties import GPKG, component_properties, properties_polygons
    props = properties_polygons(component_properties())[["Code_polygone", "drainage", "couv_drainage"]]
    geo = gpd.read_file(GPKG, layer="Couverture_pedologique", columns=["Code_polygone"], engine="pyogrio", bbox=extent)
    poly = geo.merge(props, on="Code_polygone", how="inner").to_crs(crs)
    # Un polygone dont moins de la moitie des composantes portent un drainage ne le definit pas.
    poly.loc[poly.couv_drainage < 0.5, "drainage"] = np.nan
    return poly


def ecoforest_layer(crs, extent_4326, feuillets):
    """Polygones ecoforestiers de l'emprise, avec drainage et depot ranges ; la lecture des
    feuillets coute jusqu'a trente minutes par territoire, le resultat est mis en cache."""
    import geopandas as gpd
    cache = Path(f"{ECOFOREST}/cache")
    cache.mkdir(exist_ok=True)
    key = "_".join(f"{v:.2f}" for v in extent_4326)
    f_cache = cache / f"ecoforestier-{key}.parquet"
    if f_cache.exists():
        return gpd.read_parquet(f_cache).to_crs(crs)
    pieces = []
    for f in feuillets:
        z = Path(f"{ECOFOREST}/feuillets/{f}.zip")
        d = Path(f"{ECOFOREST}/feuillets/{f}")
        log = Path(f"{ECOFOREST}/feuillets/telechargement.log")
        complete = log.exists() and f"{f} code 200" in log.read_text()
        if not (z.exists() and complete):
            print(f"    feuillet {f} absent ou incomplet", flush=True)
            continue
        if not d.exists():
            import zipfile
            zipfile.ZipFile(z).extractall(d)
        g = next(d.rglob("*.gpkg"))
        lay = f"pee_maj_{f.lower()}"
        # Tout le feuillet est lu, puis coupe a l'emprise : pyogrio filtre par bbox dans le CRS de la couche.
        extent = gpd.GeoSeries.from_xy([extent_4326[0], extent_4326[2]], [extent_4326[1], extent_4326[3]], crs=4326).to_crs("EPSG:32198").total_bounds + np.array([-5e3, -5e3, 5e3, 5e3])
        pee = gpd.read_file(g, layer=lay, columns=["cl_drai", "dep_sur"], engine="pyogrio", bbox=tuple(extent))
        pee["drainage"] = ecoforest_rank(pee.cl_drai)
        pee["depot"] = ecoforest_deposit_rank(pee.dep_sur)
        pieces.append(pee[["drainage", "depot", "dep_sur", "geometry"]])
    if not pieces:
        return None
    eco = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs="EPSG:32198")
    eco["dep_sur"] = eco.dep_sur.astype(str)
    eco.to_parquet(f_cache)
    return eco.to_crs(crs)


def ecoforest_rasters(crs, extent_4326, feuillets, transform, shape_value, column_names=("drainage", "depot")):
    """Rasters ecoforestiers de l'emprise, construits FEUILLET PAR FEUILLET : un feuillet est lu,
    rasterise sur la grille du territoire, puis libere. Tenir tous les polygones d'un
    territoire en memoire coutait 12,6 Go pour 2,6 millions de polygones au Saguenay et
    montait a 51 Go avec les copies, ce qui a fait planter le poste (2026-09-24). Le
    resultat est mis en cache par emprise et resolution. `colonnes` nomme des colonnes
    calculables par polygone : drainage, depot, famille, mince.
    """
    import geopandas as gpd
    cache = Path(f"{ECOFOREST}/cache")
    cache.mkdir(exist_ok=True)
    key = "_".join(f"{v:.2f}" for v in extent_4326) + f"-{RESOLUTION:.0f}m"
    f_cache = cache / f"rasters-{key}.npz"
    if f_cache.exists():
        z = np.load(f_cache)
        if all(c in z.files for c in column_names):
            return {c: z[c] for c in column_names}
    out = {c: np.full(shape_value, np.nan, dtype=np.float32) for c in column_names}
    log = Path(f"{ECOFOREST}/feuillets/telechargement.log")
    for f in feuillets:
        z = Path(f"{ECOFOREST}/feuillets/{f}.zip")
        d = Path(f"{ECOFOREST}/feuillets/{f}")
        complete = log.exists() and f"{f} code 200" in log.read_text()
        if not (z.exists() and complete):
            print(f"    feuillet {f} absent ou incomplet", flush=True)
            continue
        if not d.exists():
            import zipfile
            zipfile.ZipFile(z).extractall(d)
        g = next(d.rglob("*.gpkg"))
        extent = gpd.GeoSeries.from_xy([extent_4326[0], extent_4326[2]], [extent_4326[1], extent_4326[3]], crs=4326).to_crs("EPSG:32198").total_bounds + np.array([-5e3, -5e3, 5e3, 5e3])
        pee = gpd.read_file(g, layer=f"pee_maj_{f.lower()}", columns=["cl_drai", "dep_sur"], engine="pyogrio", bbox=tuple(extent)).to_crs(crs)
        if "drainage" in column_names:
            pee["drainage"] = ecoforest_rank(pee.cl_drai)
        if "depot" in column_names:
            pee["depot"] = ecoforest_deposit_rank(pee.dep_sur)
        if "famille" in column_names or "mince" in column_names:
            from reconstruire_drainage import famille_ecoforestier
            pee["famille"], pee["mince"] = famille_ecoforestier(pee.dep_sur)
        for c in column_names:
            r = rasterize_layer(pee, c, transform, shape_value)
            takes = np.isfinite(r) & ~np.isfinite(out[c])
            out[c][takes] = r[takes]
        del pee
    np.savez_compressed(f_cache, **out)
    return out


def sigeom_layer(crs, extent_4326):
    import geopandas as gpd
    z = gpd.read_file(SIGEOM, layer="F10E15_ZONE_MORPH_SEDIM", columns=["CODE_DEPOT_MORP_SEDM"], engine="pyogrio", bbox=extent_4326)
    z["groupe"] = z.CODE_DEPOT_MORP_SEDM.map(CODE_TO_GROUP)
    area_value = z.geometry.to_crs(crs).area
    extended = z.CODE_DEPOT_MORP_SEDM.isin([c for v in SIGEOM_EXTENSION.values() for c in v])
    outside = z.groupe.isna()
    print(f"  SIGEOM : {100 * area_value[extended].sum() / area_value.sum():.0f} % de la surface par codes rattaches par extension, {100 * area_value[outside].sum() / area_value.sum():.0f} % hors cle ("
          + ", ".join(f"{c} {100 * a / area_value.sum():.0f} %" for c, a in area_value[outside].groupby(z.CODE_DEPOT_MORP_SEDM[outside]).sum().sort_values(ascending=False).head(6).items()) + ")", flush=True)
    z["drainage"] = z.groupe.map(SIGEOM_RANK)
    return z.to_crs(crs)


def sheets_of(extent_4326):
    import json
    from shapely.geometry import box, shape
    g = json.load(open(f"{ECOFOREST}/URL_250K_MAJ.geojson", encoding="utf-8"))
    b = box(*extent_4326)
    return [f["properties"]["Feuillet250K"] for f in g["features"] if shape(f["geometry"]).intersects(b)]


def build_territory(reg):
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
    extent_4326 = tuple(uh.to_crs(4326).total_bounds)
    ncol = int(np.ceil((x1 - x0) / RESOLUTION))
    nlig = int(np.ceil((y1 - y0) / RESOLUTION))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    shape_value = (nlig, ncol)
    ids = features.rasterize(((g, int(u)) for g, u in zip(uh.geometry, uh.uhrh)), out_shape=shape_value, transform=transform, fill=0, dtype=np.int32)
    print(f"{reg}: grille {nlig} x {ncol} a {RESOLUTION:.0f} m, {int((ids > 0).sum())} cellules dans les unites", flush=True)

    layers = {}
    layers["irda"] = rasterize_layer(irda_layer(crs, extent_4326), "drainage", transform, shape_value)
    eco = ecoforest_rasters(crs, extent_4326, sheets_of(extent_4326), transform, shape_value)
    layers["ecoforestier"] = eco["drainage"]
    eco_deposit = eco["depot"]
    sig = sigeom_layer(crs, extent_4326)
    layers["sigeom"] = rasterize_layer(sig, "drainage", transform, shape_value)
    for g in SIGEOM_GROUP:
        sig[f"est_{g}"] = (sig.groupe == g).astype(float)
    groups = {g: rasterize_layer(sig, f"est_{g}", transform, shape_value) for g in SIGEOM_GROUP}

    dans = ids > 0
    print("  couverture brute des unites : " + ", ".join(f"{k} {np.isfinite(v[dans]).mean():.2f}" for k, v in layers.items()), flush=True)

    value, source = rank_layers([layers[field_name] for field_name in ORDER])

    dans = ids > 0
    n_uhrh = np.bincount(ids[dans])
    def somme(mask):
        return np.bincount(ids[dans & mask], minlength=len(n_uhrh))
    finite = np.isfinite(value)
    num = np.bincount(ids[dans & finite], weights=value[dans & finite], minlength=len(n_uhrh))
    den = somme(finite)
    par_uhrh = pd.DataFrame({"n": n_uhrh, "num": num, "den": den, **{field_name: somme(source == k) for k, field_name in enumerate(ORDER, start=1)}})
    # Chaque source seule, hors hierarchie, et le depot ecoforestier : pour juger quel etage
    # porte l'information, pas seulement la carte fusionnee.
    only = dict(layers)
    only["depot_ecoforestier"] = eco_deposit
    for field_name, r in only.items():
        ok = np.isfinite(r)
        par_uhrh[f"num_{field_name}"] = np.bincount(ids[dans & ok], weights=r[dans & ok], minlength=len(n_uhrh))
        par_uhrh[f"den_{field_name}"] = somme(ok)
    for g, r in groups.items():
        par_uhrh[f"sigeom_{g}"] = somme(np.isfinite(r) & (r > 0.5))
    par_uhrh = par_uhrh[par_uhrh.n > 0]

    out = []
    for t in _parse_reach(Path(f"{proj}/physitel/troncon.trl")):
        uids = [u for u in t["uhrh_ids"] if u in par_uhrh.index]
        if not uids:
            continue
        s = par_uhrh.loc[uids].sum()
        row = {"region": reg, "troncon": int(t["id"]), "aire_m2": float(s.n) * RESOLUTION ** 2}
        row["drainage"] = float(s.num / s.den) if s.den > 0 else np.nan
        row["couv_drainage"] = float(s.den / s.n)
        for k in ("irda", "ecoforestier", "sigeom"):
            row[f"part_{k}"] = float(s[k] / s.n)
        for g in SIGEOM_GROUP:
            row[f"sigeom_{g}"] = float(s[f"sigeom_{g}"] / s.n)
        for field_name in only:
            row[f"drainage_{field_name}"] = float(s[f"num_{field_name}"] / s[f"den_{field_name}"]) if s[f"den_{field_name}"] > 0 else np.nan
            row[f"couv_drainage_{field_name}"] = float(s[f"den_{field_name}"] / s.n)
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | couverture mediane {d.couv_drainage.median():.2f} | part IRDA {d.part_irda.mean():.2f}, ecoforestier {d.part_ecoforestier.mean():.2f}, SIGEOM {d.part_sigeom.mean():.2f} | drainage median {d.drainage.median():.2f} | {time.time() - t0:.0f} s", flush=True)
    return d


def save(d):
    """Fusionne le territoire dans le parquet des le calcul fini : un arret en cours de
    route ne perd que le territoire en cours, pas les precedents."""
    f = f"{OUTPUT}/drainage-hierarchique{SUFFIX}-troncons.parquet"
    if os.path.exists(f):
        old = pd.read_parquet(f)
        d = pd.concat([old[~old.region.isin(d.region.unique())], d], ignore_index=True)
    d.to_parquet(f, index=False)
    return f, len(d), len(d.columns)


def main(regions):
    for r in regions:
        d = build_territory(r)
        if d is not None:
            f, n, k = save(d)
            print(f"  enregistre : {f} ({n} tronçons, {k} colonnes)", flush=True)
    print("croisement fait", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
