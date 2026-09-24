"""Carte hierarchique de la classe de drainage, croisee par troncon.

Reproduit la construction d'une collegue d'Essi : en chaque point, la classe de drainage de
l'IRDA quand elle existe, sinon celle de l'inventaire ecoforestier, sinon les depots de
surface du SIGEOM regroupes par permeabilite. La regle s'applique cellule par cellule sur
une grille de 100 m, dans la projection de la plateforme, puis la valeur est moyennee par
unite hydrologique et par troncon, ponderee par la surface. Une cellule que rien ne couvre
reste VIDE : le troncon porte sa fraction couverte et la valeur n'est jamais comblee.

Echelle commune, rang de 1 a 7 : tres rapidement (excessif), rapidement, bien, moderement
bien, imparfaitement, mal, tres mal draine. C'est l'echelle de l'IRDA
(`irda_proprietes.DRAINAGE`), et la dizaine du code ecoforestier `cl_drai` plus un.

La cle des depots du SIGEOM est celle de la collegue, cinq classes de permeabilite validees
par un geologue du Quaternaire. Elle ne nomme pas tous les codes rencontres : les tills,
alluvions, sediments marins et organiques a suffixe sont rattaches a leur famille dans
`SIGEOM_EXTENSION`, a faire valider ; le roc et les codes restants restent vides.

    .venv/Scripts/python.exe .runs/quebec/croiser_drainage.py mont slso slno outv gasp sagu
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

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SOURCES = f"{_paths.DATA_ROOT}/sources"
SORTIE = f"{_paths.DATA_ROOT}/derives/auxiliaires"
ECOFOR = f"{SOURCES}/ecoforestier"
SIGEOM = f"{SOURCES}/sigeom-quaternaire/sigeom.gpkg"
RESOLUTION = float(os.environ.get("DRAINAGE_RESOLUTION_M", "100"))
# Ordre de preseance des sources, du premier au dernier recours.
ORDRE = os.environ.get("DRAINAGE_ORDRE", "irda,ecoforestier,sigeom").split(",")
# Suffixe du parquet de sortie, pour tenir plusieurs ordres cote a cote.
SUFFIXE = os.environ.get("DRAINAGE_SUFFIXE", "")

# Cle de permeabilite des depots de la collegue d'Essi, validee par un geologue du
# Quaternaire (2026-09-23), placee sur l'echelle de drainage de 1 a 7 par pas egaux.
SIGEOM_RANG = {"tres_permeable": 1.0, "permeable": 2.5, "moyennement_permeable": 4.0, "peu_permeable": 5.5, "tres_peu_permeable": 7.0}
SIGEOM_GROUPE = {
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
CODE_VERS_GROUPE = {c: g for g, codes in SIGEOM_GROUPE.items() for c in codes}
CODE_VERS_GROUPE.update({c: g for g, codes in SIGEOM_EXTENSION.items() for c in codes if c not in CODE_VERS_GROUPE})


# Depots de surface de l'inventaire ecoforestier, places sur la meme echelle par la cle de la
# collegue transposee aux familles genetiques du code MFFP : 1 glaciaire (till), 2 fluvio-
# glaciaire, 3 fluviatile, 4 lacustre, 5 marin, 6 littoral, 7 organique, 8 pente et
# alteration, 9 eolien. Le roc (R, RC, RS) et l'anthropique restent vides ; les prefixes M et R
# (depot mince sur roc) sont retires pour lire la famille.
DEPOT_ECOFOR_RANG = {
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


def rang_depot_ecoforestier(codes):
    """Rang de permeabilite du depot ecoforestier ; vide pour le roc, l'anthropique et l'absent."""
    out = []
    for c in codes:
        c = str(c).strip().upper() if isinstance(c, str) else ""
        if len(c) >= 2 and c[0] in "MR" and c[1].isdigit():
            c = c[1:]
        r = DEPOT_ECOFOR_RANG.get(c[:2]) if len(c) >= 2 else None
        if r is None and c[:1].isdigit():
            r = DEPOT_ECOFOR_RANG.get(c[:1])
        out.append(np.nan if r is None else r)
    return np.array(out, dtype=float)


def rang_ecoforestier(code):
    """Dizaine du code `cl_drai` plus un ; 16 (complexe) et les codes absents restent vides."""
    v = pd.to_numeric(code, errors="coerce")
    dizaine = np.floor(v / 10.0)
    ok = np.isfinite(v) & (v != 16) & (dizaine >= 0) & (dizaine <= 6)
    return np.where(ok, dizaine + 1.0, np.nan)


def hierarchiser(couches):
    """Cellule par cellule, la premiere couche qui porte une valeur l'emporte.

    Rend la valeur retenue et le rang de la couche qui l'a donnee, 1 pour la premiere ;
    0 la ou aucune ne repond.
    """
    valeur = np.full(couches[0].shape, np.nan, dtype=np.float32)
    source = np.zeros(couches[0].shape, dtype=np.int8)
    for k, c in enumerate(couches, start=1):
        vide = ~np.isfinite(valeur)
        prend = vide & np.isfinite(c)
        valeur = np.where(prend, c, valeur)
        source = np.where(prend, k, source)
    return valeur, source


def rasteriser(gdf, colonne, transform, forme):
    """Rasterise une valeur flottante par polygone ; les cellules sans polygone valent nan."""
    from rasterio import features
    g = gdf[gdf[colonne].notna()]
    if g.empty:
        return np.full(forme, np.nan, dtype=np.float32)
    formes = ((geom, float(val)) for geom, val in zip(g.geometry, g[colonne]))
    return features.rasterize(formes, out_shape=forme, transform=transform, fill=np.nan, dtype=np.float32)


def couche_irda(crs, emprise):
    import geopandas as gpd
    from irda_proprietes import GPKG, proprietes_composantes, proprietes_polygones
    props = proprietes_polygones(proprietes_composantes())[["Code_polygone", "drainage", "couv_drainage"]]
    geo = gpd.read_file(GPKG, layer="Couverture_pedologique", columns=["Code_polygone"], engine="pyogrio", bbox=emprise)
    poly = geo.merge(props, on="Code_polygone", how="inner").to_crs(crs)
    # Un polygone dont moins de la moitie des composantes portent un drainage ne le definit pas.
    poly.loc[poly.couv_drainage < 0.5, "drainage"] = np.nan
    return poly


def couche_ecoforestier(crs, emprise_4326, feuillets):
    """Polygones ecoforestiers de l'emprise, avec drainage et depot ranges ; la lecture des
    feuillets coute jusqu'a trente minutes par territoire, le resultat est mis en cache."""
    import geopandas as gpd
    cache = Path(f"{ECOFOR}/cache")
    cache.mkdir(exist_ok=True)
    cle = "_".join(f"{v:.2f}" for v in emprise_4326)
    f_cache = cache / f"ecoforestier-{cle}.parquet"
    if f_cache.exists():
        return gpd.read_parquet(f_cache).to_crs(crs)
    morceaux = []
    for f in feuillets:
        z = Path(f"{ECOFOR}/feuillets/{f}.zip")
        d = Path(f"{ECOFOR}/feuillets/{f}")
        journal = Path(f"{ECOFOR}/feuillets/telechargement.log")
        complet = journal.exists() and f"{f} code 200" in journal.read_text()
        if not (z.exists() and complet):
            print(f"    feuillet {f} absent ou incomplet", flush=True)
            continue
        if not d.exists():
            import zipfile
            zipfile.ZipFile(z).extractall(d)
        g = next(d.rglob("*.gpkg"))
        lay = f"pee_maj_{f.lower()}"
        # Tout le feuillet est lu, puis coupe a l'emprise : pyogrio filtre par bbox dans le CRS de la couche.
        emprise = gpd.GeoSeries.from_xy([emprise_4326[0], emprise_4326[2]], [emprise_4326[1], emprise_4326[3]], crs=4326).to_crs("EPSG:32198").total_bounds + np.array([-5e3, -5e3, 5e3, 5e3])
        pee = gpd.read_file(g, layer=lay, columns=["cl_drai", "dep_sur"], engine="pyogrio", bbox=tuple(emprise))
        pee["drainage"] = rang_ecoforestier(pee.cl_drai)
        pee["depot"] = rang_depot_ecoforestier(pee.dep_sur)
        morceaux.append(pee[["drainage", "depot", "dep_sur", "geometry"]])
    if not morceaux:
        return None
    eco = gpd.GeoDataFrame(pd.concat(morceaux, ignore_index=True), crs="EPSG:32198")
    eco["dep_sur"] = eco.dep_sur.astype(str)
    eco.to_parquet(f_cache)
    return eco.to_crs(crs)


def couche_sigeom(crs, emprise_4326):
    import geopandas as gpd
    z = gpd.read_file(SIGEOM, layer="F10E15_ZONE_MORPH_SEDIM", columns=["CODE_DEPOT_MORP_SEDM"], engine="pyogrio", bbox=emprise_4326)
    z["groupe"] = z.CODE_DEPOT_MORP_SEDM.map(CODE_VERS_GROUPE)
    aire = z.geometry.to_crs(crs).area
    etendu = z.CODE_DEPOT_MORP_SEDM.isin([c for v in SIGEOM_EXTENSION.values() for c in v])
    hors = z.groupe.isna()
    print(f"  SIGEOM : {100 * aire[etendu].sum() / aire.sum():.0f} % de la surface par codes rattaches par extension, {100 * aire[hors].sum() / aire.sum():.0f} % hors cle ("
          + ", ".join(f"{c} {100 * a / aire.sum():.0f} %" for c, a in aire[hors].groupby(z.CODE_DEPOT_MORP_SEDM[hors]).sum().sort_values(ascending=False).head(6).items()) + ")", flush=True)
    z["drainage"] = z.groupe.map(SIGEOM_RANG)
    return z.to_crs(crs)


def feuillets_de(emprise_4326):
    import json
    from shapely.geometry import box, shape
    g = json.load(open(f"{ECOFOR}/URL_250K_MAJ.geojson", encoding="utf-8"))
    b = box(*emprise_4326)
    return [f["properties"]["Feuillet250K"] for f in g["features"] if shape(f["geometry"]).intersects(b)]


def une_region(reg):
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
    ncol = int(np.ceil((x1 - x0) / RESOLUTION))
    nlig = int(np.ceil((y1 - y0) / RESOLUTION))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    forme = (nlig, ncol)
    ids = features.rasterize(((g, int(u)) for g, u in zip(uh.geometry, uh.uhrh)), out_shape=forme, transform=transform, fill=0, dtype=np.int32)
    print(f"{reg}: grille {nlig} x {ncol} a {RESOLUTION:.0f} m, {int((ids > 0).sum())} cellules dans les unites", flush=True)

    couches = {}
    couches["irda"] = rasteriser(couche_irda(crs, emprise_4326), "drainage", transform, forme)
    eco = couche_ecoforestier(crs, emprise_4326, feuillets_de(emprise_4326))
    couches["ecoforestier"] = rasteriser(eco, "drainage", transform, forme) if eco is not None else np.full(forme, np.nan, dtype=np.float32)
    depot_eco = rasteriser(eco, "depot", transform, forme) if eco is not None else np.full(forme, np.nan, dtype=np.float32)
    sig = couche_sigeom(crs, emprise_4326)
    couches["sigeom"] = rasteriser(sig, "drainage", transform, forme)
    for g in SIGEOM_GROUPE:
        sig[f"est_{g}"] = (sig.groupe == g).astype(float)
    groupes = {g: rasteriser(sig, f"est_{g}", transform, forme) for g in SIGEOM_GROUPE}

    dans = ids > 0
    print("  couverture brute des unites : " + ", ".join(f"{k} {np.isfinite(v[dans]).mean():.2f}" for k, v in couches.items()), flush=True)

    valeur, source = hierarchiser([couches[nom] for nom in ORDRE])

    dans = ids > 0
    n_uhrh = np.bincount(ids[dans])
    def somme(masque):
        return np.bincount(ids[dans & masque], minlength=len(n_uhrh))
    fini = np.isfinite(valeur)
    num = np.bincount(ids[dans & fini], weights=valeur[dans & fini], minlength=len(n_uhrh))
    den = somme(fini)
    par_uhrh = pd.DataFrame({"n": n_uhrh, "num": num, "den": den, **{nom: somme(source == k) for k, nom in enumerate(ORDRE, start=1)}})
    # Chaque source seule, hors hierarchie, et le depot ecoforestier : pour juger quel etage
    # porte l'information, pas seulement la carte fusionnee.
    seules = dict(couches)
    seules["depot_ecoforestier"] = depot_eco
    for nom, r in seules.items():
        ok = np.isfinite(r)
        par_uhrh[f"num_{nom}"] = np.bincount(ids[dans & ok], weights=r[dans & ok], minlength=len(n_uhrh))
        par_uhrh[f"den_{nom}"] = somme(ok)
    for g, r in groupes.items():
        par_uhrh[f"sigeom_{g}"] = somme(np.isfinite(r) & (r > 0.5))
    par_uhrh = par_uhrh[par_uhrh.n > 0]

    out = []
    for t in _parse_troncon(Path(f"{proj}/physitel/troncon.trl")):
        uids = [u for u in t["uhrh_ids"] if u in par_uhrh.index]
        if not uids:
            continue
        s = par_uhrh.loc[uids].sum()
        row = {"region": reg, "troncon": int(t["id"]), "aire_m2": float(s.n) * RESOLUTION ** 2}
        row["drainage"] = float(s.num / s.den) if s.den > 0 else np.nan
        row["couv_drainage"] = float(s.den / s.n)
        for k in ("irda", "ecoforestier", "sigeom"):
            row[f"part_{k}"] = float(s[k] / s.n)
        for g in SIGEOM_GROUPE:
            row[f"sigeom_{g}"] = float(s[f"sigeom_{g}"] / s.n)
        for nom in seules:
            row[f"drainage_{nom}"] = float(s[f"num_{nom}"] / s[f"den_{nom}"]) if s[f"den_{nom}"] > 0 else np.nan
            row[f"couv_drainage_{nom}"] = float(s[f"den_{nom}"] / s.n)
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | couverture mediane {d.couv_drainage.median():.2f} | part IRDA {d.part_irda.mean():.2f}, ecoforestier {d.part_ecoforestier.mean():.2f}, SIGEOM {d.part_sigeom.mean():.2f} | drainage median {d.drainage.median():.2f} | {time.time() - t0:.0f} s", flush=True)
    return d


def enregistrer(d):
    """Fusionne le territoire dans le parquet des le calcul fini : un arret en cours de
    route ne perd que le territoire en cours, pas les precedents."""
    f = f"{SORTIE}/drainage-hierarchique{SUFFIXE}-troncons.parquet"
    if os.path.exists(f):
        ancien = pd.read_parquet(f)
        d = pd.concat([ancien[~ancien.region.isin(d.region.unique())], d], ignore_index=True)
    d.to_parquet(f, index=False)
    return f, len(d), len(d.columns)


def main(regions):
    for r in regions:
        d = une_region(r)
        if d is not None:
            f, n, k = enregistrer(d)
            print(f"  enregistre : {f} ({n} tronçons, {k} colonnes)", flush=True)
    print("croisement fait", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
