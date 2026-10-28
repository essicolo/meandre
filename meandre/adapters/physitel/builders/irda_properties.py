"""Propriétés numériques des sols de l'IRDA, agrégées au tronçon.

Source : géopaquet de la couverture pédologique du Québec, version 2026_01. Chaque polygone
se décompose en composantes pondérées par leur pourcentage ; chaque composante porte ses
propriétés pédologiques et, pour 57 % d'entre elles, une texture de surface de la BDHP.

Variables produites, chacune avec sa propre couverture :
- texture en deux coordonnées ilr (sable, limon, argile), tirée de la BDHP ou, à défaut, du
  centre de gravité de la classe granulométrique de famille du matériau de surface ;
- rang de drainage, de 1 (très rapidement drainé) à 7 (très mal drainé) ;
- logarithme décimal de la perméabilité en mm/h, au centre géométrique de la classe de
  Wischmeier et Smith (1978) ;
- code de structure, de 1 (granulaire très fine) à 4 (en blocs, lamellaire ou massive) ;
- groupe hydrologique, de 1 (A) à 4 (D) ;
- part des sols à contact lithique mince ou très mince ;
- part des sols organiques.

Les moyennes des classes numériques et des coordonnées ilr sont pondérées par le pourcentage
des composantes, puis par l'aire au sein des unités hydrologiques et des tronçons.

    python -m meandre.adapters.physitel.builders.irda_properties outv gasp ...
"""
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from meandre.adapters.physitel.physitel_loader import _parse_reach
from meandre.utils import paths as _paths

IRDA = f"{_paths.DATA_ROOT}/irda"
GPKG = f"{IRDA}/couverture_pedologique_2026_01.gpkg"
PLATE = "LN24HA"
REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud"]

# Classes granulométriques de famille, en sommets (sable %, argile %) du triangle textural.
# La fraction fine d'une classe squelettique est celle de la classe sans préfixe.
_SANDY = [(70, 0), (100, 0), (85, 15)]
_LOAMY = [(0, 0), (70, 0), (85, 15), (65, 35), (0, 35)]
_CLAYEY = [(0, 35), (65, 35), (0, 100)]
FAMILIES = {
    "Sableuse": _SANDY,
    "Squelettique-sableuse": _SANDY,
    "Loameuse-grossière": [(15, 0), (70, 0), (85, 15), (82, 18), (15, 18)],
    "Loameuse-fine": [(15, 18), (82, 18), (65, 35), (15, 35)],
    "Limoneuse-grossière": [(0, 0), (15, 0), (15, 18), (0, 18)],
    "Limoneuse-fine": [(0, 18), (15, 18), (15, 35), (0, 35)],
    "Loameuse": _LOAMY,
    "Squelettique-loameuse": _LOAMY,
    "Argileuse": _CLAYEY,
    "Squelettique-argileuse": _CLAYEY,
    "Argileuse-fine": [(0, 35), (65, 35), (40, 60), (0, 60)],
    "Argileuse-très fine": [(0, 60), (40, 60), (0, 100)],
}
DRAINAGE = {"Très rapidement drainé": 1, "Rapidement drainé": 2, "Bien drainé": 3, "Modérément bien drainé": 4, "Imparfaitement drainé": 5, "Mal drainé": 6, "Très mal drainé": 7}
# Classes de perméabilité de Wischmeier et Smith (1978), bornes en mm/h ; classes extrêmes fermées à 500 et 0,5.
PERMEABILITY = {1: (152, 500), 2: (51, 152), 3: (15, 51), 4: (5, 15), 5: (1.5, 5), 6: (0.5, 1.5)}
GROUP = {"A": 1.0, "B": 2.0, "C": 3.0, "C/D": 3.5, "D": 4.0}
VARIABLES = ["ilr_1", "ilr_2", "drainage", "log_permeabilite", "structure", "groupe_hydro", "lithique", "organique"]


def centroid(vertices):
    """Centre de gravité d'un polygone du triangle, rendu en (sable, limon, argile) %."""
    x = np.array([p[0] for p in vertices], dtype=float)
    y = np.array([p[1] for p in vertices], dtype=float)
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    a = c.sum() / 2
    sand = ((x + x1) * c).sum() / (6 * a)
    clay = ((y + y1) * c).sum() / (6 * a)
    return sand, 100 - sand - clay, clay


def component_properties():
    import nuee
    cx = sqlite3.connect(GPKG)
    pp = pd.read_sql("select Composante, Sorte, Classe_drainage, Granulo_1, Profondeur_sol from Proprietes_pedologiques", cx)
    bd = pd.read_sql("select Composante, Sable, Limon, Argile, Permeabilite, Code_structure, Groupe_hydrologique from BDHP_2026", cx)
    cx.close()
    d = pp.merge(bd, on="Composante", how="left")
    centers = {k: centroid(v) for k, v in FAMILIES.items()}
    tex = d[["Sable", "Limon", "Argile"]].to_numpy(dtype=float, copy=True)
    without = ~np.isfinite(tex).all(axis=1)
    for i in np.flatnonzero(without):
        g = d.Granulo_1.iat[i]
        if g in centers:
            tex[i] = centers[g]
    ok = np.isfinite(tex).all(axis=1) & (d.Sorte == "Minéral").to_numpy()
    coords = np.full((len(d), 2), np.nan)
    coords[ok] = nuee.ilr(nuee.multiplicative_replacement(nuee.closure(tex[ok])))
    d["ilr_1"], d["ilr_2"] = coords[:, 0], coords[:, 1]
    d["texture_source"] = np.where(~ok, None, np.where(without, "centre de classe", "BDHP"))
    d["drainage"] = d.Classe_drainage.map(DRAINAGE)
    d["log_permeabilite"] = d.Permeabilite.map({k: 0.5 * (np.log10(a) + np.log10(b)) for k, (a, b) in PERMEABILITY.items()})
    d["structure"] = d.Code_structure.astype(float)
    d["groupe_hydro"] = d.Groupe_hydrologique.map(GROUP)
    mineral = d.Sorte.isin(["Minéral", "Organique"])
    d["lithique"] = np.where(mineral, d.Profondeur_sol.isin(["Lithique mince", "Lithique très mince"]).astype(float), np.nan)
    d["organique"] = np.where(mineral, (d.Sorte == "Organique").astype(float), np.nan)
    return d


def properties_polygons(comp):
    cx = sqlite3.connect(GPKG)
    pps = pd.read_sql("select Code_polygone, Composante, Pourcentage from Couverture_pps", cx)
    cx.close()
    d = pps.merge(comp[["Composante"] + VARIABLES], on="Composante", how="left")
    lines = {"Code_polygone": d.Code_polygone.unique()}
    step_output = pd.DataFrame(lines).set_index("Code_polygone")
    for v in VARIABLES:
        m = d[v].notna()
        w = d.Pourcentage.where(m, 0.0)
        num = (d[v].fillna(0.0) * w).groupby(d.Code_polygone).sum()
        den = w.groupby(d.Code_polygone).sum()
        step_output[v] = num / den.replace(0, np.nan)
        step_output[f"couv_{v}"] = den / 100.0
    return step_output.reset_index()


def build_territory(reg, poly):
    import geopandas as gpd
    proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
    uh = gpd.read_file(f"{proj}/physitel/uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    ir = poly.to_crs(uh.crs)
    ir = ir[ir.intersects(uh.union_all().envelope)]
    inter = gpd.overlay(uh, ir, how="intersection", keep_geom_type=True)
    inter["area_value"] = inter.geometry.area
    # Une unité hydrologique peut compter plusieurs polygones : son aire est leur somme.
    a_uhrh = uh.assign(area_value=uh.geometry.area).groupby("uhrh").area_value.sum()
    par_uhrh = pd.DataFrame(index=a_uhrh.index)
    for v in VARIABLES:
        poids = inter.area_value * inter[f"couv_{v}"].fillna(0.0)
        par_uhrh[f"num_{v}"] = (inter[v].fillna(0.0) * poids).groupby(inter.uhrh).sum()
        par_uhrh[f"den_{v}"] = poids.groupby(inter.uhrh).sum()
    par_uhrh = par_uhrh.fillna(0.0)
    out = []
    for t in _parse_reach(Path(f"{proj}/physitel/troncon.trl")):
        uids = [u for u in t["uhrh_ids"] if u in a_uhrh.index]
        if not uids:
            continue
        at = float(a_uhrh.loc[uids].sum())
        s = par_uhrh.loc[uids].sum()
        row = {"region": reg, "troncon": int(t["id"]), "aire_m2": at}
        for v in VARIABLES:
            row[v] = s[f"num_{v}"] / s[f"den_{v}"] if s[f"den_{v}"] > 0 else np.nan
            row[f"couv_{v}"] = s[f"den_{v}"] / at if at > 0 else 0.0
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | couverture médiane texture {100 * d.couv_ilr_1.median():.0f} %, drainage {100 * d.couv_drainage.median():.0f} %, perméabilité {100 * d.couv_log_permeabilite.median():.0f} % | maximum {d[[c for c in d.columns if c.startswith('couv_')]].max().max():.3f}", flush=True)
    return d


def main(regions):
    import geopandas as gpd
    comp = component_properties()
    print(f"{len(comp)} composantes | texture BDHP {int((comp.texture_source == 'BDHP').sum())}, centre de classe {int((comp.texture_source == 'centre de classe').sum())}, sans texture {int(comp.texture_source.isna().sum())}", flush=True)
    props = properties_polygons(comp)
    geo = gpd.read_file(GPKG, layer="Couverture_pedologique", columns=["Code_polygone"], engine="pyogrio")
    poly = geo.merge(props, on="Code_polygone", how="inner")
    poly = poly[poly[[f"couv_{v}" for v in VARIABLES]].sum(axis=1) > 0]
    print(f"{len(poly)} polygones porteurs d'au moins une propriété", flush=True)
    tout = [build_territory(r, poly) for r in regions]
    d = pd.concat(tout, ignore_index=True)
    f = f"{IRDA}/irda-proprietes-troncons.parquet"
    d.to_parquet(f, index=False)
    print(f"\n{f} : {len(d)} tronçons, {len(d.columns)} colonnes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or REGIONS))


RESERVOIRS = ["ilr_1", "ilr_2", "drainage", "log_permeabilite", "lithique", "organique"]


def columns_nodes(reg, node_ids):
    """Colonnes IRDA pour l'entrée du champ, dans l'ordre des nœuds.

    Chaque propriété est centrée et réduite sur les tronçons couverts à plus de moitié de la
    province, puis multipliée par sa couverture : un tronçon non cartographié reçoit la valeur
    neutre 0. La dernière colonne est la couverture du drainage, qui sert de masque.
    """
    t = pd.read_parquet(f"{IRDA}/irda-proprietes-troncons.parquet")
    x = np.zeros((len(node_ids), len(RESERVOIRS) + 1), dtype=np.float32)
    m = t[t.region == reg].set_index("troncon").reindex([int(i) for i in node_ids])
    for j, v in enumerate(RESERVOIRS):
        ref = t.loc[(t[f"couv_{v}"] >= 0.5) & t[v].notna(), v]
        cover = m[f"couv_{v}"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        z = ((m[v] - ref.mean()) / ref.std()).fillna(0.0).to_numpy()
        x[:, j] = z * cover
    x[:, -1] = m["couv_drainage"].fillna(0.0).clip(0.0, 1.0).to_numpy()
    return x, [f"irda_{v}" for v in RESERVOIRS] + ["irda_couverture"]
