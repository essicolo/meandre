"""Pedo-paysages du Canada v3.2, croises par troncon.

Les Pedo-paysages (SISCan, Agriculture et Agroalimentaire Canada) decoupent le pays en
polygones au 1:1 000 000, chacun porte des composantes de sol en pourcentage, et chaque sol
nomme porte une classe de drainage etablie par les leves pedologiques, une profondeur de nappe,
une couche restrictive, et des couches avec conductivite hydraulique saturee, retentions a 33
et 1500 kPa et densite apparente. Contrairement a l'IRDA, la couverture inclut la foret du
sud du Quebec.

Par unite hydrologique, chaque variable est moyennee sur l'aire des polygones, ponderee par le
pourcentage des composantes qui la portent ; la fraction couverte est portee a cote, et une
valeur manquante n'est jamais comblee. Echelle de drainage : rang de 1 (tres rapidement) a
7 (tres mal draine), la meme que le croisement IRDA et la carte hierarchique.

    .venv/Scripts/python.exe .runs/quebec/croiser_slc.py mont slso slno outv gasp sagu
"""
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SLC = f"{_paths.DATA_ROOT}/sources/slc"
SORTIE = f"{_paths.DATA_ROOT}/derives/auxiliaires"
DRAINAGE = {"VR": 1.0, "R": 2.0, "W": 3.0, "MW": 4.0, "I": 5.0, "P": 6.0, "VP": 7.0}
VARIABLES = ["drainage", "log_ksat", "kp33", "kp1500", "densite", "nappe_cm", "restriction_cm"]


def proprietes_des_sols():
    import geopandas as gpd
    nom = gpd.read_file(f"{SLC}/soil_name_canada_v2r20231107.dbf")
    couche = gpd.read_file(f"{SLC}/soil_layer_canada_v2r20231108.dbf")
    s = pd.DataFrame({"SOIL_ID": nom.SOIL_ID})
    s["drainage"] = nom.DRAINAGE.map(DRAINAGE)
    s["nappe_cm"] = pd.to_numeric(nom.WATERTBL, errors="coerce").where(lambda v: v >= 0)
    s["restriction_cm"] = pd.to_numeric(nom.ROOTRESTRI, errors="coerce").where(lambda v: v >= 0)
    # Couches minerales du premier metre, ponderees par leur epaisseur ; les valeurs
    # negatives sont des codes d'absence dans ces tables.
    c = couche.copy()
    for k in ("UDEPTH", "LDEPTH", "KSAT", "KP33", "KP1500", "BD"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c = c[(c.UDEPTH >= 0) & (c.LDEPTH > c.UDEPTH)]
    c["ep"] = (c.LDEPTH.clip(upper=100) - c.UDEPTH.clip(upper=100)).clip(lower=0)
    c = c[c.ep > 0]
    c["log_ksat"] = np.log10(c.KSAT.where(c.KSAT > 0))
    c["kp33"] = c.KP33.where(c.KP33 >= 0)
    c["kp1500"] = c.KP1500.where(c.KP1500 >= 0)
    c["densite"] = c.BD.where(c.BD > 0)
    agg = {}
    for v in ("log_ksat", "kp33", "kp1500", "densite"):
        w = c.ep.where(c[v].notna(), 0.0)
        agg[v] = (c[v].fillna(0.0) * w).groupby(c.SOIL_ID).sum() / w.groupby(c.SOIL_ID).sum().replace(0, np.nan)
    s = s.merge(pd.DataFrame(agg).reset_index(), on="SOIL_ID", how="left")
    return s.set_index("SOIL_ID")


def une_region(reg, poly, cmp, sols):
    import geopandas as gpd
    proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
    uh = gpd.read_file(f"{proj}/physitel/uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    p = poly.to_crs(uh.crs)
    p = p[p.intersects(uh.union_all().envelope)][["POLY_ID", "geometry"]]
    inter = gpd.overlay(uh, p, how="intersection", keep_geom_type=True)
    inter["aire"] = inter.geometry.area
    a_uhrh = uh.assign(aire=uh.geometry.area).groupby("uhrh").aire.sum()
    # Valeur par polygone : moyenne des composantes ponderee par leur pourcentage, et part
    # du pourcentage qui porte la variable.
    cm = cmp[cmp.POLY_ID.isin(p.POLY_ID)].join(sols, on="SOIL_ID")
    par_poly = {}
    for v in VARIABLES:
        w = cm.PERCENT.where(cm[v].notna(), 0.0).astype(float)
        num = (cm[v].fillna(0.0) * w).groupby(cm.POLY_ID).sum()
        den = w.groupby(cm.POLY_ID).sum()
        par_poly[f"val_{v}"] = num / den.replace(0, np.nan)
        par_poly[f"part_{v}"] = den / cm.PERCENT.astype(float).groupby(cm.POLY_ID).sum()
    par_poly = pd.DataFrame(par_poly)
    inter = inter.join(par_poly, on="POLY_ID")
    par_uhrh = pd.DataFrame(index=a_uhrh.index)
    for v in VARIABLES:
        poids = inter.aire * inter[f"part_{v}"].fillna(0.0)
        par_uhrh[f"num_{v}"] = (inter[f"val_{v}"].fillna(0.0) * poids).groupby(inter.uhrh).sum()
        par_uhrh[f"den_{v}"] = poids.groupby(inter.uhrh).sum()
    par_uhrh = par_uhrh.fillna(0.0)
    out = []
    for t in _parse_troncon(Path(f"{proj}/physitel/troncon.trl")):
        uids = [u for u in t["uhrh_ids"] if u in a_uhrh.index]
        if not uids:
            continue
        at = float(a_uhrh.loc[uids].sum())
        s = par_uhrh.loc[uids].sum()
        row = {"region": reg, "troncon": int(t["id"]), "aire_m2": at}
        for v in VARIABLES:
            row[v] = float(s[f"num_{v}"] / s[f"den_{v}"]) if s[f"den_{v}"] > 0 else np.nan
            row[f"couv_{v}"] = float(s[f"den_{v}"] / at) if at > 0 else 0.0
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | couverture mediane drainage {d.couv_drainage.median():.2f}, conductivite {d.couv_log_ksat.median():.2f} | drainage median {d.drainage.median():.2f}, log10 K_sat median {d.log_ksat.median():.2f}", flush=True)
    return d


def main(regions):
    import geopandas as gpd
    poly = gpd.read_file(f"{SLC}/ca_all_slc_v3r2.shp")
    cmp = gpd.read_file(f"{SLC}/ca_all_slc_v3r2_cmp.dbf")
    sols = proprietes_des_sols()
    print(f"{len(poly)} polygones, {len(cmp)} composantes, {len(sols)} sols nommes, drainage connu pour {int(sols.drainage.notna().sum())}", flush=True)
    d = pd.concat([une_region(r, poly, cmp, sols) for r in regions], ignore_index=True)
    f = f"{SORTIE}/slc-troncons.parquet"
    if os.path.exists(f):
        ancien = pd.read_parquet(f)
        d = pd.concat([ancien[~ancien.region.isin(d.region.unique())], d], ignore_index=True)
    d.to_parquet(f, index=False)
    print(f"\n{f} : {len(d)} tronçons, {len(d.columns)} colonnes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
