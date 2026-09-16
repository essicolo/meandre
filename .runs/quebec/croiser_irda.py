"""Croise la couverture pedologique de l'IRDA avec les unites hydrologiques, puis agrege
au troncon, dans les memes unites que les attributs territoriaux du modele.

Ce que le modele recoit aujourd'hui pour decrire le sol : trois pourcentages
granulometriques, dont trois classes couvrent 90 pour cent de la province et dont une seule
en couvre la moitie. Deux petits bassins voisins recoivent donc presque surement les memes
attributs, et le champ spatial ne peut pas les distinguer. Mesure du 2026-09-15 : le KGE
median vaut 0,526 sous 100 km2 contre 0,763 entre mille et trois mille.

Ce que l'IRDA ajoute, au 1:20 000 : une classe de drainage en cinq modalites, etablie au
terrain par l'examen morphologique du profil, un materiau parental en six classes
genetiques, et l'affleurement rocheux cartographie comme tel, cette derniere variable etant
aujourd'hui une colonne de zeros faute de donnee.

    .venv/Scripts/python.exe .runs/quebec/croiser_irda.py mont slso slno outv
"""
import os
import sys

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

DRAIN = ["Excessif", "Bon", "Imparfait", "Mauvais", "Très mauvais"]
MATER = ["ARGILE", "TILL", "SABLE", "LIMON", "GRAVIER", "ORGANIQUE"]
PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SORTIE = os.environ.get("IRDA_SORTIE", f"{_paths.DATA_ROOT}/irda")


def classe(s, liste):
    if not isinstance(s, str):
        return None
    for d in sorted(liste, key=len, reverse=True):
        if s.lower().endswith(d.lower()):
            return d
    return None


def materiau(s):
    if not isinstance(s, str):
        return None
    m = s.split(" ")[0].upper()
    return m if m in MATER else None


def une_region(reg, irda):
    proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
    shp = f"{proj}/physitel/uhrh.shp"
    trl = f"{proj}/physitel/troncon.trl"
    if not (os.path.exists(shp) and os.path.exists(trl)):
        print(f"{reg}: projet PHYSITEL introuvable")
        return None
    uh = gpd.read_file(shp)
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    epsg = uh.crs
    ir = irda.to_crs(epsg)
    # Intersection : l'aire de chaque classe dans chaque unite hydrologique.
    inter = gpd.overlay(uh, ir, how="intersection", keep_geom_type=True)
    inter["aire"] = inter.geometry.area

    tr = _parse_troncon(Path(f"{proj}/physitel/troncon.trl"))
    lignes = []
    # Une unite hydrologique peut compter plusieurs polygones : son aire est leur somme.
    a_uhrh = uh.assign(aire=uh.geometry.area).groupby("uhrh").aire.sum().to_dict()
    for cat, col_ in (("drainage", DRAIN), ("materiau", MATER), ("roc", ["AFFLEUREMENT"])):
        g = inter[inter[cat].notna()].groupby(["uhrh", cat]).aire.sum().reset_index()
        for r in g.itertuples():
            lignes.append({"uhrh": int(r.uhrh), "cle": f"{cat}_{getattr(r, cat)}",
                           "aire": float(r.aire)})
    par_uhrh = pd.DataFrame(lignes)
    if par_uhrh.empty:
        print(f"{reg}: aucune intersection")
        return None
    piv = par_uhrh.pivot_table(index="uhrh", columns="cle", values="aire",
                               aggfunc="sum").fillna(0.0)

    out = []
    for t in tr:
        uids = [u for u in t["uhrh_ids"] if u in a_uhrh]
        if not uids:
            continue
        at = sum(a_uhrh[u] for u in uids)
        row = {"troncon": int(t["id"]), "aire_m2": at}
        sous = piv.reindex(uids).fillna(0.0).sum(axis=0)
        for k, v in sous.items():
            row[k] = float(v) / at if at > 0 else 0.0
        out.append(row)
    d = pd.DataFrame(out).fillna(0.0)
    d["region"] = reg
    cols = [c for c in d.columns if c.startswith(("drainage_", "materiau_", "roc_"))]
    d["couvert"] = d[[c for c in cols if c.startswith("drainage_")]].sum(axis=1)
    print(f"{reg}: {len(d)} tronçons | couverture pédologique médiane "
          f"{100 * d.couvert.median():.0f} % | tronçons couverts à plus de 50 % : "
          f"{int((d.couvert > 0.5).sum())} | couverture maximale {d.couvert.max():.3f}")
    return d


def main(regions):
    f = f"{SORTIE}/irda-brut-QC.parquet"
    if os.path.exists(f):
        irda = gpd.read_parquet(f)
    else:
        print("lecture de la couverture complète (une seule fois)", flush=True)
        irda = gpd.read_file(f"{SORTIE}/couverture_pedologique_2026_01.geojson",
                             columns=["Symbole"], engine="pyogrio")
        irda["drainage"] = irda.Symbole.map(lambda s: classe(s, DRAIN))
        irda["materiau"] = irda.Symbole.map(materiau)
        irda["roc"] = np.where(irda.Symbole == "AFFLEUREMENT", "AFFLEUREMENT", None)
        irda = irda[irda.drainage.notna() | irda.roc.notna()].reset_index(drop=True)
        irda.to_parquet(f)
        print(f"cache écrit : {f} ({len(irda)} polygones)", flush=True)
    tout = []
    for reg in regions:
        d = une_region(reg, irda)
        if d is not None:
            tout.append(d)
    if tout:
        d = pd.concat(tout, ignore_index=True).fillna(0.0)
        g = f"{SORTIE}/irda-troncons.parquet"
        d.to_parquet(g, index=False)
        print(f"\n{g} : {len(d)} tronçons, {len(d.columns)} colonnes")
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
