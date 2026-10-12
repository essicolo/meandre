"""Épaisseur et nature des dépôts meubles d'après les puits du SIH, croisées par tronçon.

Le Système d'information hydrogéologique (MELCCFP) décrit plus de 260 000 puits et forages,
avec la séquence des matériaux recoupés et leur épaisseur. PHYSITEL n'a aucune profondeur du
roc exploitable (nulle sur tous les tronçons) ; ces rapports de forage sont la seule mesure
directe de l'épaisseur des dépôts disponible à l'échelle de la province.

Par puits : profondeur du roc, somme des épaisseurs au-dessus de la première couche de roc,
si le puits l'atteint ; sinon, la profondeur forée est une borne inférieure et le puits compte
dans la part des puits arrêtés dans les dépôts, signe d'un aquifère granulaire épais. Les dépôts
au-dessus du roc sont rangés par leur matériau principal (avant AVEC, PASSANT, à) : fins (argile,
silt), grossiers (sable, gravier, blocs), till.

Par tronçon, sur les puits tombant dans ses unités hydrologiques : médiane du logarithme de la
profondeur du roc, part des puits arrêtés dans les dépôts, parts des épaisseurs fines,
grossières et de till. La couverture vaut 1 à partir de `--puits-min` puits, 0 sous ce seuil ;
une valeur manquante n'est jamais comblée.

    .venv/Scripts/python.exe .runs/quebec/croiser_sih.py mont slso slno outv gasp sagu
"""
import argparse
import os
import sys
import unicodedata
import zipfile

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SIH = f"{_paths.DATA_ROOT}/sources/hydrogeologie-sih"
SORTIE = f"{_paths.DATA_ROOT}/derives/auxiliaires"
ROC = ("roc", "granit", "gneiss", "schiste", "shale", "calcaire", "gres", "sedimentaire", "marbre", "dolomi", "basalt", "quartzite", "ardoise", "anorthosite", "gabbro", "volcan", "conglom", "metamorph", "igne", "serpentin", "diorite", "syenite", "argilite", "socle")
FIN = ("argile", "silt", "glaise")
GROSSIER = ("sable", "gravier", "bloc", "cailloux", "galet")
VARIABLES = ["log_roc_m", "part_non_atteint", "part_fin", "part_grossier", "part_till"]


def _cle(texte) -> str:
    t = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode().lower()
    for coupe in (" avec ", " passant ", " a "):
        t = t.split(coupe)[0]
    return t.strip()


def _classe(texte) -> str:
    t = _cle(texte)
    if t.startswith(ROC) or any(t.startswith(r) for r in ROC):
        return "roc"
    if t.startswith("till"):
        return "till"
    if t.startswith(FIN):
        return "fin"
    if t.startswith(GROSSIER):
        return "grossier"
    return "autre"


def puits_sih() -> pd.DataFrame:
    """Une ligne par puits : position, profondeur du roc ou borne, parts des dépôts."""
    import geopandas as gpd

    gpkg = Path(SORTIE) / "cache" / "Puits_SIH.gpkg"
    if not gpkg.exists():
        gpkg.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(f"{SIH}/Puits_SIH.gpkg.zip") as z:
            nom = next(n for n in z.namelist() if n.endswith(".gpkg"))
            with z.open(nom) as src, open(gpkg, "wb") as dst:
                dst.write(src.read())
    p = gpd.read_file(gpkg, layer="Puits_et_Forages", ignore_geometry=True)
    m = gpd.read_file(gpkg, layer="Materiaux_recoupes", ignore_geometry=True)
    m = m[m.EPAISSEUR_COUCHE.notna() & (m.EPAISSEUR_COUCHE >= 0)].copy()
    m["classe"] = m.TYPE_MATERIAU.map(_classe)
    m = m.sort_values(["CLE_PUITS", "NO_SEQ_COUCHE"])
    # Couches au-dessus de la première couche de roc.
    m["roc_vu"] = (m.classe == "roc").groupby(m.CLE_PUITS).cumsum() > 0
    dessus = m[~m.roc_vu]
    ep = dessus.groupby("CLE_PUITS").EPAISSEUR_COUCHE.sum()
    atteint = m.groupby("CLE_PUITS").roc_vu.any()
    parts = dessus.pivot_table(index="CLE_PUITS", columns="classe", values="EPAISSEUR_COUCHE", aggfunc="sum").fillna(0.0)
    w = pd.DataFrame({"ep_depots_m": ep.reindex(atteint.index).fillna(0.0), "atteint": atteint})
    tot = parts.sum(axis=1).replace(0, np.nan)
    for c in ("fin", "grossier", "till"):
        w[f"part_{c}"] = (parts[c] / tot) if c in parts else 0.0
    p = p.drop_duplicates("CLE_PUITS").set_index("CLE_PUITS")[["LONGITUDE", "LATITUDE"]]
    w = w.join(p, how="inner").dropna(subset=["LONGITUDE", "LATITUDE"])
    print(f"{len(w)} puits avec matériaux et position, {int(w.atteint.sum())} atteignent le roc ; profondeur médiane du roc {w.ep_depots_m[w.atteint].median():.1f} m", flush=True)
    return w


def une_region(reg, puits, puits_min):
    import geopandas as gpd

    proj = f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020"
    uh = gpd.read_file(f"{proj}/physitel/uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    a_uhrh = uh.assign(aire=uh.geometry.area).groupby("uhrh").aire.sum()
    pts = gpd.GeoDataFrame(puits, geometry=gpd.points_from_xy(puits.LONGITUDE, puits.LATITUDE), crs="EPSG:4326").to_crs(uh.crs)
    xmin, ymin, xmax, ymax = uh.total_bounds
    pts = pts.cx[xmin:xmax, ymin:ymax]
    j = gpd.sjoin(pts, uh, predicate="within", how="inner")
    j["log_roc_m"] = np.log10(j.ep_depots_m.where(j.atteint & (j.ep_depots_m > 0)).clip(lower=0.1))
    j["part_non_atteint"] = (~j.atteint).astype(float)
    par_uhrh = {u: g for u, g in j.groupby("uhrh")}
    out = []
    for t in _parse_troncon(Path(f"{proj}/physitel/troncon.trl")):
        uids = [u for u in t["uhrh_ids"] if u in a_uhrh.index]
        if not uids:
            continue
        g = [par_uhrh[u] for u in uids if u in par_uhrh]
        g = pd.concat(g) if g else j.iloc[:0]
        row = {"region": reg, "troncon": int(t["id"]), "aire_m2": float(a_uhrh.loc[uids].sum()), "n_puits": len(g)}
        for v in VARIABLES:
            s = g[v].dropna()
            row[v] = float(s.median() if v == "log_roc_m" else s.mean()) if len(s) >= puits_min else np.nan
            row[f"couv_{v}"] = 1.0 if len(s) >= puits_min else 0.0
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(j)} puits, {len(d)} tronçons, couverts {d.couv_log_roc_m.mean():.2f} | profondeur du roc médiane {10 ** d.log_roc_m.median():.1f} m, part arrêtée dans les dépôts {d.part_non_atteint.median():.2f}, part fine {d.part_fin.median():.2f}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*", default=["mont"])
    ap.add_argument("--puits-min", type=int, default=3)
    a = ap.parse_args()
    puits = puits_sih()
    d = pd.concat([une_region(r.lower(), puits, a.puits_min) for r in a.regions], ignore_index=True)
    f = f"{SORTIE}/sih-troncons.parquet"
    if os.path.exists(f):
        ancien = pd.read_parquet(f)
        d = pd.concat([ancien[~ancien.region.isin(d.region.unique())], d], ignore_index=True)
    d.to_parquet(f, index=False)
    print(f"\n{f} : {len(d)} tronçons", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
