"""Texture PHYSITEL par tronçon, lue comme Hydrotel la lit.

Chaque unité hydrologique porte dans `type_sol.cla` l'indice, compté à partir de zéro, d'une
rangée de la table des sols ; la fraction de sable, de silt et d'argile de cette classe est
moyennée sur le tronçon, pondérée par l'aire des unités. Remplace la table du 2026-09-24,
écrite avant la correction de l'indice qui lisait chaque classe une rangée trop haut.

    .venv/Scripts/python.exe .runs/quebec/texture_physitel_troncons.py mont slso slno outv gasp sagu
"""
import os
import sys

from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.adapters.physitel.physitel_loader import _TEXTURE_FRACTIONS, _parse_troncon, _parse_type_sol_cla
from meandre.utils import paths as _paths

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
SORTIE = f"{_paths.DATA_ROOT}/derives/auxiliaires/physitel-texture-troncons.parquet"


def une_region(reg):
    import geopandas as gpd

    proj = Path(f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020/physitel")
    uh = gpd.read_file(proj / "uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    aire = uh.assign(aire=uh.geometry.area).groupby(uh[col].astype(int)).aire.sum()
    classe = _parse_type_sol_cla(proj / "type_sol.cla", {})
    out = []
    for t in _parse_troncon(proj / "troncon.trl"):
        uids = [u for u in t["uhrh_ids"] if u in aire.index]
        if not uids:
            continue
        a = aire.loc[uids]
        f = [_TEXTURE_FRACTIONS[max(0, min(classe.get(u, 0), len(_TEXTURE_FRACTIONS) - 1))] for u in uids]
        f = pd.DataFrame(f, index=uids, columns=["f_sand", "f_silt", "f_clay"]).mul(a, axis=0).sum() / a.sum()
        out.append({"region": reg, "troncon": int(t["id"]), "aire_m2": float(a.sum()), **f.to_dict()})
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | sable {d.f_sand.median():.2f}, argile {d.f_clay.median():.2f} en médiane", flush=True)
    return d


def main(regions):
    d = pd.concat([une_region(r) for r in regions], ignore_index=True)
    if os.path.exists(SORTIE):
        ancien = pd.read_parquet(SORTIE)
        ancien = ancien[~ancien.region.isin(d.region.unique())]
        d = pd.concat([ancien[[c for c in d.columns if c in ancien.columns]], d], ignore_index=True)
    d.to_parquet(SORTIE, index=False)
    print(f"{SORTIE} : {len(d)} tronçons", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
