"""Texture PHYSITEL par tronçon, lue comme Hydrotel la lit.

Chaque unité hydrologique porte dans `type_sol.cla` l'indice, compté à partir de zéro, d'une
rangée de la table des sols ; la fraction de sable, de silt et d'argile de cette classe est
moyennée sur le tronçon, pondérée par l'aire des unités. Remplace la table du 2026-09-24,
écrite avant la correction de l'indice qui lisait chaque classe une rangée trop haut.

    python -m meandre.adapters.physitel.builders.reach_texture mont slso slno outv gasp sagu
"""
import os
import sys

from pathlib import Path

import pandas as pd


from meandre.adapters.physitel.physitel_loader import _TEXTURE_FRACTIONS, _parse_reach, _parse_soil_type_cla
from meandre.utils import paths as _paths

PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")
OUTPUT = f"{_paths.DATA_ROOT}/derives/auxiliaires/physitel-texture-troncons.parquet"


def build_territory(reg):
    import geopandas as gpd

    proj = Path(f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020/physitel")
    uh = gpd.read_file(proj / "uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    area_value = uh.assign(area_value=uh.geometry.area).groupby(uh[col].astype(int)).area_value.sum()
    classe = _parse_soil_type_cla(proj / "type_sol.cla", {})
    out = []
    for t in _parse_reach(proj / "troncon.trl"):
        uids = [u for u in t["uhrh_ids"] if u in area_value.index]
        if not uids:
            continue
        a = area_value.loc[uids]
        f = [_TEXTURE_FRACTIONS[max(0, min(classe.get(u, 0), len(_TEXTURE_FRACTIONS) - 1))] for u in uids]
        f = pd.DataFrame(f, index=uids, columns=["f_sand", "f_silt", "f_clay"]).mul(a, axis=0).sum() / a.sum()
        out.append({"region": reg, "troncon": int(t["id"]), "aire_m2": float(a.sum()), **f.to_dict()})
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | sable {d.f_sand.median():.2f}, argile {d.f_clay.median():.2f} en médiane", flush=True)
    return d


def main(regions):
    d = pd.concat([build_territory(r) for r in regions], ignore_index=True)
    if os.path.exists(OUTPUT):
        old = pd.read_parquet(OUTPUT)
        old = old[~old.region.isin(d.region.unique())]
        d = pd.concat([old[[c for c in d.columns if c in old.columns]], d], ignore_index=True)
    d.to_parquet(OUTPUT, index=False)
    print(f"{OUTPUT} : {len(d)} tronçons", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["mont"]))
