"""Épaisseur des dépôts quaternaires par tronçon, d'après SIGEOM (zones morphosédimentologiques).

Chaque polygone reçoit l'épaisseur inscrite (moyenne de VAL_EPAIS_MINIM et VAL_EPAIS_MAXIM, ou la
seule des deux qui existe) ; à défaut, la médiane des épaisseurs inscrites pour son code de dépôt,
calculée sur tout le Québec. Les codes sans aucune épaisseur inscrite restent sans valeur. Le roc
(codes en R) vaut son épaisseur inscrite, 0,3 m en médiane.

Par tronçon : moyenne géométrique de l'épaisseur sur la surface cartographiée, part de la surface
à 1 m ou moins, part à 5 m ou plus, part cartographiée avec une épaisseur. Grille de 100 m dans la
projection du projet PHYSITEL, comme `depots_troncons.py`.

    .venv/Scripts/python.exe experiments/plan_ml/epaisseur-depots.py mont slso outv slno
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, ".runs", "quebec"))
from croiser_drainage import RESOLUTION, SIGEOM, rasteriser
from meandre.adapters.physitel.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

SORTIE = f"{_paths.DATA_ROOT}/experiments/plan_ml/epaisseur-depots-troncons.parquet"
COUCHE = "F10E15_ZONE_MORPH_SEDIM"


def epaisseur_par_code():
    import pyogrio
    d = pyogrio.read_dataframe(SIGEOM, layer=COUCHE, read_geometry=False, columns=["CODE_DEPOT_MORP_SEDM", "VAL_EPAIS_MINIM", "VAL_EPAIS_MAXIM"])
    d["e"] = epaisseur_inscrite(d)
    return d.groupby(d.CODE_DEPOT_MORP_SEDM.fillna(""))["e"].median()


def epaisseur_inscrite(d):
    lo = d.VAL_EPAIS_MINIM.where(d.VAL_EPAIS_MINIM > 0)
    hi = d.VAL_EPAIS_MAXIM.where(d.VAL_EPAIS_MAXIM > 0)
    return pd.concat([lo, hi], axis=1).mean(axis=1)


def une_region(reg, par_code):
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    t0 = time.time()
    proj = Path(_paths.PLATFORMS_ROOT) / "LN24HA" / f"{reg.upper()}_LN24HA_2020" / "physitel"
    uh = gpd.read_file(proj / "uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    x0, y0, x1, y1 = uh.total_bounds
    forme = (int(np.ceil((y1 - y0) / RESOLUTION)), int(np.ceil((x1 - x0) / RESOLUTION)))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    ids = features.rasterize(((g, int(u)) for g, u in zip(uh.geometry, uh.uhrh)), out_shape=forme, transform=transform, fill=0, dtype=np.int32)
    z = gpd.read_file(SIGEOM, layer=COUCHE, columns=["CODE_DEPOT_MORP_SEDM", "VAL_EPAIS_MINIM", "VAL_EPAIS_MAXIM"], engine="pyogrio", bbox=tuple(uh.to_crs(4326).total_bounds)).to_crs(uh.crs)
    e = epaisseur_inscrite(z)
    z["e"] = e.fillna(z.CODE_DEPOT_MORP_SEDM.fillna("").map(par_code))
    z["loge"] = np.log(z.e.clip(lower=0.1))
    z["mince"] = (z.e <= 1.0).astype(float).where(z.e.notna())
    z["epais"] = (z.e >= 5.0).astype(float).where(z.e.notna())
    dans = ids > 0
    n = np.bincount(ids[dans])
    par = pd.DataFrame({"n": n})
    for c in ("loge", "mince", "epais"):
        r = rasteriser(z, c, transform, forme)
        ok = dans & np.isfinite(r)
        par[f"n_{c}"] = np.bincount(ids[ok], minlength=len(n))
        par[f"s_{c}"] = np.bincount(ids[ok], weights=r[ok], minlength=len(n))
    par = par[par.n > 0]
    out = []
    for t in _parse_troncon(proj / "troncon.trl"):
        u = [x for x in t["uhrh_ids"] if x in par.index]
        if not u:
            continue
        s = par.loc[u].sum()
        k = float(s.n_loge)
        out.append({"region": reg, "troncon": int(t["id"]), "part_epaisseur": k / float(s.n), "epaisseur_m": float(np.exp(s.s_loge / k)) if k > 0 else np.nan, "part_mince": float(s.s_mince / k) if k > 0 else np.nan, "part_epais": float(s.s_epais / k) if k > 0 else np.nan})
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | part avec épaisseur {d.part_epaisseur.mean():.2f} | épaisseur médiane {d.epaisseur_m.median():.2f} m | mince {d.part_mince.mean():.2f} | épais {d.part_epais.mean():.2f} | {time.time() - t0:.0f} s", flush=True)
    return d


def main(regions):
    par_code = epaisseur_par_code()
    old = pd.read_parquet(SORTIE) if os.path.exists(SORTIE) else None
    d = pd.concat([une_region(r, par_code) for r in regions], ignore_index=True)
    if old is not None:
        d = pd.concat([old[~old.region.isin(regions)], d], ignore_index=True)
    d.to_parquet(SORTIE, index=False)
    print(f"-> {SORTIE} ({len(d)} tronçons)")


if __name__ == "__main__":
    main([r.lower() for r in sys.argv[1:]])
