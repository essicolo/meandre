"""Carte des classes de drainage des sols de l'IRDA, pour la presentation.

Produit un PNG UNE FOIS. La diapositive ne fait que le referencer, pour que le rendu du
document ne relance pas une lecture de 1,45 gigaoctet ni un trace de plusieurs minutes.

Entree : les caches produits par le depouillement de la couverture pedologique,
`irda-drainage-QC.parquet` et `irda-affleurement-QC.parquet`.

    .venv/Scripts/python.exe .runs/quebec/carte_drainage_irda.py
"""
import os
import sys

import geopandas as gpd
import lets_plot as lp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import palette as pal
from meandre.utils import paths as _paths

DRAIN = ["Excessif", "Bon", "Imparfait", "Mauvais", "Très mauvais"]
SORTIE = os.environ.get("MEANDRE_FIGS", ".reports/quebec/figs")
SIMPLIF = float(os.environ.get("IRDA_SIMPLIF", "0.002"))


def anneaux(gdf, colonnes=()):
    rows = []
    for r_ in gdf.itertuples():
        g = r_.geometry
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for k, pg in enumerate(polys):
            xs, ys = pg.exterior.coords.xy
            rows.append(pd.DataFrame({"lon": xs, "lat": ys, "grp": f"{r_.cle}-{k}",
                                      **{c: getattr(r_, c) for c in colonnes}}))
    return pd.concat(rows, ignore_index=True)


def main():
    lp.LetsPlot.setup_html()
    rep = f"{_paths.DATA_ROOT}/quebec/rapport"
    irda = f"{_paths.DATA_ROOT}/irda"
    f = f"{irda}/irda-drainage-QC.parquet"
    if not os.path.exists(f):
        print(f"cache absent : {f}")
        return 1
    sol = gpd.read_parquet(f)
    sol["drainage"] = pd.Categorical(sol.drainage, categories=DRAIN, ordered=True)
    sol = sol.sort_values("drainage")
    aire = sol.to_crs(32198).area
    part = (100 * aire / aire.sum()).round(0)
    reg = gpd.read_parquet(f"{rep}/regions.parquet")
    a_qc = reg.to_crs(32198).area.sum()
    couvert = 100 * aire.sum() / a_qc

    sol["geometry"] = sol.geometry.simplify(SIMPLIF)
    ex = sol.explode(index_parts=False).reset_index(drop=True)
    ex = ex[ex.geometry.area > (SIMPLIF ** 2) * 4]
    ex["cle"] = [f"{d}-{i}" for i, d in enumerate(ex.drainage)]
    pts = anneaux(ex, ["drainage"])
    print(f"{len(ex)} contours, {len(pts)} points tracés", flush=True)

    contours = pd.read_parquet(f"{rep}/regions-contours.parquet")
    p = (lp.ggplot()
         + lp.geom_path(lp.aes("lon", "lat", group="grp"), data=contours,
                        color=pal.GRIS, size=.3)
         + lp.geom_polygon(lp.aes("lon", "lat", group="grp", fill="drainage"), data=pts, size=0)
         + lp.scale_fill_manual(values=pal.SEQUENTIEL_5, breaks=DRAIN, name="drainage du sol")
         + lp.coord_map()
         + lp.labs(x="", y="")
         + lp.theme_minimal()
         + lp.theme(axis_text=lp.element_blank(), panel_grid=lp.element_blank()))
    os.makedirs(SORTIE, exist_ok=True)
    lp.ggsave(p + lp.ggsize(1180, 620), "drainage-qc.png", path=SORTIE, scale=2)
    txt = ", ".join(f"{d} {int(v)} %" for d, v in zip(sol.drainage, part))
    print(f"{SORTIE}/drainage-qc.png")
    print(f"couverture : {couvert:.0f} pour cent de la superficie des quinze régions")
    print(f"répartition de la surface cartographiée : {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
