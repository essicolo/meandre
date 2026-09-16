"""Carte de l'effet des prelevements et rejets sur le debit annuel. PNG rendu UNE FOIS.

La version interactive demandait a lets_plot de tracer 23 816 chemins sur un fond de carte :
elle n'en dessinait qu'une partie et se recadrait sur ce qui passait, d'ou une carte zoomee
sur un pate rouge en Monterégie au lieu du Quebec meridional.

Ici, le reseau de contexte est reduit aux troncons qui portent plus de 5 m3/s, les troncons
affectes sont traces par-dessus, et l'emprise est celle des quinze regions.

    .venv/Scripts/python.exe .runs/quebec/carte_effet_prelevements.py
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

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
SEUIL_CONTEXTE = 5.0     # m3/s : le reseau de contexte
SEUIL_EFFET = 0.5        # pour cent : en deca, le troncon est dit non affecte
SORTIE = os.environ.get("MEANDRE_FIGS", ".reports/quebec/figs")


def main():
    lp.LetsPlot.setup_html()
    rep = f"{_paths.DATA_ROOT}/quebec/rapport"
    reseau = gpd.read_parquet(f"{rep}/reseau.parquet")
    contours = pd.read_parquet(f"{rep}/regions-contours.parquet")
    lignes = []
    modeles = os.environ.get("MEANDRE_MODELES", f"{_paths.DATA_ROOT}/quebec/checkpoints-ref-2026-09-15")
    bras = os.environ.get("MEANDRE_BRAS", "variations")
    for reg in REGIONS:
        fa, fs = f"{modeles}/ref-{reg}-{bras}.npz", f"{modeles}/ref-{reg}-{bras}-sans.npz"
        # Vaudreuil n'a aucune station : son modèle transféré n'a pas été réentraîné.
        if not os.path.exists(fa):
            fa, fs = f"{rep}/rap-{reg}-avec.npz", f"{rep}/rap-{reg}-sans.npz"
        if not (os.path.exists(fa) and os.path.exists(fs)):
            continue
        a, s = np.load(fa, allow_pickle=True), np.load(fs, allow_pickle=True)
        qn = np.clip(s["q_annuel"], 1e-6, None)
        lignes.append(pd.DataFrame({
            "region": reg, "node_idx": np.arange(len(qn)),
            "q_nat": s["q_annuel"], "effet": 100.0 * (a["q_annuel"] - s["q_annuel"]) / qn}))
    d = pd.concat(lignes, ignore_index=True)
    g = reseau.merge(d, on=["region", "node_idx"], how="inner")
    g = g[~g.est_lac].copy()
    g["effet_borne"] = g.effet.clip(-10, 10)
    contexte = g[(g.q_nat > SEUIL_CONTEXTE) & (g.effet.abs() <= SEUIL_EFFET)]
    touche = g[g.effet.abs() > SEUIL_EFFET].sort_values("effet", key=abs)
    print(f"contexte {len(contexte)} tronçons, affectés {len(touche)}")

    p = (lp.ggplot()
         + lp.geom_path(lp.aes("lon", "lat", group="grp"), data=contours,
                        color=pal.GRIS_CLAIR, size=.3)
         + lp.geom_path(data=contexte, color=pal.GRIS_CLAIR, size=.35)
         + lp.geom_path(lp.aes(color="effet_borne"), data=touche, size=1.1)
         + lp.scale_color_gradient2(low=pal.ROUGE, mid=pal.GRIS_PALE, high=pal.BLEU,
                                    midpoint=0, limits=[-10, 10], name="effet (%)")
         + lp.coord_map()
         + lp.labs(x="", y="")
         + lp.theme_minimal()
         + lp.theme(axis_text=lp.element_blank(), panel_grid=lp.element_blank()))
    os.makedirs(SORTIE, exist_ok=True)
    lp.ggsave(p + lp.ggsize(1180, 620), "carte-effet.png", path=SORTIE, scale=2)

    # VERSION ZOOMABLE. Le fond de carte porte deja les rivieres et les lacs, donc la couche
    # de contexte y est inutile : on ne trace que les troncons affectes, 788 chemins au lieu
    # de 23 816. C'est ce nombre qui tuait la version precedente, laquelle n'en dessinait
    # qu'une partie et recadrait la vue sur ce qui passait. L'emprise est imposee pour que
    # la carte s'ouvre sur le Quebec meridional et non sur le premier troncon venu.
    # `location` CENTRE la carte sans la cadrer : avec les deux coins, lets_plot se place sur
    # leur milieu et garde un zoom par defaut, ce qui ouvrait la carte sur le Lac-Saint-Jean.
    # Le niveau se calcule : a un niveau z, une fenetre de 1180 pixels couvre 1659 / 2**z
    # degres de longitude. Il faut couvrir 15,2 degres, donc le niveau 6, qui en couvre 25,9 ;
    # le 7 n'en couvrirait que 13,0.
    bb = [float(touche.total_bounds[0]), float(touche.total_bounds[1]),
          float(touche.total_bounds[2]), float(touche.total_bounds[3])]
    span = bb[2] - bb[0]
    niveau = max(1, min(15, int(np.floor(np.log2(1659.0 / (span * 1.15))))))
    print(f"emprise {span:.1f} degrés de longitude -> niveau de zoom {niveau}")
    pz = (lp.ggplot()
          # `const_size_zoomin` vaut -1 par defaut : l'epaisseur d'un trait a taille
          # CONSTANTE croit sans limite avec le zoom, si bien qu'une riviere devient un
          # pate des qu'on s'approche. A zero, le trait garde son epaisseur en pixels,
          # comme sur n'importe quelle carte en ligne.
          + lp.geom_livemap(location=bb, zoom=niveau, const_size_zoomin=0)
          + lp.geom_path(lp.aes(color="effet_borne"), data=touche, size=1.2)
          + lp.scale_color_gradient2(low=pal.ROUGE, mid=pal.GRIS_PALE, high=pal.BLEU,
                                     midpoint=0, limits=[-10, 10], name="effet (%)")
          + lp.theme_minimal())
    lp.ggsave(pz + lp.ggsize(1180, 620), "carte-effet.html", path=SORTIE, iframe=False)
    print(f"{SORTIE}/carte-effet.html | emprise {[round(x, 2) for x in bb]}")
    h = int((touche.effet > SEUIL_EFFET).sum())
    b = int((touche.effet < -SEUIL_EFFET).sum())
    print(f"{SORTIE}/carte-effet.png")
    print(f"{h} tronçons reçoivent de l'eau, {b} en perdent, "
          f"effet de {g.effet.min():.1f} à {g.effet.max():.0f} pour cent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
