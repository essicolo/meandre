"""Combien de tronçons SWOT observe-t-il sur chaque territoire, avant de tout télécharger.

SWOT mesure l'élévation et la pente de la surface libre par tronçon d'environ dix kilomètres,
ce qui est une observation DIRECTE des objets que le modèle route : Manning, les coefficients
de Muskingum et la géométrie ne sont contraints aujourd'hui que par le débit à une quinzaine
de jauges. Mais la mission n'observe de façon fiable que les rivières de plus de cent mètres
de large, et le produit par tronçon pèse une dizaine de gigaoctets sur le Québec méridional.

Ce banc fait le test préalable : quelques granules suffisent à compter les tronçons distincts
par territoire. Sous quelques dizaines, le chantier ne vaut pas le téléchargement.

    .venv/Scripts/python.exe .runs/quebec/swot_troncons_observables.py --granules 8
"""
import argparse
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

warnings.filterwarnings("ignore")
DOSSIER = f"{_paths.DATA_ROOT}/swot"
# Emprises approximatives des territoires, en degrés (ouest, sud, est, nord).
TERRITOIRES = {"outv": (-79.5, 45.0, -73.5, 48.5), "mont": (-74.5, 44.9, -72.0, 46.2),
               "slso": (-76.5, 45.0, -72.5, 47.5), "slno": (-75.0, 46.0, -70.5, 49.0),
               "sagu": (-73.5, 47.5, -69.5, 50.5), "gasp": (-68.5, 47.5, -64.0, 49.5)}
LARGEUR_MIN = 100.0


def granules(n, debut, fin):
    """Granules du produit par tronçon qui coupent le Québec méridional."""
    import earthaccess

    earthaccess.login(strategy="netrc")
    return earthaccess.search_data(short_name="SWOT_L2_HR_RiverSP_reach_D",
                                   bounding_box=(-80.0, 45.0, -64.0, 51.0),
                                   temporal=(debut, fin), count=n)


def _lire(chemin):
    """Table des tronçons d'un granule, quelle que soit sa mise en boîte."""
    import geopandas as gpd

    chemin = str(chemin)
    if chemin.endswith(".zip"):
        with zipfile.ZipFile(chemin) as z:
            shp = [n for n in z.namelist() if n.endswith(".shp")]
        if not shp:
            return None
        return gpd.read_file(f"zip://{chemin}!{shp[0]}")
    return gpd.read_file(chemin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--granules", type=int, default=8)
    p.add_argument("--debut", default="2024-05-01")
    p.add_argument("--fin", default="2024-07-31")
    a = p.parse_args()
    import earthaccess

    os.makedirs(DOSSIER, exist_ok=True)
    g = granules(a.granules, a.debut, a.fin)
    print(f"{len(g)} granules retenus, telechargement dans {DOSSIER}", flush=True)
    fichiers = earthaccess.download(g, DOSSIER)

    vus = {k: set() for k in TERRITOIRES}
    larges = {k: set() for k in TERRITOIRES}
    n_lus = 0
    for f in fichiers:
        t = _lire(f)
        if t is None or t.empty:
            continue
        n_lus += 1
        # Les coordonnees du tronçon sont portees par p_lon et p_lat dans le produit par
        # tronçon ; a defaut on retombe sur le centroide de la geometrie.
        if "p_lon" in t.columns:
            lon, lat = t["p_lon"].to_numpy(float), t["p_lat"].to_numpy(float)
        else:
            c = t.geometry.centroid
            lon, lat = c.x.to_numpy(), c.y.to_numpy()
        ids = t["reach_id"].astype(str).to_numpy()
        # `p_width` est la largeur a priori de la base de rivieres, en metres.
        w = t["p_width"].to_numpy(float) if "p_width" in t.columns else np.full(len(t), np.nan)
        for reg, (o, s, e, n) in TERRITOIRES.items():
            dedans = (lon >= o) & (lon <= e) & (lat >= s) & (lat <= n)
            vus[reg].update(ids[dedans])
            larges[reg].update(ids[dedans & (w >= LARGEUR_MIN)])
    print(f"{n_lus} granules lus\n")
    print(f"{'territoire':<12s} {'tronçons vus':>13s} {'dont >= 100 m':>15s}")
    for reg in TERRITOIRES:
        print(f"{reg:<12s} {len(vus[reg]):>13d} {len(larges[reg]):>15d}")
    print("")
    print("Ces comptes viennent de quelques passes seulement : ils donnent la DENSITE, pas")
    print("le total. Le total se deduit en couvrant le cycle de 21 jours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
