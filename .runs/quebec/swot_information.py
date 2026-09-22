"""Combien de directions indépendantes l'altimétrie SWOT porte-t-elle, et qu'ajoute-t-elle.

Mille huit tronçons observés ne font pas mille huit contraintes : l'élévation de deux
tronçons voisins de la même rivière est presque la même variable. La grandeur qui décide est
le nombre de directions indépendantes, et surtout l'apport PROPRE au reste des observations,
mesuré comme le rang effectif de l'ensemble moins celui de l'ensemble privé de la source.

Deux grandeurs de SWOT sont lues : l'élévation de la surface libre, `wse`, et la pente,
`slope`. La pente est le lien direct avec Manning et la géométrie du tronçon ; l'élévation
porte la dynamique de crue.

    .venv/bin/python .runs/quebec/swot_information.py outv --passe 326
"""
import argparse
import glob
import os
import re
import sys
import warnings
import zipfile
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

warnings.filterwarnings("ignore")
_ici = os.path.dirname(os.path.abspath(__file__))
_inf = SourceFileLoader("inf", os.path.join(_ici, "information_de_la_neige.py")).load_module()

DOSSIER = f"{_paths.DATA_ROOT}/swot"
EMPRISES = {"outv": (-79.5, 45.0, -73.5, 48.5), "slso": (-76.5, 45.0, -72.5, 47.5),
            "slno": (-75.0, 46.0, -70.5, 49.0), "sagu": (-73.5, 47.5, -69.5, 50.5),
            "mont": (-74.5, 44.9, -72.0, 46.2), "gasp": (-68.5, 47.5, -64.0, 49.5)}
# Valeur de remplissage du produit, qui vaut sinon des milliers de metres.
REMPLISSAGE = 1e10
LARGEUR_MIN = 100.0


def _table(chemin):
    import geopandas as gpd

    with zipfile.ZipFile(chemin) as z:
        shp = [n for n in z.namelist() if n.endswith(".shp")]
    return gpd.read_file(f"zip://{chemin}!{shp[0]}") if shp else None


def series(reg, passe, grandeur):
    """Matrice dates x tronçons pour une grandeur, sur une passe et un territoire."""
    o, s, e, n = EMPRISES[reg]
    lignes, colonnes = {}, set()
    for f in sorted(glob.glob(f"{DOSSIER}/*_{passe}_NA_*.zip")):
        m = re.search(r"_(\d{8})T\d{6}_", os.path.basename(f))
        if not m:
            continue
        t = _table(f)
        if t is None or t.empty or grandeur not in t.columns or "p_lon" not in t.columns:
            continue
        lo, la = t["p_lon"].to_numpy(float), t["p_lat"].to_numpy(float)
        w = t["p_width"].to_numpy(float) if "p_width" in t.columns else np.zeros(len(t))
        v = t[grandeur].to_numpy(float)
        v = np.where(np.abs(v) >= REMPLISSAGE * 0.1, np.nan, v)
        garde = (lo >= o) & (lo <= e) & (la >= s) & (la <= n) & (w >= LARGEUR_MIN) & np.isfinite(v)
        if not garde.any():
            continue
        ids = t["reach_id"].astype(str).to_numpy()[garde]
        d = pd.Timestamp(m.group(1))
        lignes.setdefault(d, {}).update(dict(zip(ids, v[garde])))
        colonnes.update(ids)
    if not lignes:
        return None
    cols = sorted(colonnes)
    dates = sorted(lignes)
    mat = np.full((len(dates), len(cols)), np.nan)
    rang = {c: i for i, c in enumerate(cols)}
    for i, d in enumerate(dates):
        for c, v in lignes[d].items():
            mat[i, rang[c]] = v
    return pd.DataFrame(mat, index=pd.DatetimeIndex(dates), columns=cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--passe", default="326")
    a = p.parse_args()
    reg = a.region.lower()

    print(f"{reg}, passe {a.passe}\n")
    print(f"{'grandeur':<22s} {'tronçons':>9s} {'visites':>8s} {'95 %':>7s} "
          f"{'participation':>14s} {'reproductible':>14s} {'hasard':>8s}")
    gardees = {}
    for g in ("wse", "slope"):
        d = series(reg, a.passe, g)
        if d is None:
            print(f"{g:<22s} {'absent':>9s}")
            continue
        gardees[g] = d
        # Pas MENSUEL, comme pour toutes les autres sources : la passe revient tous les
        # vingt-et-un jours, donc une visite par mois en moyenne.
        axe = pd.date_range(d.index.min().normalize(), d.index.max().normalize(), freq="D")
        plein = np.ones(len(axe), dtype=bool)
        brut = d.reindex(axe).to_numpy()
        mm = _inf._mensualise(brut, axe, plein)
        n95, part, cols = _inf._rang_effectif(mm, recouvrement_min=6)
        rep, pp = _inf._reproductibilite(mm, k=3, recouvrement_min=4)
        n95_t = f"{n95:>7d}" if np.isfinite(n95) else f"{'—':>7s}"
        part_t = f"{part:>14.1f}" if np.isfinite(part) else f"{'—':>14s}"
        rep_t = f"{rep:>14.2f}" if np.isfinite(rep) else f"{'—':>14s}"
        haz = f"{3.0 / pp:>8.2f}" if pp and np.isfinite(rep) else f"{'—':>8s}"
        # TEMOIN DE BRUIT. Avec quarante-quatre visites seulement, du bruit independant
        # gonfle le rang effectif a lui seul : on refait la mesure sur une matrice de meme
        # forme et de memes trous, remplie de bruit. Ce que le bruit obtient est le plancher
        # au-dessous duquel le chiffre ne dit rien.
        rng = np.random.default_rng(1234)
        faux = np.where(np.isfinite(mm), rng.standard_normal(mm.shape), np.nan)
        n_bruit, part_bruit, _ = _inf._rang_effectif(faux, recouvrement_min=6)
        rep_bruit, _pb = _inf._reproductibilite(faux, k=3, recouvrement_min=4)
        nb_t = f"{n_bruit:>7d}" if np.isfinite(n_bruit) else f"{'—':>7s}"
        pb_t = f"{part_bruit:>9.1f}" if np.isfinite(part_bruit) else f"{'—':>9s}"
        rb_t = f"{rep_bruit:>10.2f}" if np.isfinite(rep_bruit) else f"{'—':>10s}"
        print(f"{g:<22s} {d.shape[1]:>9d} {d.shape[0]:>8d} {n95_t} {part_t} {rep_t} {haz}")
        print(f"{'  temoin de bruit':<22s} {'':>9s} {'':>8s} {nb_t} {pb_t:>14s} "
              f"{rb_t:>14s} {haz}")
    if gardees:
        d = list(gardees.values())[0]
        print("")
        print(f"periode couverte : {d.index.min().date()} a {d.index.max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
