"""NEISIM : équivalent en eau de la neige sur grille, échantillonné aux nœuds d'un territoire.

NEISIM est un produit du gouvernement du Québec, sur grille d'environ deux kilomètres et
demi, au pas journalier, de 1980 à 2025. C'est un MODÈLE, pas une mesure, et l'employer
comme cible importe sa structure. La condition préalable a donc été mesurée avant d'écrire
ce chargeur : sur 246 sites du réseau CanSWE et 111 532 couples de valeurs journalières, le
rapport médian par site vaut 0,98 avec des quartiles de 0,85 à 1,29, la corrélation médiane
0,87 et l'erreur absolue moyenne 33 mm, soit 30 % de la moyenne observée de 112 mm. Le banc
qui l'établit est `.runs/quebec/neisim_contre_canswe.py`.

Ce que NEISIM apporte que CanSWE n'a pas est la COUVERTURE. Le réseau au sol compte 76
sites en Outaouais et aucun au Saint-Laurent sud-ouest : là où il se tait, aucune
observation de masse du manteau ne contraint la colonne.

Le découpage interne du fichier est une carte complète par pas de temps. Lire la série d'un
seul point force à décompresser tout le fichier : l'accès efficace est par TRANCHE DE TEMPS.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from meandre.utils import paths as _paths

DEFAULT_FILE = f"{_paths.DATA_ROOT}/neisim/NEISIM_QCMERI_EENEIG_GOENS_QCHRES_24H_MEDIAN.nc"


def _grid(file):
    """Coordonnées de la grille et axe de temps, sans lire le champ."""
    import xarray as xr

    d = xr.open_dataset(file, decode_timedelta=False)
    lon, lat = d.x.values.astype("float64"), d.y.values.astype("float64")
    # NEISIM horodate ses journées à 05 h UTC et les jeux au sol à minuit : sans
    # normalisation à la DATE, l'intersection des deux index est vide.
    time_values = pd.to_datetime(d.time.values).normalize()
    d.close()
    return lon, lat, time_values


def node_cells(node_lon, node_lat, file=None):
    """Cellule de grille la plus proche de chaque nœud, et la liste des cellules distinctes.

    Retourne ``(cellules, rang, lon, lat, temps)`` où ``cellules`` est (n_cellules, 2) en
    indices de grille et ``rang`` (n_noeuds,) donne la cellule de chaque nœud.
    """
    file = file or DEFAULT_FILE
    lon, lat, time_values = _grid(file)
    ilon = np.abs(lon[None, :] - np.asarray(node_lon)[:, None]).argmin(axis=1)
    ilat = np.abs(lat[None, :] - np.asarray(node_lat)[:, None]).argmin(axis=1)
    cells, rank = np.unique(np.stack([ilon, ilat], axis=1), axis=0, return_inverse=True)
    return cells, np.ravel(rank), lon, lat, time_values


def read_cells(cells, file=None, block=512, log=True):
    """Séries d'équivalent en eau aux cellules données, en millimètres, (n_cellules, T)."""
    import netCDF4 as nc

    file = file or DEFAULT_FILE
    root = nc.Dataset(file)
    var = root.variables["een"]
    n_t = var.shape[2]
    out = np.empty((len(cells), n_t), dtype="float32")
    for a0 in range(0, n_t, block):
        a1 = min(a0 + block, n_t)
        out[:, a0:a1] = np.ma.filled(var[:, :, a0:a1], np.nan)[cells[:, 0], cells[:, 1], :]
        if log and a0 % (block * 8) == 0:
            print(f"  neisim {100 * a1 / n_t:3.0f} %", flush=True)
    root.close()
    return out


def load_at_nodes(node_lon, node_lat, times, file=None):
    """Équivalent en eau de NEISIM à chaque nœud, aligné sur l'axe de temps demandé.

    Retourne ``(valeurs, couverture)`` : ``valeurs`` est (T, n_noeuds) en millimètres avec
    NaN hors de la période de NEISIM, ``couverture`` la part de jours renseignés.
    """
    cells, rank, _lon, _lat, time_n = node_cells(node_lon, node_lat, file)
    series = read_cells(cells, file)
    axe = pd.DatetimeIndex(pd.to_datetime(times)).normalize()
    pos = pd.Series(np.arange(len(time_n)), index=time_n)
    pos = pos[~pos.index.duplicated()]
    ou = pos.reindex(axe).to_numpy()
    value_list = np.full((len(axe), len(rank)), np.nan, dtype="float32")
    vu = np.isfinite(ou)
    if vu.any():
        value_list[vu] = series[:, ou[vu].astype(int)].T[:, rank]
    return value_list, float(vu.mean())


def file_present(file=None) -> bool:
    return os.path.exists(file or DEFAULT_FILE)
