"""Télécharge ESPO-G6-R2 v1.0.0 (Ouranos : CMIP6 réduit et ajusté sur RDRS v2.1, P, Tmin, Tmax journaliers,
calendrier sans 29 février) découpé à l'emprise des tronçons du domaine de meandre, par OPeNDAP depuis
PAVICS (sans authentification).

Périodes : référence 1991-2020 (prise du jeu SSP2-4.5 ; l'historique s'arrête en 2014 et les deux scénarios
ne diffèrent qu'à partir de 2015), puis 2041-2070 et 2071-2100 pour SSP2-4.5 et SSP3-7.0. Ordre : la
référence de tous les modèles globaux, puis le milieu du siècle, puis la fin, pour qu'un arrêt laisse des
périodes complètes.

Un fichier par modèle global, expérience, période et variable ; précipitation en mm/j et températures en
°C, stockées en entiers à 0,01 près (compression). Un fichier déjà présent est sauté (reprise).

Usage : python download_espo_g6.py <dossier de sortie> [k n]
(k et n : ce processus ne traite que les modèles globaux de rang k modulo n, pour paralléliser)
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds"
PATH = "datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0"
# Mailles de la grille RDRS v2.1 couvrant les 27 834 tronçons avec 0,2° de marge (mesuré le 2026-11-11 :
# chaque tronçon a une maille à moins de 0,08°).
RLAT, RLON = slice(353, 522), slice(466, 684)
PERIODS = [("ssp245", "reference", 1991, 2020), ("ssp245", "ssp245", 2041, 2070), ("ssp370", "ssp370", 2041, 2070), ("ssp245", "ssp245", 2071, 2100), ("ssp370", "ssp370", 2071, 2100)]
VARIABLES = {"pr": (86400.0, 0.0), "tasmin": (1.0, -273.15), "tasmax": (1.0, -273.15)}


def valid(values):
    """Un bloc est valide si chaque jour porte le masque terre-mer attendu (environ 32 % de mailles
    sans valeur dans la boîte) : le serveur renvoie parfois un bloc entier de zéros sans erreur
    (constaté le 2026-11-11 : tasmax 1991-1995 et pr 2011-2015 de BCC-CSM2-MR)."""
    nan_frac = np.isnan(values).reshape(values.shape[0], -1).mean(axis=1)
    return bool((np.abs(nan_frac - 0.3168) < 0.01).all())


def block(ds, var, a, b):
    """Bloc d'années a à b, redemandé jusqu'à cinq fois s'il est invalide."""
    for attempt in range(5):
        x = ds[[var]].isel(time=slice((a - 1950) * 365, (b - 1950 + 1) * 365), rlat=RLAT, rlon=RLON).load()
        if valid(x[var].values):
            return x
        print(f"{var} {a}-{b} : bloc invalide, nouvelle requête", flush=True)
        time.sleep(30)
    raise RuntimeError(f"{var} {a}-{b} : bloc invalide après cinq requêtes")


def datasets():
    xml = urllib.request.urlopen(f"{ROOT}/catalog/{PATH}/catalog.xml", timeout=120).read().decode()
    return re.findall(r'urlPath="([^"]+\.ncml)"', xml)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    found = datasets()
    models = sorted({re.search(r"ScenarioMIP_NAM_(.+?)_ssp\d+_", p).group(1) for p in found})
    by_key = {(re.search(r"ScenarioMIP_NAM_(.+?)_ssp\d+_", p).group(1), re.search(r"_(ssp\d+)_", p).group(1)): p for p in found}
    if len(sys.argv) > 3:
        k, n = int(sys.argv[2]), int(sys.argv[3])
        models = models[k::n]
    print(f"{len(models)} modèles globaux", flush=True)
    for source, label, y0, y1 in PERIODS:
        for model in models:
            if (model, source) not in by_key:
                continue
            for var, (scale, offset) in VARIABLES.items():
                dest = out / f"espo-g6-r2_{model}_{label}_{y0}-{y1}_{var}.nc"
                if dest.exists():
                    continue
                t0 = time.time()
                for attempt in range(4):
                    try:
                        ds = xr.open_dataset(f"{ROOT}/dodsC/{by_key[(model, source)]}", decode_times=False)
                        # Requêtes de cinq ans : une seule de trente ans dépasserait la limite du serveur.
                        blocks = [block(ds, var, a, min(a + 4, y1)) for a in range(y0, y1 + 1, 5)]
                        sub = xr.concat(blocks, dim="time").drop_vars("time")
                        sub[var] = sub[var] * scale + offset
                        sub[var].attrs = {"units": "mm/d" if var == "pr" else "degC"}
                        sub.attrs = {"source": by_key[(model, source)], "calendar": "noleap", "first_day": f"{y0}-01-01", "n_days": sub.sizes["time"]}
                        part = dest.with_suffix(".part")
                        sub.to_netcdf(part, encoding={var: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4}})
                        part.replace(dest)
                        break
                    except Exception as e:
                        print(f"{dest.name} : essai {attempt + 1} échoué ({e})", flush=True)
                        time.sleep(60)
                print(f"{dest.name} : {dest.stat().st_size / 1e6:.0f} Mo en {time.time() - t0:.0f} s" if dest.exists() else f"{dest.name} : ABANDON", flush=True)


if __name__ == "__main__":
    main()
