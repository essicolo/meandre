"""Télécharge les sorties journalières du MRCC5 piloté par CMIP6 (ensemble d'Ouranos, grille NAM-12,
environ 12 km), découpées à une boîte couvrant le Québec méridional, par OPeNDAP depuis le serveur
THREDDS d'Ouranos (sans authentification).

Variables : celles qu'il faut pour les six entrées de la colonne (précipitation, températures minimale
et maximale, rayonnements descendants et montants de courtes et grandes longueurs d'onde pour le
rayonnement net, vent à 10 m, humidité spécifique pour la pression de vapeur).

Un fichier par fichier source de cinq ans, compressé ; un fichier déjà présent est sauté (reprise).

Usage :
    python download_crcm5_cmip6.py <dossier de sortie> <pilote> <expérience> <première année> <dernière année>
    python download_crcm5_cmip6.py /data/crcm5 ERA5 evaluation 1991 2020
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

import xarray as xr

ROOT = "https://esgf.ouranos.ca/thredds"
BASE = "mrcc5-cmip6/CORDEX-CMIP6/DD/NAM-12/OURANOS"
VARIABLES = ["pr", "tasmin", "tasmax", "rsds", "rsus", "rlds", "rlus", "sfcWind", "huss"]
# Boîte du Québec méridional (44,5 à 63° N, 80 à 57° O) dans la grille tournée NAM-12, mesurée sur un
# fichier du 2026-11-11 : 8,5 % du domaine.
RLAT, RLON = slice(291, 490), slice(382, 558)


def links(path):
    xml = urllib.request.urlopen(f"{ROOT}/catalog/{path}/catalog.xml", timeout=60).read().decode()
    return re.findall(r'xlink:href="([^"]+)/catalog.xml"', xml), re.findall(r'urlPath="([^"]+\.nc)"', xml)


def leaf(driver, experiment, variable):
    """Chemin du dossier de version le plus récent d'une variable journalière."""
    path = f"{BASE}/{driver}/{experiment}"
    for _ in range(3):  # membre, modèle régional, version de la simulation
        path = f"{path}/{links(path)[0][0]}"
    path = f"{path}/day/{variable}"
    return f"{path}/{sorted(links(path)[0])[-1]}"


def main():
    out, driver, experiment, y0, y1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    target = Path(out) / driver / experiment
    target.mkdir(parents=True, exist_ok=True)
    for variable in VARIABLES:
        for url_path in links(leaf(driver, experiment, variable))[1]:
            a, b = (int(x[:4]) for x in re.search(r"_(\d{8})-(\d{8})\.nc$", url_path).groups())
            if b < y0 or a > y1:
                continue
            dest = target / Path(url_path).name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            t0 = time.time()
            for attempt in range(4):
                try:
                    ds = xr.open_dataset(f"{ROOT}/dodsC/{url_path}").isel(rlat=RLAT, rlon=RLON)
                    part = dest.with_suffix(".part")
                    ds.load().to_netcdf(part, encoding={variable: {"zlib": True, "complevel": 4}})
                    part.replace(dest)
                    break
                except Exception as e:  # coupure réseau : on réessaie
                    print(f"{dest.name} : essai {attempt + 1} échoué ({e})", flush=True)
                    time.sleep(30)
            print(f"{dest.name} : {dest.stat().st_size / 1e6:.0f} Mo en {time.time() - t0:.0f} s" if dest.exists() else f"{dest.name} : ABANDON", flush=True)


if __name__ == "__main__":
    main()
