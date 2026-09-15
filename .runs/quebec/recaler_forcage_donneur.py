"""Recale la precipitation d'une region SANS station, avec le facteur d'une region donneuse.

Le recalage de volume ferme le bilan P = Q + ET(P) sur l'ecoulement OBSERVE aux stations.
Une region sans station n'a donc pas de cible : le 2026-09-15, la tentative sur Vaudreuil,
qui ne porte ni station ni observation, a produit une precipitation entierement non definie,
sur 1 835 532 valeurs, et le fichier a ete supprime.

Or si le MODELE d'une telle region est transfere, son facteur de recalage peut l'etre aussi.
Le donneur est choisi par la distance de Gower sur les attributs du territoire, qui est la
regle deja employee dans le depot pour le choix des donneurs de champion. Pour Vaudreuil
elle designe la Monteregie a 0,202, devant le Saint-Laurent sud-ouest a 0,279.

    .venv/Scripts/python.exe .runs/quebec/recaler_forcage_donneur.py vaud mont
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths


def facteur(reg):
    """Facteur effectivement applique a une region deja recalee."""
    racine = _paths.DATA_ROOT
    a = f"{racine}/quebec/forcing-{reg}.nc"
    b = f"{racine}/quebec/forcing-{reg}-budyko.nc"
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    pa = float(xr.open_dataset(a).forcing[:, :, 0].mean())
    pb = float(xr.open_dataset(b).forcing[:, :, 0].mean())
    return pb / pa


def main(reg, donneur):
    racine = _paths.DATA_ROOT
    f = facteur(donneur)
    if f is None or not np.isfinite(f):
        print(f"le donneur {donneur} n'a pas de forçage recalé")
        return 1
    src = f"{racine}/quebec/forcing-{reg}.nc"
    if not os.path.exists(src):
        print(f"forçage brut absent : {src}")
        return 1
    d = xr.open_dataset(src)
    F = d["forcing"].values.copy()
    t = d["time"].values
    V = list(d["var"].values.astype(str))
    d.close()
    avant = float(np.nanmean(F[:, :, 0])) * 365.25
    F[:, :, 0] = (F[:, :, 0] * f).astype(np.float32)
    apres = float(np.nanmean(F[:, :, 0])) * 365.25
    if not np.isfinite(apres) or apres <= 0:
        print("ARRÊT : la précipitation recalée n'est pas finie, rien n'est écrit")
        return 2
    out = f"{racine}/quebec/forcing-{reg}-budyko.nc"
    xr.Dataset({"forcing": (("time", "node", "var"), F)},
               coords={"time": t, "node": np.arange(F.shape[1]), "var": V}).to_netcdf(out)
    print(f"[{reg}] facteur emprunté à {donneur} : {f:.4f} | "
          f"P {avant:.0f} -> {apres:.0f} mm/an | {out}")
    print(f"[{reg}] valeurs finies : {int(np.isfinite(F[:, :, 0]).sum())} sur {F[:, :, 0].size}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "vaud",
                  sys.argv[2] if len(sys.argv) > 2 else "mont"))
