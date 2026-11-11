"""Copie corrigée des forçages CaSR : le vent converti de nœuds en m/s (2026-11-11).

Le vent de CaSR (P_UVC_10m) est fourni en nœuds ; build_forcing_region.py le multipliait par 0,748
(passage de 10 m à 2 m) sans le convertir, si bien que u2 valait 1,94 fois le vent réel et que l'ETP
de Penman était surestimée d'environ 11 % par an (Montérégie, 2010-2019). Ce script écrit, pour chaque
territoire, forcing-<territoire><suffixe><ajout>.nc où u2 est multiplié par 0,514444 ; le fichier
d'origine n'est pas modifié.

Usage : python -m meandre.adapters.physitel.builders.fix_forcing_wind <dossier quebec> [suffixe] [ajout]
        (par défaut -casr-brut et -vent)
"""
import glob
import os
import sys

import xarray as xr

KNOT_TO_MS = 0.514444


def main():
    folder = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else "-casr-brut"
    extra = sys.argv[3] if len(sys.argv) > 3 else "-vent"
    for src in sorted(glob.glob(f"{folder}/forcing-*{suffix}.nc")):
        dst = src[:-3] + extra + ".nc"
        if os.path.exists(dst):
            print(f"{os.path.basename(dst)} existe, sauté")
            continue
        ds = xr.open_dataset(src).load()
        i = list(ds["var"].values).index("u2")
        before = float(ds.forcing[..., i].mean())
        ds["forcing"][..., i] = ds.forcing[..., i] * KNOT_TO_MS
        part = dst + ".part"
        ds.to_netcdf(part)
        os.replace(part, dst)
        print(f"{os.path.basename(dst)} : u2 moyen {before:.2f} -> {before * KNOT_TO_MS:.2f} m/s")


if __name__ == "__main__":
    main()
