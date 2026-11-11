"""Vérifie les fichiers ESPO-G6 téléchargés : chaque jour doit porter le masque terre-mer attendu
(environ 32 % de mailles sans valeur). Imprime les fichiers invalides ; --delete les supprime pour
que download_espo_g6.py les reprenne.

Usage : python check_espo_files.py <dossier> [--delete]
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

bad = []
files = sorted(Path(sys.argv[1]).glob("espo-g6-r2_*.nc"))
for f in files:
    ds = xr.open_dataset(f, mask_and_scale=False)
    var = list(ds.data_vars)[0]
    n = ds.sizes["time"]
    frac = np.concatenate([(ds[var].isel(time=slice(a, a + 365)).values == -32768).reshape(-1, ds.sizes["rlat"] * ds.sizes["rlon"]).mean(axis=1) for a in range(0, n, 365)])
    ds.close()
    wrong = np.abs(frac - 0.3168) >= 0.01
    if wrong.any():
        years = sorted({int(f.stem.split("_")[-2][:4]) + i // 365 for i in np.flatnonzero(wrong)})
        bad.append(f)
        print(f"{f.name} : {wrong.sum()} jours invalides, années {years[0]}-{years[-1]}", flush=True)
print(f"{len(bad)} fichiers invalides sur {len(files)}")
if "--delete" in sys.argv:
    for f in bad:
        f.unlink()
