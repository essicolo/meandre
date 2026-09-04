#!/bin/bash
# Relance CaSR apres reboot Windows (GPU debloque). Synchronise le cache CaSR
# deduplique (cote Windows) vers le natif WSL, puis lance l'entrainement.
#
# Usage (depuis PowerShell Windows, ou WSL) :
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/parse01/documents-locaux/GitHub/meandre/.runs/slso/_run_casr.sh casr
#   wsl -d Ubuntu-22.04 -- bash .../meandre/.runs/slso/_run_casr.sh casr-penman
# Defaut = casr (McGuinness, test equitable controle). casr-penman = Penman/CaSR.
set -e
cd /mnt/c/Users/parse01/documents-locaux/GitHub/meandre
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

VARIANT="${1:-casr}"   # casr | casr-penman | casr-eti
CFG=".runs/slso/config/slso-physitel-hydrotel-${VARIANT}.toml"
LOG=".runs/slso/_casr_${VARIANT}_run.log"

# 1) Sync du cache CaSR (riox : mosaïque 2x2 + reproj bilinéaire, couverture complète)
#    vers le natif WSL (9p /mnt/c tue duckdb/netcdf). Le cache = celui que la config
#    pointe (6 canaux par défaut, 8 canaux -eb pour la fonte ETI).
mkdir -p /home/essi/slso-data
CACHE=$(grep -m1 '^forcing_cache' "$CFG" | sed -E 's/.*"([^"]+)".*/\1/' | xargs basename)
SRC=".runs/slso/data/${CACHE}"
DST="/home/essi/slso-data/${CACHE}"
echo "sync cache CaSR ($CACHE) -> $DST"
cp -f "$SRC" "$DST"
DST="$DST" .venv-wsl/bin/python - <<'PY'
import os, xarray as xr, pandas as pd, numpy as np
d = xr.open_dataset(os.environ["DST"])
t = pd.to_datetime(d["time"].values); f = d["forcing"].values
print(f"cache WSL : T={d.sizes['time']} var={d.sizes.get('var')} doublons={int(t.duplicated().sum())} NaN={bool(np.isnan(f).any())} {t.min().date()}..{t.max().date()}")
assert d.sizes["time"] == 9132 and int(t.duplicated().sum()) == 0 and not np.isnan(f).any(), "cache CaSR invalide !"
print("[ok] cache CaSR valide (9132 j, 0 doublon, 0 NaN)")
PY

# 2) Lancement entrainement
echo "lancement $CFG -> $LOG"
rm -f "$LOG"
exec .venv-wsl/bin/python .runs/slso/slso.py "$CFG" > "$LOG" 2>&1
