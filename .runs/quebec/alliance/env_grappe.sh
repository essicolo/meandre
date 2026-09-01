#!/usr/bin/env bash
# ETAPE 4, sur le noeud de CONNEXION de Fir : batir l'environnement Python une seule fois.
# Les noeuds de calcul n'ont AUCUN acces Internet : tout s'installe ici, depuis la
# logitheque locale de l'Alliance (--no-index), jamais depuis PyPI.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 scipy-stack
VENV=$HOME/venv-meandre
[ -d "$VENV" ] || virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
# torch est fourni compile pour les cartes de la grappe : ne JAMAIS le prendre sur PyPI.
pip install --no-index torch numpy pandas xarray netCDF4 duckdb pyarrow tomli
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda compile:", torch.version.cuda)
PY
echo "Environnement pret : source $VENV/bin/activate"
