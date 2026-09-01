#!/usr/bin/env bash
# ETAPE 4, sur le noeud de CONNEXION de Fir : batir l'environnement Python une seule fois.
# Les noeuds de calcul n'ont AUCUN acces Internet : tout s'installe ici, depuis la
# logitheque locale de l'Alliance (--no-index), jamais depuis PyPI.
set -euo pipefail
module purge
# arrow fournit pyarrow : a l'Alliance il vient d'un MODULE, jamais de pip (la roue de
# pypi est un leurre qui echoue avec un message explicatif). Le module doit etre charge
# AVANT l'activation de l'environnement, sinon il reste invisible.
module load StdEnv/2023 gcc arrow python/3.11 scipy-stack
VENV=$HOME/venv-meandre
[ -d "$VENV" ] || virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
# torch est fourni compile pour les cartes de la grappe : ne JAMAIS le prendre sur PyPI.
# mpi4py : la roue netCDF4 de l'Alliance est compilee avec le support MPI et refuse de
# s'importer sans lui. L'erreur apparait tard, au premier ouverture de forcage.
pip install --no-index torch numpy pandas xarray netCDF4 mpi4py duckdb tomli
python - <<'PY'
import torch
import pyarrow, netCDF4, xarray, duckdb
print("netCDF4", netCDF4.__version__)
print("torch", torch.__version__, "| cuda:", torch.version.cuda,
      "| pyarrow", pyarrow.__version__)
PY
echo "Environnement pret : source $VENV/bin/activate"
