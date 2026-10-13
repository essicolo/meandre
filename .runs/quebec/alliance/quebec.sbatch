#!/usr/bin/env bash
# LE QUEBEC ENTIER : un seul modele, un seul champ spatial, les quatorze territoires fondus en un
# domaine (ETL_REGION=quebec, mode domaine du pilote). La recette vient entierement du TOML ;
# seuls restent au lanceur le mode, le recalcul par segments, les epoques et le point de reprise.
#
#   Mesure (un pas d'entrainement, pic de memoire et duree d'un bloc), deux variantes :
#   sbatch --account=def-atlas01 --array=0-1 --time=03:00:00 --export=ALL,MODE=sonde quebec.sbatch
#     indice 0 : sans recalcul par segments ; indice 1 : segments de 10 jours
#   Entrainement :
#   sbatch --account=def-atlas01 --export=ALL,MODE=entrainement,ETL_EPOCHS=8,SEGMENTS=0 quebec.sbatch
#SBATCH --job-name=meandre-quebec
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000M
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
PROJET=${PROJET:-$HOME/projects/def-atlas01/$USER}
TRAVAIL=${TRAVAIL:-$HOME/scratch}
JOURNAUX=$TRAVAIL/meandre/journaux
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints"
MODE=${MODE:-sonde}
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$MODE" = sonde ]; then
  SEGS=(0 10); SEGMENTS=${SEGS[$IDX]}
fi
SEGMENTS=${SEGMENTS:-0}
TAG=${ETL_TAG:--quebec1013}
exec > "$JOURNAUX/quebec-$MODE-seg$SEGMENTS$TAG-${SLURM_JOB_ID}.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"
mkdir -p .runs/quebec/checkpoints

# Donnees des quatorze territoires sur le disque local du noeud ; le cache du forcage fondu
# (memmap de 5,6 Go) s'y ecrit aussi. Les projets PHYSITEL restent dans l'espace de projet.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/derives/auxiliaires"
D=$PROJET/meandre/donnees
for R in abit cnda cndb cndc cndd cnde gasp labi mont outm outv sagu slno slso; do
  cp "$D/quebec/$R.duckdb" "$D/quebec/forcing-$R-casr-brut.nc" "$LOCAL/quebec/"
  cp "$D"/derives/auxiliaires/*-"$R".* "$LOCAL/derives/auxiliaires/" 2>/dev/null || true
done
cp "$D"/quebec/*.parquet "$LOCAL/quebec/"
cp -r "$D/quebec/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/*.parquet "$LOCAL/derives/auxiliaires/"
export MEANDRE_DATA="$LOCAL" MEANDRE_DERIVES="$LOCAL/derives" MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEANDRE_DIAG_CPU=1
COMMUN=(ETL_CONFIG=.runs/quebec/config/socle-2026-10-13.toml ETL_REGION=quebec
        JOINT_FX_SUFFIX=-casr-brut ETL_SEED=1234 ETL_FORCE=1 MEANDRE_SEGMENT_GRADIENT=$SEGMENTS
        ETL_HELDOUT=2019-01-01,2024-12-31)
echo "=== quebec $MODE, segments $SEGMENTS, $(date +%T) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$MODE" = sonde ]; then
  env "${COMMUN[@]}" ETL_EPOCHS=1 ETL_TAG=$TAG-sonde MEANDRE_SONDE_SEGMENT=1 \
      python -u .runs/quebec/etl_run.py
else
  env "${COMMUN[@]}" ETL_EPOCHS=${ETL_EPOCHS:-8} ETL_TAG=$TAG MEANDRE_SAUVER_EPOQUES=1 \
      ETL_DUMP_Q="$TRAVAIL/meandre/q-quebec$TAG.npz" \
      python -u .runs/quebec/etl_run.py
  cp -f .runs/quebec/checkpoints/best-quebec-etl${TAG}*.pt .runs/quebec/checkpoints/best-quebec-etl${TAG}*.recette.toml \
        "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
fi
echo "=== quebec fini $(date +%T) ==="
