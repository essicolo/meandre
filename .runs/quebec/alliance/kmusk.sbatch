#!/usr/bin/env bash
# BANC APPARIE DU ROUTAGE : six regions entrainees DEUX FOIS, avec et sans l'echelle
# physique de la constante de Muskingum. Trente epoques, meme graine, meme forcage, meme
# recette ; un seul changement.
#
# Pourquoi ce banc (2026-09-02). Le retard de la simulation croit avec le nombre de
# troncons traverses et pas avec le versant. La constante de Muskingum est bornee
# [4, 48] h alors qu'un troncon de 3 a 5 km est parcouru en 1 a 3 h : la borne inferieure
# excede deja le temps physique. Posee sur l'echelle du troncon, la correction supprime
# le retard, mais en INFERENCE elle donne +0.17 de KGE median sur SLSO et -0.05 sur OUTV :
# les parametres avaient ete ajustes en presence du routage lent et compensaient son
# lissage par une production plus vive. Seul un reentrainement tranche.
#
# Douze taches : indices 0-5 temoin, 6-11 echelle physique.
#   sbatch --account=<compte> --array=0-11 kmusk.sbatch
#SBATCH --job-name=kmusk
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
if [ -z "${PROJET:-}" ]; then
  for _c in "$HOME"/projects/*/"$USER"; do
    [ -d "$_c/meandre/donnees" ] && PROJET="$_c" && break
  done
  PROJET=${PROJET:-$HOME/projects/$(id -gn)/$USER}
fi
TRAVAIL=${TRAVAIL:-$HOME/scratch}
REGIONS=(outv gasp sagu mont slno slso)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 6))]}
if [ $((SLURM_ARRAY_TASK_ID / 6)) -eq 0 ]; then
  VARIANTE=temoin; PHYS=0
else
  VARIANTE=physique; PHYS=1
fi
JOURNAUX=$TRAVAIL/meandre/kmusk
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG-$VARIANTE.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/slso"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D/forcing-casr-corr.nc" "$LOCAL/slso/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

MAJ=$(echo "$REG" | tr a-z A-Z)
AUX=(ETL_WET=0.4)
[ "$REG" = slso ] && AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0)
env MEANDRE_NSUBSTEP=64 MEANDRE_KMUSK_PHYS=$PHYS \
    JOINT_FX_SUFFIX=${JOINT_FX_SUFFIX:--hyb} ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-km$VARIANTE ETL_EPOCHS=${ETL_EPOCHS:-30} \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-km${VARIANTE}.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
