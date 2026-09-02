#!/usr/bin/env bash
# BANC 1.1 : le socle (temoin) contre le socle avec les sorties mortes REBRANCHEES.
#
# Constat qui fonde ce banc (registre R59, 2026-09-02) : sept sorties du champ spatial ne
# sont lues par aucune equation. Le reseau depense de la capacite a les predire et leurs
# priors tirent sur le tronc partage. Trois meritent d'etre branchees -- rain_hours (duree
# de l'averse, dont depend le ruissellement hortonien que le registre accuse du deficit
# d'ete), vsa_b (aire contributive saturee) et le champ de canopee -- et quatre doivent
# disparaitre : manning_n, frost_alpha, alpha_T, f_wetland.
#
# NE PAS confondre avec une "liberation" des champs imposes : cette voie-la est DEJA
# REFUTEE. Le 2026-08-15, n'imposer que la courbe de retention rendait 0.592 contre 0.739
# pour le socle. On ne relache pas l'imposition, on rebranche ce qui est debranche.
#
# Douze taches : six regions x deux variantes, appariees (meme graine, meme forcage, meme
# recette). Seule la variante change, donc l'ecart se lit directement.
#
#   sbatch --account=<compte> --array=0-11 v11.sbatch
#SBATCH --job-name=v11
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
  VARIANTE=temoin; V11=0
else
  VARIANTE=rebranche; V11=1
fi
JOURNAUX=$TRAVAIL/meandre/v11
mkdir -p "$JOURNAUX" "$RACINE/.runs/quebec/checkpoints"
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
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=${JOINT_FX_SUFFIX:--hyb} ETL_FORCE=1 \
    ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-1} ETL_V11=$V11 \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-v11$VARIANTE ETL_EPOCHS=${ETL_EPOCHS:-30} \
    python -u .runs/quebec/etl_run.py
