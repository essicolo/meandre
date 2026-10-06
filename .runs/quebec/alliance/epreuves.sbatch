#!/usr/bin/env bash
# EPREUVES AVANT LA RONDE PROVINCIALE : ce que le banc de sous-bassin a etabli tient-il au
# pilote regional, entraine en entier ? Un affinage court depuis le modele regional ne peut
# pas le dire (une epoque, meme a 1e-5, le fait derailler, registre R301).
#
#   sbatch --account=def-atlas01_gpu --array=0-11 epreuves.sbatch
#
# Douze taches, seize epoques chacune, deux graines par variante, code corrige du 2026-10-06
# (prelevements dans l'entrainement, reference du terme MOD16 posee au debut de l'epoque) :
#   0-5   Outaouais : recette ; plus depots quaternaires ; plus depots et tete du temps de transfert
#   6-11  Monteregie : recette ; plus depots ; plus depots, drainage agricole et tete
# Chaque tache finit par une passe avant sur la DERNIERE epoque qui exporte les series aux
# stations (ETL_DUMP_Q), jugees sur le KGE 2022-2024, le volume, le Q7min et l'ete.
#SBATCH --job-name=meandre-epreuves
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
if [ -z "${PROJET:-}" ]; then
  for _c in "$HOME"/projects/*/"$USER"; do
    [ -d "$_c/meandre/donnees" ] && PROJET="$_c" && break
  done
fi
TRAVAIL=${TRAVAIL:-$HOME/scratch}
SORTIE=$TRAVAIL/meandre/epreuves
mkdir -p "$TRAVAIL/meandre/journaux" "$SORTIE"

TACHES=(
  "outv temoin 1234" "outv temoin 777" "outv depots 1234" "outv depots 777" "outv depots-tete 1234" "outv depots-tete 777"
  "mont temoin 1234" "mont temoin 777" "mont depots 1234" "mont depots 777" "mont depots-drain-tete 1234" "mont depots-drain-tete 777"
)
read -r REG VAR GRAINE <<< "${TACHES[$SLURM_ARRAY_TASK_ID]}"
TAG=-ep1006-$VAR-g$GRAINE
EPOQUES=${ETL_EPOCHS:-16}
exec > "$TRAVAIL/meandre/journaux/$REG$TAG-$SLURM_ARRAY_JOB_ID.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"
mkdir -p .runs/quebec/checkpoints

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/derives/auxiliaires"
D=$PROJET/meandre/donnees
cp "$D/quebec/${REG}.duckdb" "$D/quebec/forcing-${REG}-casr-brut.nc" "$LOCAL/quebec/"
cp "$D"/quebec/*.parquet "$LOCAL/quebec/"
cp -r "$D/quebec/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/*-"${REG}".* "$LOCAL/derives/auxiliaires/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/rsesq-puits.parquet "$LOCAL/derives/auxiliaires/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/depots-sigeom-troncons.parquet "$LOCAL/derives/auxiliaires/" 
export MEANDRE_DATA="$LOCAL" MEANDRE_DERIVES="$LOCAL/derives" MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEANDRE_DIAG_CPU=1

# Variantes. La tete reprend les reglages du banc (R297, R299) : routage a 24 sous-pas d'une
# heure, borne basse du temps de transfert a une heure, taux propre x 100. Le drainage est
# le profil a deux vitesses plus Hooghoudt sur la couche 2 (R296).
VARIANTE=()
# Les depots ajoutent six attributs au champ (R314, R315).
TETE=(ETL_ROUTAGE_TETE=1 ETL_ROUTAGE_SOUSPAS=24 MEANDRE_KMUSK=1,48,24)
case "$VAR" in
  depots) VARIANTE=(ETL_ATTRIBUTS_DEPOTS=1) ;;
  depots-tete) VARIANTE=(ETL_ATTRIBUTS_DEPOTS=1 "${TETE[@]}") ;;
  depots-drain-tete) VARIANTE=(ETL_ATTRIBUTS_DEPOTS=1 ETL_SOIL_TOML=.runs/quebec/config/sol-drain-appris.toml "${TETE[@]}") ;;
esac
COMMUN=(ETL_CONFIG=.runs/quebec/config/socle-2026-10-04.toml ETL_REGION=$REG ETL_SEED=$GRAINE
        JOINT_FX_SUFFIX=-casr-brut ETL_COMPILE=${ETL_COMPILE:-1} "${VARIANTE[@]}")

echo "=== $REG $VAR graine $GRAINE : entrainement $(date +%T) ==="
env -u ETL_WARM_FROM "${COMMUN[@]}" ETL_EPOCHS=$EPOQUES ETL_TAG=$TAG MEANDRE_SAUVER_EPOQUES=1 \
    python -u .runs/quebec/etl_run.py

# Jugement sur la DERNIERE epoque, jamais sur celle retenue par la validation.
DERNIER=.runs/quebec/checkpoints/best-${REG}-etl${TAG}-e${EPOQUES}.pt
echo "=== $REG $VAR graine $GRAINE : passe avant sur $DERNIER $(date +%T) ==="
env "${COMMUN[@]}" ETL_EPOCHS=0 ETL_TAG=$TAG-eval ETL_WARM_FROM="$DERNIER" \
    ETL_DUMP_Q="$SORTIE/q-$REG$TAG.npz" \
    python -u .runs/quebec/etl_run.py

cp -f .runs/quebec/checkpoints/best-${REG}-etl${TAG}*.pt .runs/quebec/checkpoints/best-${REG}-etl${TAG}*.recette.toml \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
echo "=== $REG $VAR graine $GRAINE : fini $(date +%T) ==="
