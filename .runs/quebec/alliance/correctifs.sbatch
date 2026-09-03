#!/usr/bin/env bash
# BANC APPARIE DES DEUX CORRECTIFS D'ENTRAINEMENT (2026-09-03).
#
# Deux defauts de la boucle d'entrainement ont ete corriges le meme jour. Ils sont de la
# meme famille : la perte et la selection ne portaient pas sur le meme objet.
#
#   R64  L'etat riche etait aneanti tous les 45 jours. Le manteau neigeux par classe, le
#        profil de gel, le milieu humide et le stockage des lacs vivent hors de
#        HydroState ; simulate les refabriquait a chaque appel, donc a chaque bloc.
#        L'entrainement n'avait jamais simule d'hiver continu, alors que la selection
#        juge une trajectoire continue de trois ans.
#
#   R67  Le KGE de la perte portait sur 45 jours, celui de la selection sur trois ans.
#        Mesure sur les huit regions du rapport : KGE median par fenetre -0.01 a 0.27
#        contre 0.55 a 0.78 en continu, et sur slno une correlation de 0.05 entre les
#        deux. La perte optimisait un objet sans rapport avec la metrique.
#
# Les deux corrections changent ce que l'entrainement voit et ce qu'il optimise. Leur
# effet est INCONNU et peut etre negatif : un modele entraine sur des hivers de deux
# semaines a pu apprendre a compenser, comme les parametres compensaient le routage lent.
# On les mesure donc AVANT d'en empiler d'autres, sur trois regions enneigees et
# contrastees en densite de lacs (gasp 2 %, outv 15 %, sagu 16 %).
#
# Six taches : 0-2 ancien comportement, 3-5 corrige.
#   sbatch --account=<compte> --array=0-5 correctifs.sbatch
#SBATCH --job-name=correctifs
#SBATCH --time=08:00:00
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
REGIONS=(gasp outv sagu)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 3))]}
if [ $((SLURM_ARRAY_TASK_ID / 3)) -eq 0 ]; then
  VARIANTE=ancien;  ETAT=0; KGE=0
else
  VARIANTE=corrige; ETAT=1; KGE=1
fi
JOURNAUX=$TRAVAIL/meandre/correctifs
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
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

# Forcage FIXE, tout-CaSR corrige : le seul coherent en source unique, et le seul present
# sur la grappe. Aucune fuite d'environnement ne doit pouvoir le changer, et l'absence du
# fichier est une erreur, pas un repli silencieux (R63).
unset JOINT_FX_SUFFIX
MAJ=$(echo "$REG" | tr a-z A-Z)
if [ ! -f "$LOCAL/quebec/forcing-${REG}-budyko.nc" ]; then
  echo "ARRET : forcing-${REG}-budyko.nc absent du colis." >&2
  exit 3
fi

env MEANDRE_NSUBSTEP=64 \
    MEANDRE_ETAT_CONTINU=$ETAT MEANDRE_KGE_CONTINU=$KGE \
    JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-cx$VARIANTE ETL_EPOCHS=${ETL_EPOCHS:-30} \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$VARIANTE.npz" \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-cx${VARIANTE}.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
