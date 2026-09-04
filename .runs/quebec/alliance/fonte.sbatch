#!/usr/bin/env bash
# AMPLITUDE DE LA FONTE SAISONNIERE, sous une boucle d'entrainement CORRIGEE.
#
# Pourquoi ce banc (2026-09-03). La modulation saisonniere du facteur de fonte vaut
# s(j) = 1 + amp*sin(2*pi*(j-81)/365), donc decembre x(1-amp) et juin x(1+amp), moyenne
# annuelle inchangee. L'amplitude 0.5 retenue par la recette du socle a ete calee quand
# l'entrainement REMETTAIT LE MANTEAU A ZERO tous les 45 jours (R64) : le modele n'avait
# alors jamais de manteau profond a gerer, et brider la fonte de decembre ne coutait rien.
#
# Depuis la correction, le manteau persiste. Mesure sur le banc apparie des correctifs :
#
#   volume simule / observe        hiver   printemps   ete   automne   annuel
#   gasp avant la correction        0.63      0.95     0.98    0.87      0.88
#   gasp apres la correction        0.70      1.06     1.26    1.32      1.09
#
# Le bilan annuel s'ameliore (erreur 0.12 -> 0.09) mais l'hiver reste sous-produit de
# trente pour cent et l'excedent ressort en ete et en automne. C'est la signature d'une
# fonte hivernale trop bridee : l'eau reste dans un manteau qui, lui, persiste desormais.
#
# PREDICTION POSEE D'AVANCE, pour ne pas interpreter apres coup : baisser l'amplitude doit
# remonter le rapport de volume d'hiver vers 1 et faire redescendre l'ete et l'automne, en
# Gaspesie surtout. Si l'hiver ne bouge pas, la fonte n'est pas le mecanisme et il faut
# chercher du cote de la recession de l'aquifere, figee sur ces modeles (R61).
#
# Trois amplitudes, trois regions, neuf taches :
#   0.50  la valeur du socle, calee sous l'ancienne boucle
#   0.25  moitie moins de bridage
#   0.00  aucune modulation, degre-jour du clone fidele
#
#   sbatch --account=<compte> --array=0-8 fonte.sbatch
#SBATCH --job-name=fonte
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
AMPLITUDES=(0.50 0.25 0.00)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 3))]}
AMP=${AMPLITUDES[$((SLURM_ARRAY_TASK_ID / 3))]}
ETIQ=amp${AMP/./p}
JOURNAUX=$TRAVAIL/meandre/fonte
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG-$ETIQ.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

unset JOINT_FX_SUFFIX
MAJ=$(echo "$REG" | tr a-z A-Z)
if [ ! -f "$LOCAL/quebec/forcing-${REG}-budyko.nc" ]; then
  echo "ARRET : forcing-${REG}-budyko.nc absent du colis." >&2
  exit 3
fi

env MEANDRE_NSUBSTEP=64 \
    MEANDRE_ETAT_CONTINU=1 MEANDRE_KGE_CONTINU=1 \
    JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 \
    ETL_MELT_SAISON=$AMP \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-fo$ETIQ ETL_EPOCHS=${ETL_EPOCHS:-30} \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$ETIQ.npz" \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-fo${ETIQ}.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
