#!/usr/bin/env bash
# CACHES PAR TRONCON POUR LA CARTE (demande d'Essi, 2026-09-06).
#
# La carte feuillage colore le reseau par recharge, evapotranspiration, prelevements
# relatifs et champs de parametres appris. Ces grandeurs ne sont pas dans les series aux
# stations : elles demandent une passe d'inference par region, avec le point de reprise
# de la flotte retenue, qui ecrit le cache ETL_DUMP_REACH.
#
# Deux passes par region : AVEC les prelevements et rejets, et SANS (renaturalise), pour
# que l'effet anthropique se calcule par difference.
#
#   sbatch --account=<compte> --array=0-14%6 --export=ALL,ETL_TAG=-v2lr1e-4 carte.sbatch
#
# Puis, depuis le poste :
#   scp "atlas01@narval.alliancecan.ca:scratch/meandre/carte/nb-*.npz" "D:/meandre-data/quebec/results/"
#   MEANDRE_BRAS=A-v2lr1e-4 .venv/Scripts/python.exe .runs/quebec/build_feuillage.py
#
#SBATCH --job-name=carte
#SBATCH --time=02:00:00
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
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm vaud)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SORTIE=$TRAVAIL/meandre/carte
mkdir -p "$SORTIE"
exec > "$SORTIE/$REG.log" 2>&1

cd "$RACINE"
module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "${MEANDRE_VENV:-$HOME/venv-meandre}/bin/activate"
# MISE EN PLACE LOCALE, comme la flotte. Le Saint-Laurent sud a une entree propre dans
# la table des forcages, `slso/forcing-casr-corr.nc`, qui n'existe que dans l'arborescence
# mise en place par le script : pointer directement le dossier du projet le fait echouer
# en dix-sept secondes (2026-09-06). On copie donc les memes fichiers que la flotte.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/slso"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/"
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp "$D/forcing-casr-corr.nc" "$LOCAL/slso/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
MAJ=$(echo "$REG" | tr '[:lower:]' '[:upper:]')
unset JOINT_FX_SUFFIX MEANDRE_ETAT_CONTINU MEANDRE_KGE_CONTINU MEANDRE_KMUSK

# ETL_EPOCHS=0 : aucune epoque, le pilote charge le point de reprise nomme par ETL_TAG et
# se contente d'evaluer. ETL_SANS_PRELEV=1 met prelevements et rejets a zero.
for PASSE in avec sans; do
  SP=0; [ "$PASSE" = sans ] && SP=1
  env ETL_CONFIG=.runs/quebec/config/socle.toml \
      JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-1} \
      ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
      ETL_TAG=${ETL_TAG:--v2lr1e-4} ETL_EPOCHS=0 ETL_SANS_PRELEV=$SP \
      ETL_DUMP_REACH="$SORTIE/nb-$REG-$PASSE.npz" \
      python -u .runs/quebec/etl_run.py
done
echo "carte : caches $REG ecrits dans $SORTIE"
