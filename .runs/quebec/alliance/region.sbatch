#!/usr/bin/env bash
# ETAPE 5 : UNE region = UNE tache du tableau. Les quinze sont independantes, donc elles
# tournent en parallele : la nuit de vingt heures du poste devient l'attente d'une seule
# region, environ deux heures, plus le temps de file.
#
#   sbatch --account=<compte> --array=0-14 region.sbatch
#
# Variables attendues : MEANDRE_RACINE (le depot), MEANDRE_DATA, MEANDRE_PLATFORMS.
#SBATCH --job-name=meandre
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm vaud)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
module purge && module load StdEnv/2023 python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "${MEANDRE_RACINE:-$HOME/meandre}"
# Les donnees vivent dans l'espace de projet ; on les copie sur le disque LOCAL du noeud,
# beaucoup plus rapide et sans concurrence entre les quinze taches.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
cp "$MEANDRE_DATA/quebec/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$MEANDRE_DATA"/quebec/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$MEANDRE_DATA"/quebec/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$MEANDRE_DATA/quebec/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
MAJ=$(echo "$REG" | tr a-z A-Z)
# Poids auxiliaires a zero la ou les observations satellitaires ne sont pas en cache :
# la construction de la perte refuse un poids sans sa cible.
AUX=(ETL_WET=0.4)
case "$REG" in slso|vaud|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0) ;; esac
SFX=${JOINT_FX_SUFFIX:--budyko}
[ "$REG" = vaud ] && SFX=-none
# RECETTE DU SOCLE, identique au poste : sol impose sauf K_sat et porosites, Linacre
# calee, seuil pluie/neige au bulbe humide, fonte regionale ancree, aquifere.
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=$SFX ETL_FORCE=1 ETL_REGION=$REG \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATFORMS/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=${ETL_TAG:--fir} ETL_EPOCHS=${ETL_EPOCHS:-30} \
    python -u .runs/quebec/etl_run.py
