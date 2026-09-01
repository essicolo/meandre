#!/usr/bin/env bash
# UNE region = UNE tache du tableau. Les quinze sont independantes, donc elles tournent en
# parallele : la nuit de vingt heures du poste devient l'attente d'une seule region.
#
#   sbatch --account=<compte> --array=0-14 region.sbatch
#
# RANGEMENT (impose par les quotas de l'Alliance, et par l'experience du 2026-09-01 ou
# tout avait atterri en vrac dans le dossier personnel) :
#   $HOME      code et environnement Python : petit, sauvegarde, jamais de donnees
#   $PROJET    donnees de reference : forcages, bases regionales, fichiers de calage
#   $TRAVAIL   journaux et points de reprise produits ici : grand, rapide, PURGE a 60 jours
# Les trois se declarent avant la soumission ; des defauts raisonnables prennent le relais.
#SBATCH --job-name=meandre
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
PROJET=${PROJET:-$HOME/projects/$(id -gn)/$USER}
TRAVAIL=${TRAVAIL:-$HOME/scratch}
JOURNAUX=$TRAVAIL/meandre/journaux
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints"
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm vaud)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
exec > "$JOURNAUX/$REG-$SLURM_ARRAY_JOB_ID.log" 2>&1

# arrow fournit pyarrow et mpi4py son homonyme : a l'Alliance ils viennent de MODULES,
# jamais de pip, et le module doit etre charge AVANT l'activation de l'environnement.
module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

# Les donnees vivent dans l'espace de projet ; on les copie sur le disque LOCAL du noeud,
# beaucoup plus rapide et sans concurrence entre les taches du tableau.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-casr-corr.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

MAJ=$(echo "$REG" | tr a-z A-Z)
# Poids auxiliaires a zero la ou les observations satellitaires ne sont pas en cache : la
# construction de la perte refuse un poids sans sa cible.
AUX=(ETL_WET=0.4)
case "$REG" in slso|vaud|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
SFX=${JOINT_FX_SUFFIX:--budyko}
[ "$REG" = vaud ] && SFX=-none
# ETL_COMPILE=0 : les grappes de l'Alliance n'ont pas Triton, et torch.compile du sol y
# fait echouer la simulation. La compilation n'achete que de la vitesse.
# RECETTE DU SOCLE, identique au poste : sol impose sauf K_sat et porosites, Linacre
# calee, seuil pluie/neige au bulbe humide, fonte regionale ancree, aquifere.
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=$SFX ETL_FORCE=1 ETL_REGION=$REG ETL_COMPILE=0 \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=${ETL_TAG:--fir} ETL_EPOCHS=${ETL_EPOCHS:-30} \
    python -u .runs/quebec/etl_run.py

# Les points de reprise produits redescendent dans l'espace de travail, pas dans le code.
cp -f "$RACINE"/.runs/quebec/checkpoints/best-${REG}-etl${ETL_TAG:--fir}.pt \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
