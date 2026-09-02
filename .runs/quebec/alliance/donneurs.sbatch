#!/usr/bin/env bash
# QUEL DONNEUR POUR UNE REGION PEU JAUGEE ? Neuf regions x six donneurs = 54 mesures.
# En local c'est cinq heures en file ; ici les 54 partent ensemble.
#
# La question est ouverte et elle compte : sur OUTV, transferer depuis le Lac-Saint-Jean
# donne 0.573 et depuis la Gaspesie 0.376 (mesure du 7 aout). La similarite du territoire
# (distance de Gower) ne la predit PAS : elle a fait gagner deux regions et perdre deux
# autres le 2026-09-01. Il faut donc mesurer.
#
# Selection sur la VALIDATION 2019-2021 ; la periode d'evaluation 2022-2024 est reportee
# mais ne choisit jamais. Les series aux jauges sont sauvees pour chaque couple, de sorte
# que les ponderations (donneur unique, moyenne des k plus proches, inverse de la
# distance) se comparent ensuite hors ligne, sans un calcul de plus.
#
#   sbatch --account=<compte> --array=0-53 donneurs.sbatch
#SBATCH --job-name=donneurs
#SBATCH --time=01:30:00
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
REGIONS=(abit cnda cndb cndc cndd cnde labi outm vaud)
DONNEURS=(outv-etl-canon slno-etl-socle30 gasp-etl-socle30 sagu-etl-socle30 \
          mont-etl-gen1 slso-etl-canon)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID / 6))]}
CK=${DONNEURS[$((SLURM_ARRAY_TASK_ID % 6))]}
JOURNAUX=$TRAVAIL/meandre/donneurs
mkdir -p "$JOURNAUX" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG-$CK.log" 2>&1

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
SFX=${JOINT_FX_SUFFIX:--hyb}
[ "$REG" = vaud ] && [ "$SFX" = "-budyko" ] && SFX=-none
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=$SFX ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 \
    ETL_MELT_SAISON=0.5 ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-don ETL_EPOCHS=0 ETL_EVAL_VAL=1 \
    ETL_WARM_FROM="$PROJET/meandre/checkpoints/best-$CK.pt" \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$CK.npz" \
    python -u .runs/quebec/etl_run.py
