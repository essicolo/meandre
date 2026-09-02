#!/usr/bin/env bash
# ENVELOPPE D'INCERTITUDE PAR REGION : une tete de regression quantile par region, sur
# le modele retenu GELE. La mediane de l'enveloppe est exactement le debit du modele, donc
# le KGE ne bouge pas ; seule l'incertitude s'ajoute. K=6 quantiles, perte pinball,
# selection sur la vraisemblance.
#
# Exigence d'Essi (2026-09-02) : le rapport doit porter une incertitude. Le volet
# probabiliste etait valide sur un bassin d'essai mais n'avait jamais ete deploye.
#
# Quatorze regions ; Vaudreuil est exclu, aucune station n'y est rattachee et une
# enveloppe ne peut pas s'y apprendre.
#
#   sbatch --account=<compte> --array=0-13 quantile.sbatch
#SBATCH --job-name=quantile
#SBATCH --time=03:00:00
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
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm)
MODELES=(outv-etl-canon gasp-etl-socle30 sagu-etl-socle30 mont-etl-gen1 \
         slno-etl-socle30 slso-etl-casr outv-etl-canon slno-etl-socle30 \
         slno-etl-socle30 slno-etl-socle30 slno-etl-socle30 slno-etl-socle30 \
         slso-etl-canon outv-etl-canon)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
CK=${MODELES[$SLURM_ARRAY_TASK_ID]}
JOURNAUX=$TRAVAIL/meandre/quantile
mkdir -p "$JOURNAUX" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG.log" 2>&1

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
case "$REG" in slso|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=${JOINT_FX_SUFFIX:--hyb} ETL_FORCE=1 \
    ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-1} ETL_QUANTILE=1 \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-q1 ETL_EPOCHS=${ETL_EPOCHS:-10} \
    ETL_WARM_FROM="$PROJET/meandre/checkpoints/best-$CK.pt" \
    python -u .runs/quebec/etl_run.py

# Le point de reprise avec tete quantile redescend dans l'espace de travail.
cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-q1.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
