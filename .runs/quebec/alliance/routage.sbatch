#!/usr/bin/env bash
# BANC APPARIE DU MODE DE ROUTAGE, en inference pure (zero epoque) : chaque modele
# retenu evalue deux fois, en "operator-lagged" et en "operator".
#
# Pourquoi ce banc (2026-09-03). Le mode operator-lagged, defaut de toutes les flottes,
# calcule la sortie de chaque lac sur son stockage de la VEILLE : un jour de delai par
# lac traverse, et les lacs en serie additionnent ces jours. Mesure sur les graphes :
# mediane de 2 lacs en serie au-dessus des stations d'outv et de sagu, de 5 a 8 pour
# abit, outm et cnde, avec une correlation de 0.65 a 0.84 entre ce nombre et l'aire
# drainee. C'est la signature du retard observe (croissant avec l'aire, nul chez
# Hydrotel). Le mode "operator" (etage) resout les lacs avec l'apport du jour.
#
# Quinze regions, deux modes : trente taches, quelques minutes chacune.
#   sbatch --account=<compte> --array=0-29 routage.sbatch
#SBATCH --job-name=routage
#SBATCH --time=01:00:00
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
# Modele retenu par region : le sien pour les six regions calibrees, le donneur designe
# par la distance de Gower pour les neuf autres.
declare -A CK=([outv]=outv-etl-canon [gasp]=gasp-etl-socle30 [sagu]=sagu-etl-socle30 \
  [mont]=mont-etl-gen1 [slno]=slno-etl-socle30 [slso]=slso-etl-casr \
  [abit]=outv-etl-canon [outm]=outv-etl-canon [labi]=slso-etl-casr [vaud]=mont-etl-gen1 \
  [cnda]=slno-etl-socle30 [cndb]=slno-etl-socle30 [cndc]=slno-etl-socle30 \
  [cndd]=slno-etl-socle30 [cnde]=slno-etl-socle30)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 15))]}
if [ $((SLURM_ARRAY_TASK_ID / 15)) -eq 0 ]; then MODE=operator-lagged; else MODE=operator; fi
JOURNAUX=$TRAVAIL/meandre/routage
mkdir -p "$JOURNAUX" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG-$MODE.log" 2>&1

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
# Le champion de chaque region a ete entraine sur le forcage HYBRIDE (version 1.0).
# L'evaluer sur un autre forcage donne un score faux. slso a son propre fichier
# (forcing-casr-corr.nc, resolu par la table FORCINGS de joint_data) ; les autres
# lisent forcing-<reg>-hyb.nc. On FIXE le suffixe en dur (unset toute fuite d'env de
# session) et on refuse de continuer si le fichier attendu manque, au lieu du repli
# silencieux sur forcing-<reg>-budyko.nc qui a invalide tout un banc le 2026-09-03.
unset JOINT_FX_SUFFIX
if [ "$REG" != slso ] && [ ! -f "$LOCAL/quebec/forcing-${REG}-hyb.nc" ]; then
  echo "ARRET : forcing-${REG}-hyb.nc absent du colis. Le champion est sur le forcage" \
       "hybride ; sans lui la tache retomberait en silence sur budyko et mesurerait le" \
       "mauvais modele. Televerser les forcages hybrides dans donnees/quebec." >&2
  exit 3
fi
# Les contraintes auxiliaires n'ont pas de cible hors des six regions calibrees.
AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0)
env MEANDRE_NSUBSTEP=64 ETL_ROUTING=$MODE \
    JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-rt ETL_EPOCHS=0 \
    ETL_WARM_FROM="$PROJET/meandre/checkpoints/best-${CK[$REG]}.pt" \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$MODE.npz" \
    python -u .runs/quebec/etl_run.py
