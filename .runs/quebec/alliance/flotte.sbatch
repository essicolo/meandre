#!/usr/bin/env bash
# LA PROVINCE SOUS UNE BOUCLE JUSTE : quinze regions, deux bras, pour la fin de semaine.
#
# Premiere flotte entrainee apres les correctifs des 3 et 4 septembre : continuite de
# l'etat riche entre blocs et aux frontieres de validation (R64, R75), KGE sur la sequence
# continue (R67), filet par bloc et plancher du KGE (R77), surface des lacs (R60),
# conductivite modulee et non ecrasee, Z2/Z3 en tenseurs. La recette est LUE DANS UN
# FICHIER (socle.toml, section [recette]) : c'est elle qui definit le modele, plus un bloc
# de variables shell (R63 et inventaire du 2026-09-03).
#
# Deux bras, pour ne pas laisser dormir la fin de semaine :
#   A  socle, boucle corrigee   indices 0-14,  ETL_TAG=-fdsA : un pas par bloc (R78),
#      historique amorce (chaque bloc voit le KGE de toute la periode), minimum de trente
#      qui compte l'historique et aucun pas sur le prior seul (R80)
#   B  socle, ancienne boucle   indices 15-29, ETL_TAG=-fdsB : un pas par epoque, sans
#      amorce, tout le reste identique. C'est le TEMOIN : lundi, A contre B region par
#      region dit si la boucle corrigee apprend, ce qu'aucun banc local n'a encore
#      montre (2026-09-04, 16 h). Le Muskingum physique (R62) attend ce verdict.
# Soumettre A et B ensemble : le verdict est la paire, pas un bras seul.
#
#   sbatch --account=<compte> --array=0-14%6  flotte.sbatch     # bras A
#   sbatch --account=<compte> --array=15-29%6 flotte.sbatch     # bras B
#
# Forcage : tout-CaSR corrige (-budyko), le seul present sur la grappe, FIXE ici. Le
# fichier attendu manquant est une erreur, pas un repli silencieux (R63).
#SBATCH --job-name=flotte
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
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm vaud)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 15))]}
if [ $((SLURM_ARRAY_TASK_ID / 15)) -eq 0 ]; then
  BRAS=A; TAG=-fdsA; KMUSK=4,48,24; PAS_PAR_BLOC=1; AMORCE=1
else
  BRAS=B; TAG=-fdsB; KMUSK=4,48,24; PAS_PAR_BLOC=0; AMORCE=0
fi
JOURNAUX=$TRAVAIL/meandre/flotte
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG-$BRAS${FLOTTE_TAG:-}.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/slso"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/"
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D/forcing-casr-corr.nc" "$LOCAL/slso/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

# Aucune fuite de session ne doit changer le forcage ni les leviers de la boucle.
unset JOINT_FX_SUFFIX MEANDRE_ETAT_CONTINU MEANDRE_KGE_CONTINU MEANDRE_KMUSK
# R78 : un pas d'optimisation par bloc (~65 par epoque) au lieu d'un par epoque.
# Sans lui, trente epoques valaient vingt-cinq pas d'Adam et le champ n'apprenait
# presque rien. PAS_PAR_BLOC=0 restitue l'ancien comportement.
PAS_PAR_BLOC=${PAS_PAR_BLOC:-1}
AMORCE=${AMORCE:-1}
MAJ=$(echo "$REG" | tr a-z A-Z)
if [ "$REG" != slso ] && [ ! -f "$LOCAL/quebec/forcing-${REG}-budyko.nc" ]; then
  echo "ARRET : forcing-${REG}-budyko.nc absent du colis." >&2
  exit 3
fi
# Les contraintes auxiliaires n'ont pas de cible hors des six regions calibrees : une
# contrainte sans cible ne contraint rien et deplace la solution.
AUX=(ETL_WET=0.4)
case "$REG" in slso|vaud|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
# PAIRE DE CONTROLE (2026-09-05). La flotte du 4 septembre a montre que les trois
# effondrements de validation (GASP dans les deux bras, MONT-B, SAGU-A) sont dans les
# regions ou GRACE est active, et que pendant l'effondrement TOUS les termes de la perte
# montent, le biais saisonnier GRACE de 28 fois : c'est une divergence, pas un compromis.
# FLOTTE_AUX remplace le bloc de contraintes auxiliaires et FLOTTE_TAG s'ajoute au tag,
# pour relancer une region avec une contrainte eteinte sans toucher au reste :
#   sbatch --account=<compte> --array=6,21 \
#     --export=ALL,FLOTTE_AUX="ETL_WET=0.4 ETL_WTWS=0 ETL_WTWSCLIM=0",FLOTTE_TAG=-sansgrace \
#     flotte.sbatch        # GASP, bras A (6) et B (21), MOD16 gardee, GRACE eteinte
if [ -n "${FLOTTE_AUX:-}" ]; then read -r -a AUX <<< "$FLOTTE_AUX"; fi
TAG="$TAG${FLOTTE_TAG:-}"

env ETL_CONFIG=.runs/quebec/config/socle.toml MEANDRE_PAS_PAR_BLOC=$PAS_PAR_BLOC \
    MEANDRE_HISTORIQUE_AMORCE=$AMORCE \
    JOINT_FX_SUFFIX=-budyko MEANDRE_KMUSK="$KMUSK" \
    ETL_FORCE=1 ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" \
    ETL_TAG=$TAG ETL_EPOCHS=${ETL_EPOCHS:-30} \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$BRAS${FLOTTE_TAG:-}.npz" \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl${TAG}.pt" \
      "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl${TAG}.recette.toml" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
