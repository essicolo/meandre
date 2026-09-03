#!/usr/bin/env bash
# BANC APPARIE DES BORNES DE MUSKINGUM, sur le forcage COHERENT tout-CaSR.
#
# Pourquoi ce banc (2026-09-03). Le temps de transfert de Muskingum est borne [4, 48] h
# par le champ spatial, alors que le troncon median mesure 2.9 a 3.5 km : a 1 m/s il est
# parcouru en une heure. La borne INFERIEURE excede donc de quatre a cinq fois le temps
# physique, et chaque troncon se comporte en reservoir d'un demi-jour, effet compose le
# long de la chaine topologique. C'est le mecanisme candidat du retard d'un a trois jours
# mesure sur la simulation, retard qui croit avec l'aire drainee et le nombre de troncons
# traverses, et que la reference Hydrotel n'a pas.
#
# Un premier banc, le 2026-09-02, posait l'echelle de K sur la longueur du troncon lue
# dans le graphe. Il concluait que l'echelle physique perd sur quatre regions sur six.
# CE VERDICT EST NUL : la longueur des troncons de lac etait en fait leur SURFACE en
# metres carres (corrige le 2026-09-03), soit 437 a 656 km au lieu de 3 a 5, ce qui
# donnait des constantes de transfert de 167 h par lac. On ne refait pas la meme mesure :
# on agit directement sur les BORNES du champ, par MEANDRE_KMUSK="min,max,init", ce qui
# ne depend d'aucun attribut de longueur.
#
# Deux bras, meme graine, meme recette, meme forcage, trente epoques :
#   temoin    MEANDRE_KMUSK=4,48,24    les bornes actuelles
#   physique  MEANDRE_KMUSK=0.5,12,3   une plage qui contient le temps de parcours reel
#
# Forcage : tout-CaSR corrige (le suffixe -budyko), seul forcage COHERENT en source
# unique, et le seul present sur la grappe. Le bras temoin reproduit donc la version
# 1.0.1 et sert de controle : s'il ne retrouve pas la mediane connue, le banc est suspect
# avant meme d'etre lu.
#
#   sbatch --account=<compte> --array=0-11 muskingum.sbatch
#SBATCH --job-name=musk
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
REGIONS=(outv gasp sagu mont slno slso)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID % 6))]}
if [ $((SLURM_ARRAY_TASK_ID / 6)) -eq 0 ]; then
  VARIANTE=temoin;   KMUSK=4,48,24
else
  VARIANTE=physique; KMUSK=0.5,12,3
fi
JOURNAUX=$TRAVAIL/meandre/muskingum
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
cp "$D/forcing-casr-corr.nc" "$LOCAL/slso/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

# Forcage FIXE : aucune fuite d'environnement de session ne doit pouvoir le changer.
# Le repli silencieux sur un autre forcage a invalide un banc entier le 2026-09-03.
unset JOINT_FX_SUFFIX
MAJ=$(echo "$REG" | tr a-z A-Z)
if [ "$REG" != slso ] && [ ! -f "$LOCAL/quebec/forcing-${REG}-budyko.nc" ]; then
  echo "ARRET : forcing-${REG}-budyko.nc absent du colis." >&2
  exit 3
fi
AUX=(ETL_WET=0.4)
[ "$REG" = slso ] && AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0)

env MEANDRE_NSUBSTEP=64 MEANDRE_KMUSK="$KMUSK" \
    JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_TAG=-mk$VARIANTE ETL_EPOCHS=${ETL_EPOCHS:-30} \
    ETL_DUMP_Q="$JOURNAUX/q-$REG-$VARIANTE.npz" \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-mk${VARIANTE}.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
