#!/usr/bin/env bash
# ENVELOPPE PROBABILISTE SUR LE MODELE PRESENTE : une tete de quantiles par region, sur
# le socle GELE du point de reprise de la flotte v4.
#
# POURQUOI CE FICHIER ET PAS quantile.sbatch (2026-09-11). L'ancien job designe des
# points de reprise d'une generation anterieure (outv-etl-canon, gasp-etl-socle30) et
# n'ecrit AUCUN export de quantiles, seulement un point de reprise. Les configurations
# .runs/quebec/config/<region>-quantile.toml, elles, designent checkpoints/best-<reg>.pt
# qui n'existe pas : slso.py saute alors le depart a chaud SANS RIEN DIRE et gele une
# physique jamais entrainee. Mesure sur neuf regions : le debit exporte s'ecarte de 31 a
# 47 % de celui de la flotte. Ici le depart a chaud est OBLIGATOIRE et son absence
# arrete la tache.
#
# POURQUOI SUR LA GRAPPE ET PAS EN LOCAL. Sous Windows, Triton manque et torch.compile
# est indisponible : mesure le 2026-09-10, la plus petite region a pris 70 minutes par
# epoque, soit environ 85 heures pour les neuf. La compilation est active ici.
#
# CE QUE LA PHASE FAIT. Le pilote gele tout sauf la tete de quantiles (K=6, perte
# pinball), met a zero tous les poids de debit et selectionne sur la vraisemblance. La
# mediane de l'enveloppe reste exactement le debit du point de reprise, donc aucun score
# deterministe ne bouge.
#
# ARBITRAGE SUR LE NOMBRE D'EPOQUES, mesure sur la Cote-Nord C le 2026-09-10 : la
# vraisemblance s'ameliore de facon monotone (35,6 a 32,8 en huit epoques) pendant que la
# COUVERTURE se degrade (intervalle a 90 % : 0,813 a l'epoque 0, 0,762 a l'epoque 7 ;
# intervalle a 50 % : 0,40 a la fin). L'enveloppe se resserre a mesure qu'on l'entraine.
# Si le juge est le diagramme de Talagrand, peu d'epoques valent mieux que beaucoup.
# Poser ETL_EPOCHS pour trancher ; defaut 8, la valeur deja mesuree.
#
#   sbatch --account=<compte> --array=0-8%3 quantile_v4.sbatch
#   sbatch --account=<compte> --array=0-8%3 --export=ALL,ETL_EPOCHS=3 quantile_v4.sbatch
#
#SBATCH --job-name=quantv4
#SBATCH --time=06:00:00
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
# Les neuf regions qui portent un point de reprise de la flotte v4.
REGIONS=(cndc cndb abit sagu mont slso slno outv gasp)
# SLURM_ARRAY_TASK_ID n'est pose que par l'ordonnanceur. Lance a la main, le script
# disait seulement « unbound variable » : il dit maintenant quoi faire, et accepte une
# region nommee pour un essai direct sur un noeud interactif.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
elif [ -n "${REG:-}" ]; then
  echo "[quantile] execution directe, region $REG"
else
  echo "Ce script se SOUMET, il ne s'execute pas :" >&2
  echo "  sbatch --account=<compte> --array=0-8%3 .runs/quebec/alliance/quantile_v4.sbatch" >&2
  echo "Pour un essai direct sur un noeud interactif : REG=cndc bash \$0" >&2
  echo "Regions, dans l'ordre des indices : ${REGIONS[*]}" >&2
  exit 2
fi
DEPART=${DEPART:--etl-fdsA-v4}
JOURNAUX=$TRAVAIL/meandre/quantile-v4
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints" "$RACINE/.runs/quebec/checkpoints"
exec > "$JOURNAUX/$REG.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

# Le point de reprise deterministe, cherche aux deux endroits ou la flotte le depose.
CKPT=""
for _c in "$TRAVAIL/meandre/checkpoints/best-${REG}${DEPART}.pt" \
          "$RACINE/.runs/quebec/checkpoints/best-${REG}${DEPART}.pt" \
          "$PROJET/meandre/checkpoints/best-${REG}${DEPART}.pt"; do
  [ -f "$_c" ] && CKPT="$_c" && break
done
if [ -z "$CKPT" ]; then
  echo "ARRET : best-${REG}${DEPART}.pt introuvable. Une tete apprise sur un modele non" >&2
  echo "        entraine ne produit qu'une enveloppe autour de rien." >&2
  exit 5
fi
echo "[quantile] depart a chaud depuis $CKPT"

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

# Prelevements reingeres dans la COPIE LOCALE du noeud, comme la flotte : les bases du
# projet portent encore deux ingestions empilees.
if [ "${REINGERER_PRELEV:-1}" = 1 ]; then
  if [ ! -f "$RACINE/data/io-eau-meandre.parquet" ]; then
    echo "ARRET : $RACINE/data/io-eau-meandre.parquet absent (derive d'io-eau a televerser)." >&2
    exit 4
  fi
  IO_EAU="$RACINE/data/io-eau-meandre.parquet" python -u "$RACINE/.runs/quebec/ingest_withdrawals.py" "$REG"
fi

MAJ=$(echo "$REG" | tr a-z A-Z)
# Le Saint-Laurent sud-ouest n'a pas de forcage -budyko : il porte son fichier propre,
# forcing-casr-corr.nc, et le pilote le resout lui-meme avec un message explicite
# (FORCINGS dans joint_data.py). La flotte porte la meme exception ; l'avoir omise a
# arrete la tache 5 du premier envoi.
if [ "$REG" != slso ] && [ ! -f "$LOCAL/quebec/forcing-${REG}-budyko.nc" ]; then
  echo "ARRET : forcing-${REG}-budyko.nc absent du colis." >&2
  exit 3
fi
# En phase quantile le pilote met a zero tous les poids de debit ET les contraintes
# auxiliaires : ce bloc n'a plus d'effet sur la perte, il est garde pour que la
# construction du modele soit identique a celle de la flotte.
AUX=(ETL_WET=0.4)
case "$REG" in slso|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac

env ETL_CONFIG=.runs/quebec/config/socle.toml \
    JOINT_FX_SUFFIX=-budyko MEANDRE_KMUSK=4,48,24 \
    MEANDRE_PAS_PAR_BLOC=1 MEANDRE_HISTORIQUE_AMORCE=1 \
    ETL_FORCE=1 ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" \
    ETL_QUANTILE=1 ETL_WARM_FROM="$CKPT" \
    ETL_TAG=-q1v4 ETL_EPOCHS=${ETL_EPOCHS:-8} \
    ETL_DUMP_Q="$JOURNAUX/quant-$REG.npz" \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE/.runs/quebec/checkpoints/best-${REG}-etl-q1v4.pt" \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
echo "[quantile] $REG termine. Export : $JOURNAUX/quant-$REG.npz"
