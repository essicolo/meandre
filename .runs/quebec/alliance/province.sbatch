#!/usr/bin/env bash
# MODELE PROVINCIAL, RECETTE DU 4 OCTOBRE : un territoire par tache du tableau, seize
# epoques, puis une passe avant sur le point de reprise de la DERNIERE epoque qui exporte
# les debits journaliers avec et sans prelevements pour la carte de la presentation.
#
#   sbatch --account=<compte> --array=0-13 province.sbatch
#
# Contrairement a region.sbatch, aucun reglage de physique n'est pose ici : la recette
# vient entierement de socle-2026-10-04.toml. Une variable posee au lanceur primerait sur
# le fichier, et c'est ainsi que region.sbatch porte encore l'ancienne loi des ancrages.
# Seuls restent au lanceur le territoire, le forcage, le nombre d'epoques et les exports.
#
# Vaudreuil n'a aucune station : il n'est pas entraine ici (indice 14 absent du tableau).
#SBATCH --job-name=meandre-prov
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
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
JOURNAUX=$TRAVAIL/meandre/journaux
CARTE=$TRAVAIL/meandre/carte
mkdir -p "$JOURNAUX" "$CARTE" "$TRAVAIL/meandre/checkpoints"
REGIONS=(gasp outv slno slso sagu mont outm abit cndd cndb cnda cnde cndc labi vaud)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
TAG=${ETL_TAG:--prov1005}
EPOQUES=${ETL_EPOCHS:-16}
exec > "$JOURNAUX/$REG$TAG-$SLURM_ARRAY_JOB_ID.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"
mkdir -p .runs/quebec/checkpoints

# Donnees copiees sur le disque local du noeud : base du territoire, son forcage CaSR brut,
# les tables provinciales (attributs bruts, lacs HydroLAKES, champs d'ancrage) et les cibles
# auxiliaires (NEISIM pour le poids hors neige du terme MOD16). Le projet PHYSITEL, lu pour
# l'occupation du sol et les milieux humides, reste dans l'espace de projet.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec" "$LOCAL/derives/auxiliaires"
D=$PROJET/meandre/donnees
cp "$D/quebec/${REG}.duckdb" "$LOCAL/quebec/"
cp "$D/quebec/forcing-${REG}-casr-brut.nc" "$LOCAL/quebec/"
cp "$D"/quebec/*.parquet "$LOCAL/quebec/"
cp -r "$D/quebec/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/*-"${REG}".* "$LOCAL/derives/auxiliaires/" 2>/dev/null || true
cp "$D"/derives/auxiliaires/rsesq-puits.parquet "$LOCAL/derives/auxiliaires/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL" MEANDRE_DERIVES="$LOCAL/derives" MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEANDRE_DIAG_CPU=1

# L'Outaouais moyen n'a pas de GRACE en cache : la perte refuse un poids sans sa cible.
AUX=()
case "$REG" in outm) AUX=(ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
COMMUN=(ETL_CONFIG=.runs/quebec/config/socle-2026-10-04.toml ETL_REGION=$REG
        JOINT_FX_SUFFIX=-casr-brut ETL_COMPILE=${ETL_COMPILE:-1} "${AUX[@]}")

# CONTROLE DE REPRODUCTION : ETL_EPOCHS=0 et ETL_WARM_FROM=<point de reprise> sautent
# l'entrainement et ne font que la passe avant, a comparer au chiffre du poste.
if [ "$EPOQUES" -gt 0 ]; then
  echo "=== $REG entrainement $(date +%T) ==="
  env -u ETL_WARM_FROM "${COMMUN[@]}" ETL_EPOCHS=$EPOQUES ETL_TAG=$TAG MEANDRE_SAUVER_EPOQUES=1 \
      ETL_DUMP_Q="$CARTE/q-$REG$TAG-entrainement.npz" \
      python -u .runs/quebec/etl_run.py
  # Le modele retenu est celui de la DERNIERE epoque, pas le meilleur en validation : la
  # selection sur trois ans de validation est un tirage (registre, regle du banc).
  DERNIER=.runs/quebec/checkpoints/best-${REG}-etl${TAG}-e${EPOQUES}.pt
else
  DERNIER=${ETL_WARM_FROM:?ETL_EPOCHS=0 demande ETL_WARM_FROM}
fi
echo "=== $REG passe avant sur $DERNIER $(date +%T) ==="
env "${COMMUN[@]}" ETL_EPOCHS=0 ETL_TAG=$TAG-carte ETL_WARM_FROM="$DERNIER" \
    ETL_DUMP_Q="$CARTE/q-$REG$TAG.npz" \
    ETL_DUMP_REACH="$CARTE/$REG.npz" ETL_DUMP_NATUREL=1 \
    python -u .runs/quebec/etl_run.py

# Le fichier journalier complet pese 1,5 Go par territoire (dix-sept grandeurs) ; la carte
# n'en lit que le debit, extrait ici dans un fichier a rapatrier.
python -c "import numpy as np; z = np.load('$CARTE/$REG-journalier.npz'); np.savez_compressed('$CARTE/$REG-q-journalier.npz', dates=z['dates'], q=z['q'])"

cp -f .runs/quebec/checkpoints/best-${REG}-etl${TAG}*.pt .runs/quebec/checkpoints/best-${REG}-etl${TAG}*.recette.toml \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
echo "=== $REG fini $(date +%T) ==="
