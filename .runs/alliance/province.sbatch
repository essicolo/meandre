#!/usr/bin/env bash
# Entrainements provinciaux en parallele : une tache par ligne d'un fichier de taches (2026-11-09).
#
#   sbatch --account=<compte> --array=0-<n-1> province.sbatch <fichier de taches>
#
# Chaque ligne du fichier : configuration TOML, graine, etiquette, puis des affectations VAR=valeur
# facultatives (par exemple ETL_CHUNK=365 ou ETL_WARM_FROM=...). Les lignes vides et celles qui
# commencent par # sont ignorees. Exemple :
#   experiments/land-cover/province-wc-texture-lacs-fixes.toml 7 -lacs-fixes-n-s7 ETL_CHUNK=365
#   experiments/land-cover/province-wc-texture-lacs-fixes.toml 1234 -lacs-fixes-n-s1234 ETL_CHUNK=365
# Le domaine provincial entier est copie sur le disque local du noeud ; une carte A100 tient des
# blocs de 365 jours, que la carte de 8 Go d'Ubuntu refuse (chantier 10).
#SBATCH --job-name=meandre-province
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000M
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
TACHES_F=${1:?fichier de taches}
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
if [ -z "${PROJET:-}" ]; then
  for _c in "$HOME"/projects/*/"$USER"; do
    [ -d "$_c/meandre/donnees" ] && PROJET="$_c" && break
  done
fi
TRAVAIL=${TRAVAIL:-$HOME/scratch}
mkdir -p "$TRAVAIL/meandre/journaux" "$TRAVAIL/meandre/points-de-reprise"

mapfile -t LIGNES < <(grep -vE '^\s*(#|$)' "$TACHES_F")
read -r CONFIG GRAINE TAG RESTE <<< "${LIGNES[$SLURM_ARRAY_TASK_ID]}"
exec > "$TRAVAIL/meandre/journaux/province$TAG-$SLURM_ARRAY_JOB_ID.log" 2>&1
echo "tache $SLURM_ARRAY_TASK_ID : $CONFIG graine $GRAINE etiquette $TAG $RESTE ; code $(git -C "$RACINE" rev-parse --short HEAD)"

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"

# Donnees sur le disque local du noeud : bases et forcages des quatorze territoires, cache du forcage
# provincial, tables auxiliaires. Les projets PHYSITEL restent lus sur l'espace du projet.
LOCAL=$SLURM_TMPDIR/donnees
D=$PROJET/meandre/donnees
mkdir -p "$LOCAL/quebec/cache" "$LOCAL/derives/auxiliaires"
cp "$D"/quebec/*.duckdb "$D"/quebec/forcing-*-casr-brut*.nc "$D"/quebec/*.parquet "$LOCAL/quebec/"
cp "$D"/quebec/cache/forcage-*casr-brut*.npy "$LOCAL/quebec/cache/" 2>/dev/null || true
cp -r "$D"/derives/auxiliaires/. "$LOCAL/derives/auxiliaires/"
export MEANDRE_DATA="$LOCAL" MEANDRE_DERIVES="$LOCAL/derives" MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Cache de compilation partage sur scratch (2026-11-10) : sans lui, chaque tache recompilait les noyaux
# dans le disque temporaire de son noeud, environ 30 minutes avant la premiere epoque ; une epoque en
# regime prend environ 30 minutes sur l'A100, d'ou la reservation de 20 heures pour 32 epoques.
export TORCHINDUCTOR_CACHE_DIR="$TRAVAIL/meandre/cache-inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Copie du code par tache. Le dossier des points de reprise est un lien vers un dossier propre a la
# tache sur scratch : le meilleur modele y est ecrit des qu'il s'ameliore, et survit a une tache
# coupee par la limite de temps (le disque local du noeud est efface a la fin de la tache).
CODE=$SLURM_TMPDIR/code
cp -r "$RACINE" "$CODE"
cd "$CODE"
GARDE="$TRAVAIL/meandre/points-de-reprise/${TAG#-}"
mkdir -p "$GARDE"
cp "$TRAVAIL"/meandre/points-de-reprise/*.pt "$GARDE"/ 2>/dev/null || true
rm -rf .runs/quebec/checkpoints
ln -s "$GARDE" .runs/quebec/checkpoints

env PYTHONIOENCODING=utf-8 ETL_FORCE=1 $RESTE python -u -m meandre run "$CONFIG" --region quebec --epochs "${ETL_EPOCHS:-32}" --seed "$GRAINE" --tag="$TAG"
ls -la "$GARDE"
echo "fin $(date)"
