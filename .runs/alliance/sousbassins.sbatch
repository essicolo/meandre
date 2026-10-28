#!/usr/bin/env bash
# LE CHAMP REND-IL LES HYDROGRAMMES PLUS NETS ? Distribution sur les sous-bassins jauges.
#
# Un sous-bassin de 30 a 130 troncons est aussi lent sur un GPU que sur un processeur
# (colonne limitee par les lancements de noyaux, pas par le calcul) : la grappe ne sert
# donc pas a accelerer une epoque mais a en faire tourner soixante EN PARALLELE, sur des
# noeuds processeur qui font moins la queue. Chaque tache : un sous-bassin, cinq epoques
# sous la boucle corrigee, socle avec K_sat libre, puis KGE, rabotage des pointes et
# platitude contre l'observe, avant et apres entrainement.
#
# Forcage : tout-CaSR (-budyko), le seul present sur la grappe. Hydrotel n'y est pas
# compare (post-traitement absent), le banc le saute de lui-meme.
#
#   sbatch --account=<compte> --array=0-58%20 sousbassins.sbatch
#SBATCH --job-name=sousbassin
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
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
JOURNAUX=$TRAVAIL/meandre/sousbassins
mkdir -p "$JOURNAUX"

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"
export JOINT_FX_SUFFIX=-budyko
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# La liste des sous-bassins est produite une fois par `banc_sousbassin.py liste` et
# deposee dans $PROJET/meandre/donnees/quebec/sousbassins.txt : "region station" par ligne.
LISTE=$D/sousbassins.txt
[ -f "$LISTE" ] || { echo "ARRET : $LISTE absent (produire avec banc_sousbassin.py liste)" >&2; exit 3; }
read -r REG STA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LISTE")
[ -n "${STA:-}" ] || { echo "ARRET : pas de ligne $((SLURM_ARRAY_TASK_ID + 1)) dans $LISTE" >&2; exit 3; }
exec > "$JOURNAUX/$REG-$STA.log" 2>&1

cp "$D/${REG}.duckdb" "$LOCAL/quebec/"
cp "$D/forcing-${REG}-budyko.nc" "$LOCAL/quebec/"
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true

python -u .runs/quebec/banc_sousbassin.py "$REG" "$STA" --entrainer ${EPOQUES:-5} --device cpu --tag="-$STA"
