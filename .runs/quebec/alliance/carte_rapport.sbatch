#!/usr/bin/env bash
# PASSES DE NATURALISATION AVEC LES PRELEVEMENTS ET REJETS CORRIGES.
#
# Le fichier source d'io-eau portait un double comptage des rejets, corrige par Essi le
# 2026-09-15. Ce que cela change, mesure le meme jour : le solde provincial passe de +62 a
# +9,0 m3/s, les entrees reconstruites de 1 851 a 318, leur couverture du volume preleve de
# 94 a 2,7 pour cent, et le taux de restitution median de 0,88 a 0,16. Au Saguenay le terme
# net de surface change de signe, de +1,39 a -0,14 m3/s. Les caches de naturalisation du
# rapport et de la presentation doivent donc etre refaits.
#
# CE N'EST PAS UN REENTRAINEMENT : deux passes avant par region, geree et naturalisee, sur
# les points de reprise valides. Le cout est domine par le lancement des noyaux sur 9 132
# pas journaliers en sequence. Sous Windows, ou Triton est absent, la mesure du 2026-09-15
# donne plus de trente minutes par passe. Ici le noyau du sol se compile.
#
#   sbatch --account=<compte> --array=0-14 .runs/quebec/alliance/carte_rapport.sbatch
#
# Les poids restent ceux ajustes avec les rejets doubles. L'ECART de naturalisation reste
# valide, les deux passes partageant les memes poids ; le calage, lui, a absorbe une part
# du flux fictif, et c'est un recalage qui le corrigera.
#SBATCH --job-name=carte-rapport
#SBATCH --time=02:00:00
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
JOURNAUX=$TRAVAIL/meandre/journaux
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/rapport"

REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm vaud)
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}
exec > "$JOURNAUX/carte-$REG-$SLURM_ARRAY_JOB_ID.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/"
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

# LES PRELEVEMENTS CORRIGES, REINGERES DANS LA COPIE LOCALE DE LA BASE. On ne touche pas a
# la base du projet : chaque tache travaille sur sa copie, donc deux taches ne peuvent pas
# s'ecrire dessus. Le fichier source voyage avec le paquet.
export IO_EAU=${IO_EAU:-$PROJET/meandre/donnees/io-eau-meandre.parquet}
if [ ! -f "$IO_EAU" ]; then
  echo "ARRET : fichier de prelevements introuvable a $IO_EAU"
  exit 1
fi
python .runs/quebec/ingest_withdrawals.py "$REG"

# Le point de reprise valide de la region. Il n'est pas suivi par le depot : s'il manque,
# la tache s'arrete tout de suite au lieu de mourir dans le chargement du modele.
case "$REG" in
  gasp) CK=best-gasp-etl-ds ;;  sagu) CK=best-sagu-etl-ds ;;
  mont) CK=best-mont-etl-ds ;;  outv) CK=best-outv-etl-qc ;;
  slso) CK=best-slso-etl-canon ;; slno) CK=best-slno-etl-canon ;;
  *) CK=best-gasp-etl-ds ;;
esac
if [ ! -f ".runs/quebec/checkpoints/$CK.pt" ]; then
  echo "ARRET : point de reprise .runs/quebec/checkpoints/$CK.pt absent"
  exit 1
fi
echo "=== region $REG, point de reprise $CK"

export MEANDRE_RAPPORT=$TRAVAIL/meandre/rapport
export MEANDRE_COMPILE_SOL=${MEANDRE_COMPILE_SOL:-1}
export PYTHONIOENCODING=utf-8
python -u .runs/quebec/carte_rapport.py "$REG"

ls -la "$MEANDRE_RAPPORT"/rap-${REG}-*.npz
