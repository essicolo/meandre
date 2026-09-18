#!/usr/bin/env bash
# Series journalieres aux stations des modeles du 15 septembre, sans aucun entrainement.
# Les caches par troncon de la nuit ne portent pas ces series ; les hydrogrammes et la carte
# des KGE par station en ont besoin. Chaque tache recharge le point de reprise de son bras,
# simule 2022-2024 et ecrit stations-<region>-<bras>.npz. Zero epoque : rien n'est sauvegarde.
#
#   sbatch --account=<compte> --array=0-27 series_stations_ref.sbatch
#SBATCH --job-name=ref-stations
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
RACINE=${MEANDRE_RACINE:-$HOME/meandre}
# Le groupe primaire n'est pas toujours celui de l'allocation (id -gn donne "atlas01"
# la ou le dossier s'appelle "def-atlas01") : on cherche le dossier qui porte les
# donnees plutot que de deviner son nom. Une tache est morte en douze secondes sur
# cette devinette le 2026-09-02.
if [ -z "${PROJET:-}" ]; then
  for _c in "$HOME"/projects/*/"$USER"; do
    [ -d "$_c/meandre/donnees" ] && PROJET="$_c" && break
  done
  PROJET=${PROJET:-$HOME/projects/$(id -gn)/$USER}
fi
TRAVAIL=${TRAVAIL:-$HOME/scratch}
JOURNAUX=$TRAVAIL/meandre/journaux
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints"
# VAUDREUIL EST ABSENT DU TABLEAU : la region ne porte aucune station et aucune
# observation, donc l'entrainement n'a pas de cible. Son modele est TRANSFERE, pas cale, et
# ses caches de naturalisation se produisent a part.
REGIONS=(outv gasp sagu mont slno slso abit cnda cndb cndc cndd cnde labi outm)
BRAS=(temoin variations)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID / 2))]}
BRAS_N=${BRAS[$((SLURM_ARRAY_TASK_ID % 2))]}
exec > "$JOURNAUX/stations-$REG-$BRAS_N-$SLURM_ARRAY_JOB_ID.log" 2>&1
echo "=== region $REG, bras $BRAS_N"
mkdir -p "$TRAVAIL/meandre/caches"

# arrow fournit pyarrow et mpi4py son homonyme : a l'Alliance ils viennent de MODULES,
# jamais de pip, et le module doit etre charge AVANT l'activation de l'environnement.
module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"
# git ne versionne pas les dossiers vides : sans cette ligne, une tache s'entraine
# vingt minutes puis meurt au moment de sauvegarder son point de reprise.
mkdir -p .runs/quebec/checkpoints

# Les donnees vivent dans l'espace de projet ; on les copie sur le disque LOCAL du noeud,
# beaucoup plus rapide et sans concurrence entre les taches du tableau.
LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-casr-corr.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
# SLSO est une exception du chargeur : sa variante corrigee de CaSR vit dans un dossier
# a elle, hors de quebec/. Sans cette copie la tache echoue apres deux minutes de mise
# en place, sur un chemin qui n'existe pas dans le disque local du noeud.
mkdir -p "$LOCAL/slso"
cp "$D/forcing-casr-corr.nc" "$LOCAL/slso/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

# LES DEUX CORRECTIONS DU 2026-09-15, sur la copie LOCALE.
export IO_EAU=${IO_EAU:-$PROJET/meandre/donnees/io-eau-meandre.parquet}
if [ ! -f "$IO_EAU" ]; then
  echo "ARRET : fichier de prelevements introuvable a $IO_EAU"
  exit 1
fi
python .runs/quebec/ingest_withdrawals.py "$REG"
python .runs/quebec/corriger_longueur_lacs.py "$REG" --appliquer

MAJ=$(echo "$REG" | tr a-z A-Z)
# Poids auxiliaires a zero la ou les observations satellitaires ne sont pas en cache : la
# construction de la perte refuse un poids sans sa cible.
AUX=(ETL_WET=0.4)
case "$REG" in slso|vaud|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
case "$BRAS_N" in
  temoin)     WDQ=0 ;;
  variations) WDQ=1.0 ;;
esac
SOCLE="$TRAVAIL/meandre/checkpoints/best-${REG}-etl-ref-${BRAS_N}.pt"
if [ ! -f "$SOCLE" ]; then
  echo "ARRET : point de reprise absent, $SOCLE"
  exit 1
fi
# Le pilote recharge le point de reprise qui porte son etiquette ; on lui en donne une copie.
cp -f "$SOCLE" ".runs/quebec/checkpoints/best-${REG}-etl-ref-${BRAS_N}.pt"
echo "=== modele evalue : $SOCLE"
SFX=-budyko
# Aucun commentaire a l'interieur de la commande continuee : une ligne de commentaire la termine.
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=$SFX ETL_FORCE=1 ETL_REGION=$REG     ETL_COMPILE=${ETL_COMPILE:-1}     ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks     ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020"     "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5     ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1     ETL_WDQ=$WDQ     ETL_TAG=-ref-$BRAS_N ETL_EPOCHS=0     ETL_DUMP_Q="$TRAVAIL/meandre/caches/stations-$REG-$BRAS_N.npz"     python -u .runs/quebec/etl_run.py

ls -la "$TRAVAIL/meandre/caches/stations-$REG-$BRAS_N.npz"
