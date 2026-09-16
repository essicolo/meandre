#!/usr/bin/env bash
# UNE region = UNE tache du tableau. Les quinze sont independantes, donc elles tournent en
# parallele : la nuit de vingt heures du poste devient l'attente d'une seule region.
#
#   sbatch --account=<compte> --dependency=afterany:<entrainement> \
#          --array=0-27 quantile_ref.sbatch
#
# SECOND ETAGE : la tete de quantiles, sur l'ossature GELEE du modele entraine la nuit
# precedente. Six quantiles, 0,05 a 0,95, poses en ECARTS a la mediane, laquelle reste
# exactement le debit deterministe : le KGE du socle est preserve par construction, et seule
# l'enveloppe s'apprend. Dix epoques, la selection se faisant sur la vraisemblance.
#
# UN BRAS PAR MODELE. Temoin et variations recoivent chacun leur tete : sans cela on ne
# saurait pas si le terme des variations, qui change la forme de l'hydrogramme, change aussi
# la calibration de l'enveloppe.
#
# DEUX PIEGES EVITES, tous deux payes ailleurs. Le forcage est impose a -budyko et non laisse
# au defaut -hyb de quantile.sbatch : l'ecart entre les deux vaut 0,14 de KGE. Et le point de
# reprise est cherche dans l'espace de TRAVAIL, ou l'entrainement depose les siens, non dans
# l'espace de projet ou dorment les anciens champions.
#
#SBATCH --job-name=ref-quantile
#SBATCH --time=04:00:00
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
exec > "$JOURNAUX/q-$REG-$BRAS_N-$SLURM_ARRAY_JOB_ID.log" 2>&1
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
# Le modele deterministe de CE bras. Sans lui, rien a geler : on s'arrete plutot que de
# reentrainer un socle par megarde.
SOCLE="$TRAVAIL/meandre/checkpoints/best-${REG}-etl-ref-${BRAS_N}.pt"
if [ ! -f "$SOCLE" ]; then
  echo "ARRET : point de reprise deterministe absent, $SOCLE"
  exit 1
fi
echo "=== socle gele : $SOCLE"
SFX=-budyko
# ETL_COMPILE : la compilation du sol exige Triton, disponible dans la logitheque de
# l'Alliance (pip install --no-index triton) mais absent des environnements batis
# avant le 2026-09-01. Sans elle, une simulation SLSO prend 27 min sur A100 contre
# 6 min en local : notre colonne est dominee par le lancement de petits noyaux, le
# cas ou la compilation apporte le plus. ETL_COMPILE=0 pour s'en passer.
# RECETTE DU SOCLE, identique au poste : sol impose sauf K_sat et porosites, Linacre
# calee, seuil pluie/neige au bulbe humide, fonte regionale ancree, aquifere.
env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=$SFX ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
    ETL_WDQ=$WDQ ETL_QUANTILE=1 \
    ETL_WARM_FROM="$SOCLE" \
    ETL_TAG=-q-$BRAS_N ETL_EPOCHS=${ETL_EPOCHS:-10} \
    # Les series par station de la periode d'evaluation, AVEC les sept quantiles :
    # sans cela la tete s'entraine et rien ne sort pour juger l'enveloppe.
    ETL_DUMP_Q="$TRAVAIL/meandre/caches/q-$REG-$BRAS_N.npz" \
    python -u .runs/quebec/etl_run.py

# Les points de reprise produits redescendent dans l'espace de travail, pas dans le code.
cp -f "$RACINE"/.runs/quebec/checkpoints/best-${REG}-etl-q-${BRAS_N}.pt \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
