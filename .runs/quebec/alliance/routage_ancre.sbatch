#!/usr/bin/env bash
# EPREUVE APPARIEE DE L'ANCRAGE GEOMETRIQUE DU ROUTAGE.
#
# Calque de region.sbatch : meme rangement, memes modules, meme recette du socle. Seules
# changent les bornes du temps de transfert Muskingum et l'activation de l'ancre.
#
#   sbatch --account=<compte> --array=0-5 .runs/quebec/alliance/routage_ancre.sbatch
#
# CE QUE LA MESURE DU 2026-09-14 A ETABLI, sans entrainement :
#   le modele bouge deux fois moins vite que la riviere d'un jour a l'autre, 0,54 en
#     mediane sur 129 stations, et lisser l'observation ameliore la correlation sur
#     95 pour cent d'entre elles ;
#   la colonne produit une eau TROIS fois plus nerveuse que l'observee et le reseau en
#     retire 87 pour cent : le sur-lissage nait entierement dans le canal ;
#   retirer l'attenuation du canal porte la nervosite de 0,40 a 2,12 et le rapport des
#     pointes de 0,60 a 1,34 ;
#   le temps de transfert appris vaut 23,96 h avec un ecart-type de 0,3 h sur 2 212
#     troncons et le coefficient de ponderation 0,202 a 0,006 pres : les deux sont
#     CONSTANTS, la ligne de poids qui les produit ayant une norme de 0,058 contre 0,878
#     pour la conductivite a saturation. Le routage est reste a son initialisation ;
#   le troncon de RIVIERE median mesure 5,7 km au Saguenay et 3,6 km en Gaspesie, une
#     fois ecartes les noeuds de lac dont la colonne de longueur porte un perimetre de
#     rive, et son temps de parcours physique vaut environ une heure : l'init constante a
#     24 h est vingt-cinq fois trop longue et la borne basse de 4 h quatre fois.
#
# TROIS BRAS, deux regions bien instrumentees, meme graine :
#   A temoin   bornes 4,48,24, la recette actuelle
#   B bornes   bornes 0.2,48,2 SANS ancre, pour separer l'effet de la borne seule
#   C ancre    bornes 0.2,48,2 AVEC l'ancre geometrique K = L / c
# Le bras B est indispensable : sans lui, un gain de C ne se distingue pas du simple fait
# d'autoriser des temps courts.
#
# LE JUGE N'EST PAS LE KGE. Mesure du 2026-09-14 : retirer l'attenuation gagne 0,006 de
# KGE en cassant la correlation de 0,139 et en corrigeant les pointes de 0,74, les deux
# effets se compensant. On juge sur la nervosite, la correlation, le rapport des pointes
# et le rapport des variabilites.
#SBATCH --job-name=routage-ancre
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
JOURNAUX=$TRAVAIL/meandre/journaux
mkdir -p "$JOURNAUX" "$TRAVAIL/meandre/checkpoints"

REGIONS=(sagu gasp)
BRAS=(temoin bornes ancre)
REG=${REGIONS[$((SLURM_ARRAY_TASK_ID / 3))]}
BRAS_N=${BRAS[$((SLURM_ARRAY_TASK_ID % 3))]}
exec > "$JOURNAUX/routage-$REG-$BRAS_N-$SLURM_ARRAY_JOB_ID.log" 2>&1

module purge && module load StdEnv/2023 gcc arrow mpi4py python/3.11 scipy-stack
source "$HOME/venv-meandre/bin/activate"
cd "$RACINE"
mkdir -p .runs/quebec/checkpoints

LOCAL=$SLURM_TMPDIR/donnees
mkdir -p "$LOCAL/quebec"
D=$PROJET/meandre/donnees/quebec
cp "$D/${REG}.duckdb" "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/forcing-${REG}*.nc "$LOCAL/quebec/" 2>/dev/null || true
cp "$D"/*.parquet "$LOCAL/quebec/" 2>/dev/null || true
# Le dossier de reference de l'evapotranspiration : le pilote le charge des que w_et > 0,
# et son absence tue la tache apres deux minutes sur un FileNotFoundError. Omis le
# 2026-09-14 en degraissant region.sbatch, ce qui a coute une soumission.
cp -r "$D/checkpoints-etbench" "$LOCAL/quebec/" 2>/dev/null || true
export MEANDRE_DATA="$LOCAL"
export MEANDRE_PLATEFORMES="$PROJET/meandre/plateformes"

MAJ=$(echo "$REG" | tr a-z A-Z)
case "$BRAS_N" in
  temoin) KMUSK="4,48,24"  ; ANCRE=0 ;;
  bornes) KMUSK="0.2,48,2" ; ANCRE=0 ;;
  ancre)  KMUSK="0.2,48,2" ; ANCRE=1 ;;
esac
echo "=== bras $BRAS_N, region $MAJ, MEANDRE_KMUSK=$KMUSK, ancre=$ANCRE"

# CONTROLE DES UNITES AVANT D'ENTRAINER. Le 2026-09-15, une tache a tourne en annoncant une
# longueur de troncon absente : le graphe etait sur le GPU et la lecture echouait dans un
# except muet. Le meme jour, la mesure a montre que la colonne de longueur porte un perimetre
# de rive sur les noeuds de lac, 461 km en mediane contre 5,7 km sur un troncon de riviere.
# Ces deux controles coutent quelques secondes et repondent avant six heures de calcul.
if [ "$ANCRE" = "1" ]; then
  python .runs/quebec/verifier_ancrage.py || exit 1
  python .runs/quebec/verifier_ancrage_region.py "$REG" || exit 1
fi

env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG \
    ETL_COMPILE=${ETL_COMPILE:-1} MEANDRE_KMUSK="$KMUSK" ETL_ROUTAGE_ANCRE=$ANCRE \
    ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
    ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
    ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 ETL_SEED=1234 \
    ETL_TAG="-rt-$BRAS_N" ETL_EPOCHS=${ETL_EPOCHS:-20} \
    python -u .runs/quebec/etl_run.py

cp -f "$RACINE"/.runs/quebec/checkpoints/best-${REG}-etl-rt-${BRAS_N}.pt \
      "$TRAVAIL/meandre/checkpoints/" 2>/dev/null || true
