#!/usr/bin/env bash
# Series journalieres aux stations des modeles du 15 septembre, sous WSL, sans entrainement.
# Zero epoque : le pilote recharge le point de reprise du bras, simule 2022-2024 et ecrit
# stations-<region>-<bras>.npz a cote des poids. Meme configuration que reference_variations.sbatch.
#
#   wsl -e bash .runs/quebec/series_stations_ref.sh [bras...]
set -uo pipefail
source ~/venv-meandre/bin/activate
cd /mnt/c/Users/parse01/documents-locaux/GitHub/meandre
export MEANDRE_DATA=/mnt/d/meandre-data
export MEANDRE_PLATEFORMES=/mnt/c/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel
export IO_EAU=/mnt/c/Users/parse01/documents-locaux/GitHub/io-eau/data/derived/io-eau-meandre.parquet
MODELES=$MEANDRE_DATA/quebec/checkpoints-ref-2026-09-15
REGIONS=(labi cnda cndb cndc cndd cnde abit outm gasp sagu mont slno slso outv)
mkdir -p .runs/quebec/checkpoints
for BRAS_N in ${@:-variations temoin}; do
for REG in "${REGIONS[@]}"; do
  SORTIE=$MODELES/stations-$REG-$BRAS_N.npz
  [ -f "$SORTIE" ] && continue
  JOURNAL=$MODELES/stations-$REG-$BRAS_N.log
  {
  echo "=== region $REG, bras $BRAS_N, $(date +%T)"
  python .runs/quebec/ingest_withdrawals.py "$REG"
  python .runs/quebec/corriger_longueur_lacs.py "$REG" --appliquer
  cp -f "$MODELES/best-$REG-etl-ref-$BRAS_N.pt" ".runs/quebec/checkpoints/best-$REG-etl-ref-$BRAS_N.pt"
  MAJ=$(echo "$REG" | tr a-z A-Z)
  AUX=(ETL_WET=0.4)
  case "$REG" in slso|vaud|outm|labi|abit|cnd*) AUX=(ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0) ;; esac
  case "$BRAS_N" in temoin) WDQ=0 ;; variations) WDQ=1.0 ;; esac
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-budyko ETL_FORCE=1 ETL_REGION=$REG \
      ETL_COMPILE=1 \
      ETL_WSNOW=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$MEANDRE_PLATEFORMES/LN24HA/${MAJ}_LN24HA_2020" \
      "${AUX[@]}" ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
      ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
      ETL_WDQ=$WDQ \
      ETL_TAG=-ref-$BRAS_N ETL_EPOCHS=0 \
      ETL_DUMP_Q="$SORTIE" \
      python -u .runs/quebec/etl_run.py
  echo "=== fin $REG $BRAS_N, $(date +%T)"
  } > "$JOURNAL" 2>&1
  grep -h "HELD-OUT" "$JOURNAL" | head -1
done
done
