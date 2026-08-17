#!/usr/bin/env bash
# BALAYAGE DE LA RECHARGE (krec) a ZERO epoque, aquifere actif, socle Linacre.
# Bornes connues : krec calibre ~1e-7 -> 0.7432 (robinet ferme, gain hivernal nul) ;
# krec mesure 5e-5 -> 0.4446 (drainage excessif, effondrement). Optimum entre les deux.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
for K in 5e-7 2e-6 8e-6 2e-5; do
  attendre
  echo "[krec] $(date +%H:%M) krec=$K"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 ETL_REGION=outv ETL_EPOCHS=0 \
      ETL_TAG="-krec$K" ETL_WET=0 ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC="$K" ETL_KGW=0.0645 \
      ETL_NO_LATENT=1 ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-krec$K.txt" 2>&1
  echo "[krec] $(date +%H:%M) fini $K : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-krec$K.txt" | tr '\n' ' ')"
done
echo "[krec] $(date +%H:%M) TERMINE"
