#!/usr/bin/env bash
# Le socle a zero epoque atteint deja 0.71-0.77 sur 4 regions sur 5 (mont excepte).
# 30 epoques la ou le controle est passe, pour mesurer ce que l'apprentissage ajoute
# comme il a ajoute +0.042 sur OUTV.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
for reg in gasp sagu slno; do
  attendre
  echo "[s30] $(date +%H:%M) $reg"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION="$reg" ETL_EPOCHS=30 ETL_TAG="-socle30" ETL_WET=0 ETL_WSNOW=0.3 \
      ETL_AQUIFER=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_SEUIL_NEIGE=1 \
      ETL_INIT_HYDROTEL=sauf_ks ETL_MELT_DIR="$PLAT/${reg^^}_LN24HA_2020" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${reg}-socle30.txt" 2>&1
  echo "[s30] $(date +%H:%M) fini $reg : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${reg}-socle30.txt" | tr '\n' ' ')"
done
echo "[s30] $(date +%H:%M) TERMINE"
