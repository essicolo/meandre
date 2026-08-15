#!/usr/bin/env bash
# Le SOCLE (tout impose sauf K_sat et porosites, ETP Linacre calee, pas d'aquifere,
# pas de codes latents, seuil pluie/neige pose) generalise-t-il aux autres regions ?
# D'abord un CONTROLE a zero epoque par region (20 min) : on n'engage 30 epoques que la
# ou le socle atteint deja le niveau de la physique ancree.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
socle() {  # $1 = region, $2 = epoques, $3 = suffixe de tag
  local reg="$1"; local ep="$2"; local sfx="$3"
  attendre
  echo "[socle] $(date +%H:%M) $reg $ep epoque(s)"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION="$reg" ETL_EPOCHS="$ep" ETL_TAG="-socle$sfx" ETL_WET=0 ETL_WSNOW=0.3 \
      ETL_AQUIFER=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_SEUIL_NEIGE=1 \
      ETL_INIT_HYDROTEL=sauf_ks ETL_MELT_DIR="$PLAT/${reg^^}_LN24HA_2020" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${reg}-socle${sfx}.txt" 2>&1
  echo "[socle] $(date +%H:%M) fini $reg$sfx : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${reg}-socle${sfx}.txt" | tr '\n' ' ')"
}
for reg in gasp sagu slno mont; do socle "$reg" 0 "0"; done   # controles a zero epoque
echo "[socle] $(date +%H:%M) CONTROLES TERMINES"
