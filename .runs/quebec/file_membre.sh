#!/usr/bin/env bash
# ANCRER SUR LE MEILLEUR MEMBRE, avec SA formule d'ETP (remarque d'Essi : les plateformes
# sont aussi des configurations de FONCTIONS, 5 McGuinness contre 1 Linacre, et nous
# etions ancres sur la seule Linacre qui est aussi la moins bonne, 0.7531 contre 0.8299).
# Controle a zero epoque d'abord : le socle MG24HK doit valoir plus que le socle LN24HA
# (0.7389) pour qu'il vaille la peine d'entrainer.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
run() { local tag="$1"; shift; attendre
  echo "[mbr] $(date +%H:%M) $tag"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_EPOCHS=30 ETL_WET=0 ETL_WSNOW=0.3 ETL_AQUIFER=0 ETL_NO_LATENT=1 \
      ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[mbr] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
run "mgk0"  ETL_TAG="-mgk0"  ETL_MEMBRE=MG24HK ETL_ETP=mcguinness ETL_EPOCHS=0
run "mgs0"  ETL_TAG="-mgs0"  ETL_MEMBRE=MG24HS ETL_ETP=mcguinness ETL_EPOCHS=0
run "mgk30" ETL_TAG="-mgk30" ETL_MEMBRE=MG24HK ETL_ETP=mcguinness
echo "[mbr] $(date +%H:%M) TERMINE"
