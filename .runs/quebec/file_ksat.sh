#!/usr/bin/env bash
# Le prior K_sat=0.04 m/j est 8x sous la valeur physique (loam = 0.0132 m/h = 0.317 m/j)
# et le champ ne s'en decolle pas (appris 0.0373). Il a ete adopte le 21 juillet pour
# compenser une fuite de masse et une ETR partielle, corrigees depuis : CADUC.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}
lancer() {
  local tag="$1"; shift
  attendre
  echo "[ksat] $(date +%H:%M) demarrage $tag"
  env MEANDRE_ABANDON=4:0.55 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_EPOCHS=30 ETL_WET=0 ETL_FORCE=1 \
      ETL_REGION=outv ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC=5e-5 \
      ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[ksat] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}
lancer "outv-ksatlit"  ETL_TAG="-ksatlit"                      # sans ETL_KSAT1 : init litterature
lancer "outv-ksatphys" ETL_TAG="-ksatphys" ETL_KSAT1=0.317      # valeur texture loam d'Hydrotel
echo "[ksat] $(date +%H:%M) TERMINE"
