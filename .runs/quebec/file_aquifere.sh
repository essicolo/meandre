#!/usr/bin/env bash
# DEFICIT HIVERNAL : en janvier-fevrier le socle ne produit que 0.80 et 0.655 du debit
# observe, de loin le plus gros ecart mensuel (Hydrotel fait l'erreur inverse, 1.29/1.24).
# L'eau de janvier vient de la nappe, et le socle tourne SANS aquifere parce que je
# l'avais coupe pour aligner la configuration sur celle d'Hydrotel, qui n'en a pas --
# mais Hydrotel compense par son sol calibre, nous non. Composantes coherentes : beta
# 0.919 (deficit de volume) et gamma 1.040 (trop de variabilite) = base manquante.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
run() { local tag="$1"; shift; attendre
  echo "[aq] $(date +%H:%M) $tag"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_EPOCHS=30 ETL_WET=0 ETL_WSNOW=0.3 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[aq] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
run "aq0"  ETL_TAG="-aq0"  ETL_AQUIFER=1 ETL_KREC=5e-5 ETL_KGW=0.0645 ETL_EPOCHS=0   # controle
run "aq30" ETL_TAG="-aq30" ETL_AQUIFER=1 ETL_KREC=5e-5 ETL_KGW=0.0645               # puis 30 ep
echo "[aq] $(date +%H:%M) TERMINE"
