#!/usr/bin/env bash
# FILE D'EXÉCUTION SÉQUENTIELLE. Un seul entraînement à la fois : deux runs simultanés
# sur la même carte font passer une époque de 450 s à 4000-11000 s (mesuré le 9 août,
# une nuit perdue). L'attente se fait par tasklist, car pgrep depuis un shell POSIX NE
# VOIT PAS les processus Windows — c'est précisément ce qui avait fait échouer le
# garde-fou précédent.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1

attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}

lancer() {  # $1 = tag, $2... = variables
  local tag="$1"; shift
  attendre
  echo "[file] $(date +%H:%M) demarrage $tag"
  env "$@" MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_EPOCHS=30 ETL_WET=0 ETL_FORCE=1 \
      ETL_KSAT1=0.04 ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC=5e-5 \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[file] $(date +%H:%M) fini $tag : $(grep -a 'JAMAIS VUES\|HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}

# O1 — validation croisée spatiale sur OUTV : les 3 plis restants (le 0 tourne déjà).
for k in 1 2 3; do
  lancer "outv-cv${k}" ETL_REGION=outv ETL_TAG="-cv${k}" ETL_FOLD="${k}/4" ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963
done

# O6 — la couche d'expérience (codes latents) sert-elle encore, sur base saine ?
lancer "outv-sanslatent" ETL_REGION=outv ETL_TAG="-sanslatent" ETL_NO_LATENT=1 \
       ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963

# O7 — Linacre ancrée contre module ET appris (le module est le défaut ; ici on coupe
# le débiaisage pour isoler l'effet du facteur d'échelle).
lancer "outv-ds1" ETL_REGION=outv ETL_TAG="-ds1" ETL_KGW=0.0645 ETL_DEMAND_SCALE=1.0

echo "[file] $(date +%H:%M) TERMINE"
