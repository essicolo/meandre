#!/usr/bin/env bash
# NOUVEAUX PROCESSUS NIVAUX en INFERENCE PURE (accord d'Essi, 2026-08-22 : « on
# pourrait tres bien ajouter des processus qui permettraient de capter les
# irregularites »). Fonte saisonniere (Anderson SNOW-17 / POTMELT_HBV de Raven) et
# sublimation (Kuzmin, SUBLIM_KUZMIN de Raven), toutes deux opt-in, fidelite par
# defaut verrouillee par 12 tests.
#
# POURQUOI ZERO EPOQUE D'ABORD : le juge est la sortie CanSWE du pilote (« neige,
# simule/mesure par mois », cible 11=0.66 et 12=0.55 vers 1.0, date de disparition a
# garder juste), PAS le KGE. Un balayage d'inference coute ~35 min par point contre
# 1h30 par entrainement, et la lecon du 2026-08-19 vaut toujours : diagnostic gratuit
# avant tout run couteux. Les DEUX mecanismes tirent le manteau dans le MEME sens
# (plus de neige en decembre) mais par des voies differentes ; on les mesure separement
# puis ensemble, avec la ligne de controle qui ferme la marche.
#
# GRILLE, du plus au moins prometteur :
#   E1  fonte saisonniere amp=0.5            (decembre x0.5, juin x1.5)
#   E2  fonte saisonniere amp=0.3            (dose moderee)
#   E3  sublimation Kuzmin seule             (15-40 mm/hiver attendus)
#   E4  amp=0.5 + sublimation + seuil +0.3   (les trois leviers nivaux ensemble)
#   E0  CONTROLE, rien d'active              (doit redonner 11=0.66, 12=0.55 exacts ;
#                                             si non, le protocole a bouge et tout
#                                             le balayage est a jeter -- dette #10)
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}
while ! grep -q "TERMINE" "$JOURNAL/log-file-seuil.txt" 2>/dev/null; do sleep 300; done
while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done

lancer() {
  local tag="$1"; shift
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[proc] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WET=0 ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" ETL_AQUIFER=1 ETL_KGW=0.0645 \
      ETL_TAG="-ctl" ETL_EPOCHS=0 ETL_CMP_NEIGE=1 ETL_STOCKS=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[proc] $(date +%H:%M) fini $tag : $(grep -a 'neige, simule/mesure' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

lancer "proc-E1" ETL_MELT_SAISON=0.5
lancer "proc-E2" ETL_MELT_SAISON=0.3
lancer "proc-E3" ETL_SUBLIM=1
lancer "proc-E4" ETL_MELT_SAISON=0.5 ETL_SUBLIM=1 ETL_SEUIL_NEIGE=0 ETL_SEUIL_VALEUR=0.3
lancer "proc-E0" ETL_TAG="-ctl"

echo "[proc] $(date +%H:%M) TERMINE"
