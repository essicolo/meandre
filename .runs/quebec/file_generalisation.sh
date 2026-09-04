#!/usr/bin/env bash
# GENERALISATION DE LA RECETTE INT1 (jour 2 des 48 h, 2026-08-23).
#
# La recette qui a repare le nival sur OUTV ne contient AUCUN bouton regional :
#   - partage pluie-neige au BULBE HUMIDE, seuil unique Twb = -0.8 (R35/R36) ;
#   - fonte saisonniere amp = 0.5, sinusoide universelle (R36) ;
#   - krec APPRIS par le champ, moyenne ancree sur 2e-5 (litterature, R34) ;
#   - k_gw par le CHAMP PROVINCIAL (GP sur les recessions de 127 stations,
#     ETL_KGW_FIELD=1) plutot que le scalaire d'OUTV -- meme esprit ;
#   - ET en anomalie a 0.4 ; GRACE clim active (0.05).
# Les seuls ancrages restants sont les paquets de plateforme (ETP Linacre + fonte
# regionale calee), c'est-a-dire la loi des ancrages, pas des reglages.
#
# LE TEST : trois regions contrastees, JAMAIS utilisees pour deriver ces valeurs.
#   gasp -- maritime, la seule ou GRACE au masque differe (R40) ;
#   sagu -- nordique, fort manteau (SWE mesure 258 mm) ;
#   slno -- centre, 32 stations.
# JUGES par region : la ligne CanSWE (le nival se repare-t-il AILLEURS sans rien
# retoucher ?), decembre et avril en debit, puis le held-out contre la reference de
# la region. C'est le coeur de la these du projet : la physique + le champ appris
# doivent porter la geographie, pas des constantes par region.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}

lancer() {
  local reg="$1"; shift
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[gen] $(date +%H:%M) demarrage $reg"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION="$reg" ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/$(echo $reg | tr a-z A-Z)_LN24HA_2020" \
      ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
      ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
      ETL_TAG="-gen1" ETL_EPOCHS=30 ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-gen-${reg}.txt" 2>&1
  echo "[gen] $(date +%H:%M) fini $reg : $(grep -a 'HELD-OUT' "$JOURNAL/log-gen-${reg}.txt" | tr '\n' ' ')$(grep -a 'neige, simule/mesure' "$JOURNAL/log-gen-${reg}.txt" | tr '\n' ' ')"
}

lancer "gasp"
lancer "sagu"
lancer "slno"

echo "[gen] $(date +%H:%M) TERMINE"
