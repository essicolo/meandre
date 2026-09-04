#!/usr/bin/env bash
# INTEGRATION DES GAGNANTS (2026-08-22 soir, 48 h d'autonomie d'Essi).
#
# Ce que chaque banc a etabli aujourd'hui :
#   R33  le terme GRACE climatologique ne mord que si krec est APPRIS (vanne dans le
#        graphe de gradient) -- jamais teste ensemble jusqu'ici ;
#   R34  krec appris + k_gw mesure = premiere nappe non nulle, -16 % de residu de
#        phase, mais nappe PLATE (la recharge est constante tant que L3 ne respire pas) ;
#   R35  le seuil pluie-neige derive de CanSWE efface le deficit d'accumulation ET bat
#        le controle au KGE (0.8014 contre 0.7905) ; mai deborde a 1.292 ;
#   E    la fonte saisonniere amp=0.5 degonfle mai (manteau 1.14 -> 0.88) en remontant
#        decembre ; la sublimation Kuzmin brute est REJETEE (mange le manteau, un vent
#        de plaine n'a pas cours sous 74 % de foret) ;
#   Twb  le seuil AIR +0.3 est REGIONAL ; son equivalent en BULBE HUMIDE tient dans
#        [-1.0, -0.7] sur les 6 regions (Jennings 2018 : la variance des seuils air est
#        l'humidite). On integre donc le seuil en Twb = -0.8, PAS en air -- remarque
#        d'Essi : « il faudrait trouver un moyen de generaliser le seuil ».
#
# F1 VALIDE d'abord (6 min, inference pure) que Twb -0.8 reproduit sur OUTV le gain de
# neige du seuil air +0.3. Puis INT1 entraine 30 epoques avec TOUT : seuil Twb, fonte
# saisonniere, krec appris (moyenne ancree), k_gw mesure, terme GRACE actif (il peut
# enfin mordre), ET en anomalie a poids reduit (R31 : 1.0 poussait +33 mm/an, on vise
# le tiers). patience=8 est enfin effective (dette #16 close).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}
while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done

lancer() {
  local tag="$1"; local epochs="$2"; shift 2
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[int] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" \
      ETL_EPOCHS="$epochs" ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[int] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')$(grep -a 'neige, simule/mesure' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

# F1 : bulbe humide seul, inference pure sur le champion. Attendu : ratios CanSWE de
# debut d'hiver proches de ceux du seuil air +0.3 en inference (E4 sans le reste :
# l'ordre de 0.85-0.95 en novembre-decembre). ETL_SEUIL_NEIGE=0 coupe le seuil projet.
lancer "int-F1" 0 ETL_TAG="-ctl" ETL_WET=0 ETL_AQUIFER=1 ETL_KGW=0.0645 \
    ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8

# INT1 : tout ensemble, 30 epoques. GRACE cette fois DANS la perte (w_tws_clim de la
# config, 0.05) puisque krec est appris : c'est le test que R33 reclame.
lancer "int-INT1" 30 ETL_TAG="-int1" ETL_WET=0.4 \
    ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
    ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW=0.0273

echo "[int] $(date +%H:%M) TERMINE"
