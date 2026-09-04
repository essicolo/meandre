#!/usr/bin/env bash
# SEUIL PLUIE-NEIGE DERIVE D'UNE MESURE DE MASSE (R30, 2026-08-22).
# PRETE MAIS NON LANCEE : en attente du feu vert d'Essi. Quatre runs sont deja en file.
#
# POURQUOI. Le seuil du projet vaut -2.2168 degres, herite de la plateforme Hydrotel : il
# compte comme PLUIE tout ce qui tombe a -2 degres. Trois chiffres independants du debit
# convergent contre lui :
#   - a site et a jour egaux, le modele a 0.65 a 0.72 de la neige mesuree par CanSWE,
#     donc il lui en manque ~35 % (novembre 0.66, decembre 0.55, le pire) ;
#   - le seuil qui produit exactement +35 % de neige sur OUTV est +0.3 degre
#     (cumuls nov-mars : 174 mm a -2.22, 232 a 0.0, 254 a +1.0) ;
#   - +0.3 est dans la plage que la litterature attend pour un partage pluie-neige (0 a +2).
# Un seuil a -2.2 est donc vraisemblablement une COMPENSATION calee pour le modele de
# fonte d'Hydrotel. L'importer dans meandre, qui a un autre module de fonte, c'est ancrer
# une compensation au lieu d'un processus -- ce que la loi des ancrages interdit.
#
# CE QUI EST NOUVEAU par rapport a R16, qui avait deja teste +1.0 : la valeur n'est plus
# choisie pour le score mais DERIVEE d'une mesure de masse, et elle tombe dans la plage
# physique. R16 concluait que le seuil gouverne l'axe decembre-mai sans toucher avril ;
# ici la cible n'est pas avril mais le MANTEAU lui-meme.
#
# CIBLE, et elle ne passe pas par la jauge : remonter la sortie CanSWE du pilote
# (« neige, simule/mesure par mois ») de 11=0.66 et 12=0.55 vers 1.0. Le debit suivra ou
# ne suivra pas, mais la masse de neige est une mesure, pas un score.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}
while ! grep -q "TERMINE" "$JOURNAL/log-file-res.txt" 2>/dev/null; do sleep 300; done
while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done

lancer() {
  local tag="$1"; shift
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[seuil] $(date +%H:%M) demarrage $tag"
  # ETL_SEUIL_NEIGE=0 desactive la reprise du seuil du projet ; ETL_SEUIL_VALEUR impose
  # le notre. Le reste est la recette du champion, a l'identique.
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WET=0 ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=0 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" ETL_AQUIFER=1 ETL_KGW=0.0645 \
      ETL_WTWSCLIM=0 ETL_EPOCHS=10 ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[seuil] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

# D1 : la valeur DERIVEE de la mesure de masse. C'est l'hypothese a tester.
lancer "seuil-D1" ETL_TAG="-seuilD1" ETL_SEUIL_VALEUR=0.3
# D2 : CONTROLE a la valeur du projet, meme code, memes 10 epoques. Sans lui on ne
# saurait pas separer l'effet du seuil de l'effet d'un entrainement court (le champion a
# tourne 30 epoques dans un autre etat du depot). Une ligne de controle dont on connait
# la reponse est la lecon de la dette #10.
lancer "seuil-D2" ETL_TAG="-seuilD2" ETL_SEUIL_VALEUR=-2.2168

echo "[seuil] $(date +%H:%M) TERMINE"
