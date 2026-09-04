#!/usr/bin/env bash
# BRANCHEMENT DES CONTRAINTES AUXILIAIRES (demande d'Essi, 2026-08-21) :
# « branchons correctement les données auxiliaires, pas tant pour le kge que
# l'identifiabilité ». Le score n'est donc PAS le critère de succès ici ; le critère
# est la PHASE du stockage (GRACE) et la MASSE du manteau (CanSWE).
#
# Ce que l'audit a etabli avant ces runs (R22, R23, R24) :
#   - le modele stocke la bonne QUANTITE d'eau sur l'annee (146 mm contre 143 pour
#     GRACE) mais par le mauvais chemin : 121 mm de manteau et un sol inerte, quand la
#     realite met 238 mm de neige compenses par un sol qui se vide ;
#   - il relache sa reserve un mois trop tot (mai : -6 mm chez nous, +45 pour GRACE) ;
#   - la contrainte GRACE etait active mais DILUEE (jugee a l'incertitude d'un mois,
#     25 mm, quand le biais est systematique et se juge a 5.4 mm) ;
#   - la contrainte neige n'a JAMAIS existe (swe_obs perdu par with_forcing), et la
#     reparer naivement aurait EMPIRE le modele (MOD10 sous-estime sous couvert et
#     demandait de fondre plus tot) ;
#   - l'ET etait libre : la recette tourne sur Linacre, pas sur le MLP, donc aucun
#     MOD16 n'entrait dans le modele.
#
# CIBLES CHIFFREES, a comparer au champion :
#   phase GRACE  : mai -6 mm  -> +45 mm ; juin -34 -> +9 ; mars +90 -> +57
#   masse neige  : pic mars 121 mm -> ~238 mm (CanSWE)
#   debit        : avril 0.753 -> 1.0 ; mai 1.138 -> 1.0
#   score        : 0.8088 en controle. UNE BAISSE N'EST PAS UN ECHEC (R11 : le debit
#                  seul prefere une physique irrealiste ; l'arbitrage score/realisme
#                  doit etre explicite, pas subi).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

# Ne compter que NOS processus : Positron et l'environnement rqh d'Essi font tourner du
# python en permanence, et `pgrep`/`ps` du shell POSIX ne voient pas les processus
# Windows (dette #7, quatre incidents de contention en trois jours).
meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}
attendre() {
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
}

lancer() {
  local tag="$1"; shift
  attendre
  echo "[aux] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" ETL_AQUIFER=1 ETL_KGW=0.0645 \
      ETL_EPOCHS=30 ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[aux] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

# A : le branchement approuve par Essi. GRACE climatologique + correction du stockage
# vu par la perte (z1 reel, wet_vol au lieu du champ mort) + ET en mode ANOMALIE.
# L'ET en niveau est ecartee : le modele evapore 422 mm/an, dans la fourchette boreale
# 400-500 des tours a flux, quand MOD16 impliquerait 539. On prend la FORME, pas le
# niveau. Trois changements a la fois : c'est une sonde de DIRECTION, pas une
# attribution -- si ca marche, il faudra ablater.
lancer "aux-A" ETL_TAG="-auxA" ETL_WET=1

# B : A MOINS le terme climatologique GRACE. C'est l'ABLATION du morceau sur lequel
# portait la question d'Essi, et le seul moyen de savoir ce qu'il a fait dans A.
#
# LE PLAN INITIAL ETAIT « A + masse CanSWE », ABANDONNE le 2026-08-21 avant demarrage.
# Motif : le forcage ne livre que 174 mm de neige (mediane nov-mars, seuil du projet
# -2.2168 degres) sur OUTV, quand CanSWE mesure un pic de 238. La cible DEPASSE la neige
# disponible : le modele ne peut pas l'atteindre sans violer son bilan de masse, et la
# contrainte l'aurait pousse contre un mur. L'ecart CanSWE-forcage est un probleme de
# FORCAGE (sous-captage de la precipitation solide, 20-50 % par vent, bien documente) ou
# de representativite des sites, pas de physique du modele. La masse CanSWE reste cablee
# et testee ; elle attend d'etre posee en ANOMALIE (la forme, pas le niveau) comme l'ET,
# ou que le sous-captage soit corrige en amont.
lancer "aux-B" ETL_TAG="-auxB" ETL_WET=1 ETL_WTWSCLIM=0

echo "[aux] $(date +%H:%M) TERMINE"
