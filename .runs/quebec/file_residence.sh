#!/usr/bin/env bash
# TEMPS DE RESIDENCE DE LA RESERVE LENTE (accord d'Essi, 2026-08-22).
#
# CE QUI RESTE APRES TOUT LE RESTE. On a elimine la quantite d'eau sous toutes ses
# formes : le domaine modelise est bon (R18, aire au noeud = 1.005 fois l'aire
# officielle sur 160 stations), le bilan de masse ferme a 0.01 %, les prelevements sont
# entres et ne changent rien (R17), le printemps n'est pas limite par l'apport (R20,
# l'ecoulement d'avril-mai vaut 0.31 a 0.45 de la neige plus la pluie), le manteau est
# juste (R28, 121 mm simules pour 108 mesures), et l'amplitude totale du stockage vaut
# 102 % de GRACE (R22). Reste la VITESSE a laquelle l'eau ressort.
#
# ET ELLE EST MESUREE. Le taux de vidange souterrain d'OUTV vaut 0.0273 /j (residence
# 37 j) sur 1316 recessions hivernales pures des jauges ; le champion tourne a 0.0645,
# soit 2.4 fois trop vite, et la composante lente reelle est a 0.0090 (111 j).
#
# LE ROBINET EST FERME. Sans ETL_KREC_LIBRE, `krec` est herite de la courbe de calage
# Hydrotel, ou il vaut ~1.3e-7 m/h -- chez Hydrotel c'est une FUITE jamais restituee,
# que son calage etrangle. La capacite de drainage de la couche profonde vaut alors
# krec x z3 x theta = 0.0036 mm/j : elle ne peut PAS se vider, d'ou une L3 epinglee a
# saturation (R25), une nappe a 0 mm toute l'annee (R21), et aucun endroit ou faire
# attendre l'eau plusieurs mois. C'est ecrit dans un commentaire d'etl_run.py depuis le
# 17 aout ; personne ne l'avait relie au trou d'avril.
#
# GRACE EST LE JUGE, PAS UN INGREDIENT. w_tws_clim reste a ZERO ici. Le but est de
# savoir si une reserve lente repare la PHASE ; si on mettait GRACE dans la perte on ne
# saurait plus si c'est la physique ou la contrainte qui a bouge. Le pilote imprime le
# cycle des stocks et l'audit GRACE de toute facon.
#
# CIBLES, dans cet ordre de priorite :
#   phase GRACE : mai -6 mm -> +45 ; juin -34 -> +9 ; mars +90 -> +57
#   debit       : avril 0.753 -> 1.0 ; mai 1.138 -> 1.0 ; decembre 1.235 -> 1.0
#   nappe       : 0 mm toute l'annee -> un cycle qui se remplit a l'automne
#   score       : 0.7885 au meilleur du run A. UNE BAISSE N'EST PAS UN ECHEC (R11).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

# Attendre que la file aux-A / aux-B ait rendu la carte. Deux entrainements simultanes
# font passer l'epoque de 450 s a plusieurs milliers (dette #7) ; et `pgrep`/`ps` du
# shell POSIX ne voient pas les processus Windows.
meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}
while ! grep -q "TERMINE" "$JOURNAL/log-file-aux.txt" 2>/dev/null; do sleep 300; done
while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done

lancer() {
  local tag="$1"; shift
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[res] $(date +%H:%M) demarrage $tag"
  # ETL_EPOCHS=10 : Essi, 2026-08-22, « pour ce genre de test on peut se permettre moins
  # d'epochs ». D'autant que `patience` n'atteint pas ce pilote (dette #16), donc un run
  # de 30 epoques irait jusqu'au bout meme effondre -- le run aux-A en fait la
  # demonstration en ce moment meme.
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WET=0 ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" \
      ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_WTWSCLIM=0 \
      ETL_EPOCHS=10 ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[res] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

# krec EST APPRIS PAR LE CHAMP, pas impose (Essi, 2026-08-22 : « krec devrait etre dans
# le nerf »). Une premiere version de cette file le posait uniforme et GELE aux valeurs
# 1e-5 et 3e-5 ; c'est le meme travers que le calage d'Hydrotel, traiter comme une
# constante regionale ce qui est une propriete du sous-sol, variable avec la profondeur
# du roc et les depots de surface. Corrige avant demarrage.
#
# CE QUI L'EMPECHE DE S'EFFONDRER. Libre, krec n'avait que le debit pour juge, et le
# debit seul le pousse a zero (R11) -- d'ou le reflexe de le geler. Un terme de prior a
# donc ete ajoute (`prior_on_krec`, opt-in, leve automatiquement par ETL_KREC_LIBRE sans
# ETL_KREC_GEL) : il ancre la MOYENNE GEOMETRIQUE du champ sur 2e-5 m/h et ne coute RIEN
# a la dispersion spatiale, exactement comme pour k_gw. Trois tests le verrouillent,
# dont un qui verifie qu'un champ disperse et un champ plat de meme moyenne coutent
# pareil -- penaliser la variance est le mecanisme connu du collapse du NeRF.
#
# LA CIBLE, 2e-5 m/h : a saturation q3 = krec x z3 x theta, soit ~0.51 mm/j pour
# z3 = 2.65 m et theta = 0.40, donc ~34 % d'un ecoulement de 549 mm/an. L'indice de
# debit de base des bassins boreaux quebecois se situe entre 0.4 et 0.6 : la cible est
# du bon ordre, legerement conservatrice. Le calage Hydrotel, lui, donne 1.3e-7, soit
# 0.0036 mm/j -- un robinet ferme, et la cause du reservoir vide.
#
# DEUX RUNS, UNE SEULE DIFFERENCE : le taux de vidange de la nappe. C1 le pose a la
# valeur MESUREE sur 1316 recessions hivernales pures (0.0273 /j, residence 37 j), C2 a
# la composante LENTE de ces memes recessions (0.0090 /j, 111 j). Le champion tourne a
# 0.0645. Si la phase se repare a C1 mais pas a C2, ou l'inverse, on saura si le probleme
# est le remplissage ou la duree de retention.
lancer "res-C1" ETL_TAG="-resC1" ETL_KGW=0.0273
lancer "res-C2" ETL_TAG="-resC2" ETL_KGW=0.0090

echo "[res] $(date +%H:%M) TERMINE"
