#!/usr/bin/env bash
# PHASE 3 : les trois graines, sur la configuration que le TENU DE COTE designe.
#
# POURQUOI CETTE PHASE REMPLACE LA FIN DE LA PHASE 2. La regle de selection de la phase 2
# choisissait sur le SOMMET de kge_med en validation, et a retenu le taux plein :
#   plafond 5e-4    : sommet kge_med 0.3988, tenue de cote 0.4277
#   plafond 1.25e-4 : sommet kge_med 0.3928, tenue de cote 0.4513
# Elle a donc retenu la configuration qui perd 0.0236 de KGE median sur les trois annees
# tenues de cote, pour un avantage de 0.006 sur une metrique de SELECTION. C'est
# exactement ce que CLAUDE.md interdit : les metriques de dev servent a selectionner, et
# seule la tenue de cote 2022-2024 compte. Regle corrigee, banc relance.
#
# LA PRECISION DU CHIFFRE, mesuree et non supposee. Deux configurations qui ne different
# que par un terme inerte (aux-1.0 et aux-prior) rendent 0.4266 et 0.4267 de tenue de
# cote : la comparaison APPARIEE, a graine fixee et un seul changement, est precise a
# 0.0001. Les 0.0236 du plafond de taux sont donc largement au-dessus du bruit. A ne pas
# confondre avec la resolution de ~0.025 du registre, qui mesure la dispersion entre
# GRAINES et s'applique aux comparaisons NON appariees.
#
# CE QUE LE PLAFOND FAIT VRAIMENT. Il ne monte pas le sommet, il empeche la chute et
# permet une REMONTEE : kge_med passe par un creux a 0.3607 a l'epoque 3 puis remonte a
# 0.3878 a l'epoque 6, alors que le taux plein tombe a 0.3109 et n'en revient pas.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb
J=/workspace/file.log
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" >> "$J"; }
med(){ grep -aoE "PROVMED [0-9]+\.[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+"; }

run(){
  local tag="$1" ep="$2"; shift 2
  local log="/workspace/log-${tag}.txt"
  [ -f "$log" ] && grep -q "PROVMED" "$log" && { note "$tag deja fait, saute"; return; }
  note "DEBUT $tag ($ep epoques) : $*"
  env "$@" PROV_TAG="$tag" PROV_EPOCHS="$ep" \
    python -u .runs/quebec/province.py > "$log" 2>&1
  note "FIN $tag : tenue de cote $(med "$log")"
  cp -f ".runs/quebec/checkpoints/best-${tag}.pt" /workspace/ 2>/dev/null || true
}

# GRACE degonflee cinquante fois ET taux plafonne : les deux verdicts de la fin de
# semaine, ensemble. La premiere graine porte le cache par troncon qui alimente le
# rapport provincial et les couches de la carte.
CFG=(PROV_AUX=0.02 PROV_LR=1.25e-4)
run "prov-g1234" 8 "${CFG[@]}" PROV_DUMP=/workspace/prov
run "prov-g7"    8 "${CFG[@]}" ETL_SEED=7
run "prov-g99"   8 "${CFG[@]}" ETL_SEED=99

A=$(med /workspace/log-prov-g1234.txt)
B=$(med /workspace/log-prov-g7.txt)
C=$(med /workspace/log-prov-g99.txt)
note "TROIS GRAINES, KGE median tenu de cote 2022-2024 : ${A:-?} ${B:-?} ${C:-?}"
note "PHASE 3 TERMINEE"
