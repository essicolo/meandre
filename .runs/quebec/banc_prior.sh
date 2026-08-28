#!/usr/bin/env bash
# BANC APPARIE sur le poids du prior, a l'echelle provinciale.
#
# Motif : le banc de canopee a montre que la recette provinciale DIVERGE. Toutes les
# metriques se degradent de facon monotone des l'epoque 1 (val_kge 0.7149 -> 0.5226 en
# cinq epoques) pendant que la perte d'entrainement MONTE (4.57 -> 7.05). La
# decomposition dit pourquoi : prior 5045 % du total de la perte de debit, tws_clim
# 3548 %, tws 1959 %. L'optimiseur ramene le champ vers ses cibles de litterature et
# satisfait GRACE, en sacrifiant l'hydrogramme.
#
# Le garde-fou de divergence ne voit rien : il declenche sur un pic a trois fois la
# moyenne mobile, pas sur une derive lente. C'est un trou a combler separement.
#
# UN SEUL CHANGEMENT. w_prior passe de 0.005 (valeur gen1, reproduite exprès pour que
# les comparaisons portent sur l'architecture) a 1e-4, soit cinquante fois moins, ce qui
# ramenerait le terme d'environ 5000 % a environ 100 % du total : le prior redevient un
# garde-fou au lieu d'etre l'objectif. Tout le reste est identique, graine comprise.
#
# CE QUE LE BANC TRANCHE : est-ce que baisser le prior arrete la divergence ? Si le bras
# B monte au lieu de descendre, la dette #19 est chiffree et le remede est connu. Si les
# deux divergent, la cause est ailleurs -- les termes GRACE, ou la montee du taux
# d'apprentissage vers 5e-4 que le registre designe depuis quatre runs.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb
EP="${PROV_EPOCHS:-5}"
OUT=/workspace/log-prior-apparie.txt
: > "$OUT"

for P in 0.005 0.0001; do
  TAG="prior-${P}"
  LOG="/workspace/log-${TAG}.txt"
  T0=$SECONDS
  PROV_TAG="$TAG" PROV_PRIOR="$P" PROV_EPOCHS=$EP \
    python -u .runs/quebec/province.py > "$LOG" 2>&1
  {
    echo "=== w_prior=$P : $(( (SECONDS - T0) / 60 )) min ==="
    grep -aE "Epoch |composantes ponderees|mediane provinciale" "$LOG" | tail -24
    echo
  } >> "$OUT"
done
echo "[prior] TERMINE" >> "$OUT"
