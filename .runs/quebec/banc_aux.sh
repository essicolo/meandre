#!/usr/bin/env bash
# BANC APPARIE sur les CONTRAINTES AUXILIAIRES, a l'echelle provinciale.
#
# Motif : le banc de canopee a montre que la recette provinciale DIVERGE. Toutes les
# metriques se degradent de facon monotone des l'epoque 1 (val_kge 0.7149 -> 0.5226 en
# cinq epoques) pendant que la perte d'entrainement MONTE (4.57 -> 7.05). La
# decomposition dit pourquoi : prior 5045 % du total de la perte de debit, GRACE
# climatologique 3548 %, GRACE mensuel 1959 %. L'optimiseur ramene le champ vers ses
# cibles de litterature et satisfait GRACE, en sacrifiant l'hydrogramme.
#
# LE GARDE-FOU NE VOIT RIEN : il declenche sur un pic a trois fois la moyenne mobile,
# pas sur une derive lente. Trou a combler separement, PAS pendant ce banc -- changer un
# garde-fou au milieu d'une comparaison appariee l'invaliderait.
#
# EN BLOC, ET C'EST DELIBERE. Degonfler le prior seul laisserait deux termes a plus de
# vingt fois le debit, et un resultat mitige ne trancherait rien. Les trois termes sont
# le MEME phenomene : des contraintes qui optimisent a la place de l'hydrogramme. La
# question posee est donc binaire -- la balance de la perte est-elle la cause de la
# divergence. Si oui, repartir entre les trois est un banc court et facile. Si non, le
# suspect restant est la montee du taux d'apprentissage vers 5e-4, que le registre
# designe depuis quatre runs.
#
# TROIS EPOQUES ET NON CINQ. La degradation est monotone des l'epoque 1, sans rebond :
# le signe est lisible a l'epoque 2, et les epoques 3 et 4 ne font que confirmer une
# pente etablie en coutant la moitie du banc.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb
EP="${PROV_EPOCHS:-3}"
OUT=/workspace/log-aux-apparie.txt
: > "$OUT"

for A in 1.0 0.02; do
  TAG="aux-${A}"
  LOG="/workspace/log-${TAG}.txt"
  T0=$SECONDS
  PROV_TAG="$TAG" PROV_AUX="$A" PROV_EPOCHS="$EP" \
    python -u .runs/quebec/province.py > "$LOG" 2>&1
  {
    echo "=== PROV_AUX=$A : $(( (SECONDS - T0) / 60 )) min ==="
    grep -aE "auxiliaires degonflees|Epoch |composantes ponderees|mediane provinciale" "$LOG" | tail -20
    echo
  } >> "$OUT"
done
echo "[aux] TERMINE" >> "$OUT"
