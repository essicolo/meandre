#!/usr/bin/env bash
# ENCHAINEMENT de fin de semaine sur le pod, sans intervention.
#
# Motif (Essi, 2026-08-28 : « mettons la gomme [...] j'aimerais pouvoir lancer une
# simulation pour obtenir des resultats lundi »). Le banc apparie du verrou de fonte
# tourne deja ; il finit vers minuit. Ensuite le pod serait IDLE et facture. On attend
# donc sa fin, on lit lequel des deux bras gagne, et on relance un run LONG avec le
# gagnant, jusqu'a epuisement du credit.
#
# LE CREDIT EST LE VRAI PLAFOND : environ 27 heures a 0.451 $/h. Le run long sera donc
# coupe en vol samedi. Ce n'est pas grave -- le meilleur checkpoint est ecrit a chaque
# amelioration -- mais il ne faut pas attendre un entrainement termine.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb

BANC=/workspace/log-canopee-prov.txt
echo "[chaine] attente de la fin du banc apparie" >> /workspace/chaine.log
while ! grep -q "TERMINE" "$BANC" 2>/dev/null; do sleep 120; done

# Lequel gagne ? On lit la mediane provinciale de chaque bras. En cas de doute on prend
# le verrou APPRIS : c'est l'hypothese a l'essai, et le temoin est deja documente.
S0=$(grep -A30 "PROV_CANOPEE=0" "$BANC" | grep -oE "mediane provinciale [0-9.]+" | head -1 | grep -oE "[0-9.]+")
S1=$(grep -A30 "PROV_CANOPEE=1" "$BANC" | grep -oE "mediane provinciale [0-9.]+" | head -1 | grep -oE "[0-9.]+")
GAGNANT=1
if [ -n "${S0:-}" ] && [ -n "${S1:-}" ]; then
  awk -v a="$S0" -v b="$S1" 'BEGIN{exit !(a>b)}' && GAGNANT=0
fi
echo "[chaine] bras 0 = ${S0:-?} | bras 1 = ${S1:-?} | run long avec PROV_CANOPEE=$GAGNANT" \
  >> /workspace/chaine.log

PROV_TAG=long-prov PROV_CANOPEE=$GAGNANT PROV_EPOCHS=40 \
  python -u .runs/quebec/province.py > /workspace/log-long-prov.txt 2>&1
echo "[chaine] run long TERMINE ou coupe" >> /workspace/chaine.log
