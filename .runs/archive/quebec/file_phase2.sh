#!/usr/bin/env bash
# PHASE 2 de la file de fin de semaine : le taux d'apprentissage.
#
# CE QUE LA PHASE 1 A ETABLI. Le banc auxiliaire a isole la cause de la divergence
# provinciale : la contrainte GRACE. A poids pleins, kge_med tombe de 0.3854 a 0.0726 en
# deux epoques ; degonflee cinquante fois, elle tient a 0.3087. Le prior physique est
# INERTE (0.0727 contre 0.0726), et mon diagnostic initial qui l'accusait reposait sur
# une ligne de journal ou ce terme s'imprimait brut sous un en-tete annoncant pondere.
#
# CE QUE LE RUN LONG REVELE ENSUITE. Meme GRACE degonflee, la trajectoire culmine puis
# retombe :
#   epoque   0       1       2       3       4
#   kge_med  0.3862  0.3854  0.3988  0.3887  0.3109
#   lr       1.25e-4 2.5e-4  3.75e-4 5e-4    5e-4
# Le sommet est a l'epoque 2 et la chute commence quand le rechauffement atteint son
# plafond de 5e-4. C'est le second suspect que le registre nomme depuis quatre runs.
#
# CE BANC. Un seul changement, le taux plafonne a 1.25e-4. Le bras TEMOIN existe deja :
# ce sont les six premieres epoques de long-prov, memes graine et configuration. On ne
# paie donc qu'un seul run pour une comparaison appariee.
#
# HUIT EPOQUES ET NON VINGT-CINQ. Cette recette culmine tot ; un run long est la mauvaise
# forme pour elle. On garde la selection par meilleur point de reprise et on investit le
# temps en GRAINES, puisque le registre exige trois graines avant d'annoncer un chiffre.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb
J=/workspace/file.log
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" >> "$J"; }
med(){ grep -aoE "PROVMED [0-9]+\.[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+"; }
# Sommet de kge_med sur tout le run : c'est ce que la selection retient, et c'est la
# seule quantite comparable entre deux trajectoires qui culminent a des epoques differentes.
sommet(){ grep -aoE "kge_med=[0-9.-]+" "$1" 2>/dev/null | cut -d= -f2 | sort -g | tail -1; }

run(){
  local tag="$1" ep="$2"; shift 2
  local log="/workspace/log-${tag}.txt"
  [ -f "$log" ] && grep -q "PROVMED" "$log" && { note "$tag deja fait, saute"; return; }
  note "DEBUT $tag ($ep epoques) : $*"
  env "$@" PROV_TAG="$tag" PROV_EPOCHS="$ep" \
    python -u .runs/quebec/province.py > "$log" 2>&1
  note "FIN $tag : tenue de cote $(med "$log") | sommet kge_med $(sommet "$log")"
  cp -f ".runs/quebec/checkpoints/best-${tag}.pt" /workspace/ 2>/dev/null || true
}

# ── 1. Le taux plafonne ─────────────────────────────────────────────────────
run "lr-bas" 8 PROV_AUX=0.02 PROV_LR=1.25e-4

S_PLEIN=$(sommet /workspace/log-long-prov.txt)
S_BAS=$(sommet /workspace/log-lr-bas.txt)
note "TAUX : plafond 5e-4 sommet ${S_PLEIN:-?} | plafond 1.25e-4 sommet ${S_BAS:-?}"

VARS=(PROV_AUX=0.02)
if [ -n "${S_BAS:-}" ] && [ -n "${S_PLEIN:-}" ]; then
  awk -v a="$S_PLEIN" -v b="$S_BAS" 'BEGIN{exit !(b>a)}' && VARS=(PROV_AUX=0.02 PROV_LR=1.25e-4)
fi
note "CONFIGURATION RETENUE : ${VARS[*]}"

# ── 2. Trois graines, parce qu'un chiffre sur une seule ne vaut rien ────────
# La resolution de l'instrument vaut ~0.025 de KGE : les poids du champ sont tires au
# hasard et le registre exige trois graines avec leur dispersion avant toute annonce.
# La premiere porte aussi le cache par troncon qui alimente le rapport et la carte.
run "final-g1234" 8 "${VARS[@]}" PROV_DUMP=/workspace/prov
run "final-g7"    8 "${VARS[@]}" ETL_SEED=7
run "final-g99"   8 "${VARS[@]}" ETL_SEED=99

A=$(med /workspace/log-final-g1234.txt)
B=$(med /workspace/log-final-g7.txt)
C=$(med /workspace/log-final-g99.txt)
note "TROIS GRAINES, KGE median tenu de cote 2022-2024 : ${A:-?} ${B:-?} ${C:-?}"
note "PHASE 2 TERMINEE"
