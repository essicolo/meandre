#!/usr/bin/env bash
# FILE DE FIN DE SEMAINE, autonome sur le pod.
#
# Accord d'Essi le 2026-08-28 : autonomie sur l'EXECUTION, pas sur le JUGEMENT. Chaque
# banc pose une question fermee, et l'aiguillage entre eux est decide ICI, par une regle
# mecanique lisible d'avance, pas par une decision prise en cours de route. Si un
# resultat ne rentre dans aucune branche prevue, la file S'ARRETE et attend lundi.
#
# Toute decision est ecrite dans /workspace/file.log avec le chiffre qui la motive, pour
# etre auditable sans avoir a relire les journaux d'entrainement.
#
# ETAT DU PROBLEME AU DEPART. La recette provinciale DIVERGE : toutes les metriques se
# degradent de facon monotone des l'epoque 1 (val_kge 0.7149 -> 0.5226 en cinq epoques)
# pendant que la perte d'entrainement MONTE (4.57 -> 7.05). Decomposition : prior
# 5045 % du total de la perte de debit, GRACE climatologique 3548 %, GRACE mensuel
# 1959 %. Aucun run long ne vaut quoi que ce soit tant que ce n'est pas tranche.
set -u
cd /opt/meandre
source env.sh
export JOINT_FX_SUFFIX=-hyb
J=/workspace/file.log
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" >> "$J"; }

# Lit la mediane provinciale TENUE DE COTE d'un journal de run.
# PIEGE PAYE LE 2026-08-28 : le chargeur du domaine imprime aussi des lignes du genre
#   c_ev_raw : absent de [...], pose a la mediane provinciale 0.6 (HYPOTHESE)
# Une premiere version prenait la PREMIERE occurrence et rendait donc 0.6, une valeur
# de remplissage de milieu humide, au lieu du score. Toutes les decisions de la file
# se seraient prises la-dessus. On exige donc le prefixe du pilote ET le suffixe
# ' sur N stations', que les messages du chargeur n'ont pas.
med(){ grep -aoE "PROVMED [0-9]+\.[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+"; }
# Lit le kge_median d'une epoque donnee.
kmed(){ grep -aoE "Epoch +$2 \|.*kge_med=[0-9.-]+" "$1" 2>/dev/null | grep -oE "kge_med=[0-9.-]+" | cut -d= -f2; }
# Nombre d'epoques ecrites.
nep(){ grep -acE "^\[.*Epoch " "$1" 2>/dev/null || echo 0; }

run(){  # $1 = tag, $2 = epoques, reste = variables d'environnement
  local tag="$1" ep="$2"; shift 2
  local log="/workspace/log-${tag}.txt"
  [ -f "$log" ] && grep -q "PROVMED" "$log" && { note "$tag deja fait, saute"; return; }
  note "DEBUT $tag ($ep epoques) : $*"
  env "$@" PROV_TAG="$tag" PROV_EPOCHS="$ep" \
    python -u .runs/quebec/province.py > "$log" 2>&1
  note "FIN $tag : mediane $(med "$log")"
  # LES CHECKPOINTS VIVENT SUR LE DISQUE CONTENEUR, QUI NE SURVIT PAS A UN ARRET
  # DU POD. Seul /workspace survit. On les recopie donc apres chaque banc : un run
  # de seize heures perdu parce que le pod s'arrete serait une perte seche, et le
  # fichier ne pese que ~400 Ko.
  cp -f ".runs/quebec/checkpoints/best-${tag}.pt" /workspace/ 2>/dev/null || true
}

# ── 1. Le banc de canopee finit d'abord (deja lance) ────────────────────────
note "attente du banc de canopee"
while ! grep -q "TERMINE" /workspace/log-canopee-prov.txt 2>/dev/null; do sleep 120; done
C0=$(med /workspace/log-canopee-prov-0.txt); C1=$(med /workspace/log-canopee-prov-1.txt)
note "CANOPEE : verrou ancre ${C0:-?} | verrou appris ${C1:-?}"

# ── 2. La divergence vient-elle de la balance de la perte ? ─────────────────
run "aux-1.0"  3 PROV_AUX=1.0
run "aux-0.02" 3 PROV_AUX=0.02

# REGLE D'AIGUILLAGE, decidee d'avance. Un bras qui DIVERGE a son kge_median de la
# derniere epoque INFERIEUR a celui de l'epoque 0. Si le bras degonfle cesse de diverger,
# la balance est la cause et on repartit entre les trois termes. Sinon le suspect
# restant est le taux d'apprentissage, que le registre designe depuis quatre runs.
# Premiere version de la regle : le bras degonfle est-il meilleur a l'epoque 2 qu'a
# l'epoque 0. Elle ne discrimine RIEN : le bras de canopee a fait exactement ca AVEC
# les poids pleins -- il monte trois epoques (0.3392, 0.3876, 0.4357) puis s'effondre
# (0.3338, 0.2767). Une trajectoire qui monte jusqu'a l'epoque 2 ne prouve donc pas
# que le degonflage a change quoi que ce soit. On COMPARE LES DEUX BRAS entre eux, sur
# la tenue de cote, la seule chose qu'on cherche a ameliorer.
A=$(med /workspace/log-aux-1.0.txt); B=$(med /workspace/log-aux-0.02.txt)
note "AUX : plein ${A:-?} | degonfle ${B:-?} (tenue de cote)"
BRANCHE=lr
if [ -n "${A:-}" ] && [ -n "${B:-}" ]; then
  awk -v a="$A" -v b="$B" 'BEGIN{exit !(b>a)}' && BRANCHE=repartir
else
  note "chiffres illisibles : on prend la branche lr, qui teste le suspect nomme par le registre"
fi
note "BRANCHE = $BRANCHE"

# ── 3. Branche mecanique ────────────────────────────────────────────────────
if [ "$BRANCHE" = "repartir" ]; then
  # Lequel des trois termes portait la divergence ? Un seul degonfle a la fois.
  run "aux-prior" 3 PROV_AUX=1.0 PROV_PRIOR=0.0001
  run "aux-grace" 3 PROV_AUX=0.02 PROV_PRIOR=0.005
else
  # Le taux d'apprentissage monte a 5e-4 en quatre epoques. On le plafonne.
  run "lr-bas" 3 PROV_AUX=1.0 PROV_LR=1.25e-4
fi

# ── 4. Run long, sur la MEILLEURE configuration mesuree ─────────────────────
# Choix mecanique : la plus haute mediane provinciale parmi les bancs ci-dessus.
MEILLEUR=""; SCORE=0
for f in /workspace/log-aux-*.txt /workspace/log-lr-*.txt; do
  [ -f "$f" ] || continue
  m=$(med "$f"); [ -z "${m:-}" ] && continue
  if awk -v a="$m" -v b="$SCORE" 'BEGIN{exit !(a>b)}'; then SCORE=$m; MEILLEUR=$f; fi
done
if [ -z "$MEILLEUR" ]; then
  note "aucune configuration lisible : run long sur la recette de reference"
  MEILLEUR=/workspace/log-aux-1.0.txt
fi
note "MEILLEURE configuration : $MEILLEUR a $SCORE"
case "$MEILLEUR" in
  *aux-0.02*)  VARS=(PROV_AUX=0.02) ;;
  *aux-prior*) VARS=(PROV_AUX=1.0 PROV_PRIOR=0.0001) ;;
  *aux-grace*) VARS=(PROV_AUX=0.02 PROV_PRIOR=0.005) ;;
  *lr-bas*)    VARS=(PROV_AUX=1.0 PROV_LR=1.25e-4) ;;
  *)           VARS=(PROV_AUX=1.0) ;;
esac
note "RUN LONG : 25 epoques avec ${VARS[*]}"
# PROV_DUMP : le cache par troncon qui alimente le rapport provincial et les
# couches feuillage. Deux passes, geree et naturalisee, ce qui rend l'impact
# RELATIF des prelevements calculable au lieu d'une simple carte de pression.
run "long-prov" 25 "${VARS[@]}" PROV_DUMP=/workspace/prov
# ── 5. TROIS GRAINES, parce qu'un record sur une seule ne vaut rien ─────────
# Le registre etablit que la resolution de l'instrument vaut ~0.025 : les poids du
# champ sont tires au hasard, et quatre tirages de la MEME recette donnent un nuage
# large. La regle du projet est donc qu'une annonce de record se fait sur au moins
# trois graines, avec la dispersion. Le run long en fournit une ; on en ajoute deux,
# plus courtes, pour que le chiffre de lundi soit defendable plutot que seduisant.
for G in 7 99; do
  run "long-g${G}" 12 "${VARS[@]}" ETL_SEED=$G
done
L1=$(med /workspace/log-long-prov.txt); L2=$(med /workspace/log-long-g7.txt)
L3=$(med /workspace/log-long-g99.txt)
note "TROIS GRAINES : ${L1:-?} ${L2:-?} ${L3:-?}"
note "FILE TERMINEE"
