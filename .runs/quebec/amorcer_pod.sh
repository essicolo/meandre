#!/usr/bin/env bash
# AMORCAGE d'un pod loue, en une commande, depuis des archives publiques.
#
# Motif (Essi, 2026-08-26 : « je pourrais placer tout ca sur pcloud, dans un dossier
# public ? »). Un pod est ephemere et sa carte n'est pas garantie disponible au
# redemarrage -- six creations ont echoue avant d'en obtenir une. Tirer les donnees
# d'un lien public va de centre a centre, a des dizaines de Mo/s, au lieu de les
# remonter depuis une liaison domestique, et survit a la destruction du pod.
#
# OU LES DONNEES ATTERRISSENT, et pourquoi ce n'est pas /workspace. Le volume monte
# sur /workspace est un systeme de fichiers RESEAU (mfs). Le forcage provincial y est
# lu en memoire projetee, une tranche par chunk : chaque lecture partirait sur le
# reseau. Les donnees vont donc sur le disque du conteneur, et seuls les checkpoints
# et les journaux, petits et ecrits rarement, restent sur /workspace ou ils survivent
# a un arret du pod.
#
# TROIS VOIES D'ACCES.
#   1. DOSSIER PUBLIC pCloud (choix d'Essi, 2026-08-26 : « c'est sans risque et plus
#      simple »). Aucun secret ne voyage, rien a revoquer, rien a saisir.
#        PCLOUD_CODE=XZabc123 bash amorcer_pod.sh
#      Le code est ce qui suit `code=` dans le lien de partage
#      https://u.pcloud.link/publink/show?code=XZabc123
#      ATTENTION : un lien public pCloud n'est PAS un repertoire ou l'on accole un nom
#      de fichier. Il faut passer par leur API de lien public, qui ne demande aucune
#      authentification : `showpublink` liste le contenu et donne un identifiant par
#      fichier, `getpublinkdownload` rend le lien de telechargement. C'est ce que fait
#      `lien_public` plus bas ; la version naive BASE/nom_de_fichier ne fonctionne pas.
#   2. Jeton pCloud, si le dossier doit rester prive :
#        PCLOUD_TOKEN=... PCLOUD_DIR=/meandre bash amorcer_pod.sh
#      (obtenir le jeton sans exposer le mot de passe : .runs/quebec/jeton_pcloud.py)
#   3. URL quelconque servant les archives cote a cote :
#        BASE=https://... [MDP=motdepasse] bash amorcer_pod.sh
set -euo pipefail

BASE="${BASE:-}"
PCLOUD_TOKEN="${PCLOUD_TOKEN:-}"
PCLOUD_CODE="${PCLOUD_CODE:-}"
PCLOUD_DIR="${PCLOUD_DIR:-/meandre}"
if [ -z "$BASE" ] && [ -z "$PCLOUD_TOKEN" ] && [ -z "$PCLOUD_CODE" ]; then
  echo "poser PCLOUD_CODE=<code du lien public>, ou PCLOUD_TOKEN, ou BASE=<url>" >&2
  exit 1
fi
RACINE="${RACINE:-/opt/meandre}"          # disque conteneur, pas /workspace
PERSIST="${PERSIST:-/workspace}"          # volume reseau : checkpoints et journaux
CURL=(curl -fsSL --retry 5 --retry-delay 5)
[ -n "${MDP:-}" ] && CURL+=(-u ":$MDP")

# Resolution de l'hote API : un compte pCloud vit soit en Europe soit aux Etats-Unis,
# et interroger le mauvais hote renvoie une erreur d'authentification trompeuse. On
# teste, plutot que de faire deviner.
resoudre_hote() {
  for h in ${PCLOUD_HOST:-eapi.pcloud.com api.pcloud.com}; do
    if curl -fsS "https://$h/userinfo?access_token=$PCLOUD_TOKEN" 2>/dev/null          | grep -q '"result": *0'; then echo "$h"; return 0; fi
  done
  echo "jeton pCloud refuse par eapi et api" >&2; return 1
}

# Lien de telechargement temporaire pour un fichier du compte. L'API rend un hote et
# un chemin separes, a recomposer.
lien_pcloud() {
  local rep
  rep=$(curl -fsS "https://$HOTE/getfilelink"         --data-urlencode "path=$PCLOUD_DIR/$1"         --data-urlencode "access_token=$PCLOUD_TOKEN")
  echo "$rep" | grep -q '"result": *0' || { echo "pCloud : $rep" >&2; return 1; }
  python3 -c "import json,sys; d=json.load(sys.stdin); print('https://'+d['hosts'][0]+d['path'])" <<< "$rep"
}
# Lien de telechargement pour un fichier d'un dossier PUBLIC, sans authentification.
# Deux appels : le contenu du dossier (pour retrouver l'identifiant du fichier par son
# nom), puis le lien de telechargement.
declare -A FICHIERS
inventaire_public() {
  local rep
  for h in ${PCLOUD_HOST:-eapi.pcloud.com api.pcloud.com}; do
    rep=$(curl -fsS "https://$h/showpublink?code=$PCLOUD_CODE" 2>/dev/null) || continue
    echo "$rep" | grep -q '"result": *0' || continue
    HOTE="$h"
    while IFS=$'	' read -r nom id; do FICHIERS["$nom"]="$id"; done < <(
      python3 -c "
import json,sys
d=json.load(sys.stdin)['metadata']
for f in (d.get('contents') or [d]):
    if not f.get('isfolder'): print(f['name'], f['fileid'], sep='	')
" <<< "$rep")
    echo "[amorce] dossier public sur $h : ${#FICHIERS[@]} fichiers"
    return 0
  done
  echo "code de lien public refuse par eapi et api" >&2; return 1
}
lien_public() {
  local id="${FICHIERS[$1]:-}"
  [ -n "$id" ] || { echo "archive $1 absente du dossier public" >&2; return 1; }
  curl -fsS "https://$HOTE/getpublinkdownload?code=$PCLOUD_CODE&fileid=$id"     | python3 -c "import json,sys; d=json.load(sys.stdin); print('https://'+d['hosts'][0]+d['path'])"
}
if [ -n "$PCLOUD_CODE" ]; then inventaire_public
elif [ -n "$PCLOUD_TOKEN" ]; then HOTE=$(resoudre_hote); echo "[amorce] API pCloud sur $HOTE"; fi

mkdir -p "$RACINE/meandre-data/quebec" "$RACINE/plateformes-hydrotel" \
         "$PERSIST/checkpoints" "$PERSIST/journaux"
cd "$RACINE"

tirer() {  # $1 = archive, $2 = destination, $3 = drapeau tar
  local a="$1" d="$2" f="$3" url
  if   [ -n "$PCLOUD_CODE" ];  then url=$(lien_public "$a")
  elif [ -n "$PCLOUD_TOKEN" ]; then url=$(lien_pcloud "$a")
  else url="$BASE/$a"; fi
  echo "[amorce] $a -> $d"
  # --no-same-owner : les archives portent des proprietaires Windows que tar tente de
  # restituer, echoue, et sort en code d'erreur alors que le contenu est intact.
  # ORDRE DES ARGUMENTS : avec -xzf, tar prend l'argument SUIVANT comme nom d'archive.
  # Le tiret de l'entree standard doit donc coller au drapeau, et --no-same-owner venir
  # avant, sinon tar cherche un fichier nomme "--no-same-owner".
  "${CURL[@]}" "$url" | tar --no-same-owner -C "$d" "$f" -
}

tirer meandre-code.tar.gz     "$RACINE"                        -xzf
tirer quebec-caches.tar.gz    "$RACINE/meandre-data/quebec"     -xzf
tirer quebec-forcages.tar     "$RACINE/meandre-data/quebec"     -xf
tirer plateformes.tar.gz      "$RACINE/plateformes-hydrotel"    -xzf

# ── PRELEVEMENTS : REINGERES, jamais herites ────────────────────────────────
# Remarque d'Essi (2026-08-27 : « j'avais specifie qu'il fallait reinjecter ces
# donnees. Je devrais le faire manuellement ? »). Non, et surtout pas : les archives
# sont un INSTANTANE FIGE, donc tout pod neuf repartait avec l'etat des caches au jour
# de leur creation, et il fallait se souvenir de rejouer l'ingestion. C'est exactement
# le mecanisme qui a laisse vivre cinq mois une copie perimee du fichier io-eau (celle
# du 15 avril contre le derive du 10 juin), avec un emissaire municipal pose sur un
# ruisseau de 13 km2 et une couverture sous-estimee d'un facteur deux et demi.
# La donnee SOURCE redevient la reference : si le parquet est dans le dossier, on
# reingere ; sinon on le DIT, plutot que de laisser un decalage silencieux.
if [ -n "${PRELEV:-io-eau-meandre.parquet}" ]; then
  _pq="$RACINE/io-eau-meandre.parquet"
  if tirer "${PRELEV:-io-eau-meandre.parquet}" "$RACINE" -xf 2>/dev/null      || "${CURL[@]}" -o "$_pq" "$BASE/${PRELEV:-io-eau-meandre.parquet}" 2>/dev/null; then
    ( cd "$RACINE" && MEANDRE_DATA="$RACINE/meandre-data" IO_EAU="$_pq"         python .runs/quebec/ingest_withdrawals.py toutes 2>&1 | tail -3 )
  else
    echo "[amorce] ATTENTION prelevements NON reingeres : deposer"          "io-eau-meandre.parquet dans le dossier source. Les caches gardent l'etat"          "du jour de l'archive, ce qui peut etre perime SANS QUE RIEN NE LE DISE."
  fi
fi

cat > "$RACINE/env.sh" <<ENV
export MEANDRE_DATA=$RACINE/meandre-data
export MEANDRE_PLATEFORMES=$RACINE/plateformes-hydrotel
export PYTHONUNBUFFERED=1
ENV
echo "[amorce] environnement : source $RACINE/env.sh"

python -m pip install -q uv 2>/dev/null || true
cd "$RACINE" && uv sync --quiet 2>/dev/null || python -m pip install -q -e . || true

source "$RACINE/env.sh"
python - <<'PY'
import torch
print(f"[amorce] torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun GPU'}")
PY
echo "[amorce] PRET. Exemple :"
echo "  cd $RACINE && PROV_EPOCHS=4 PROV_CHUNK=365 python -u .runs/quebec/province.py"
