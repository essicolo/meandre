#!/usr/bin/env bash
# Liste de transfert Globus du lot provincial (province.sbatch), sur le poste Windows.
#
# Chaque ligne porte la source sur le poste et la destination relative a l'espace de projet
# de la grappe, au format des lots de Globus (`globus transfer --batch`) :
#   <chemin du poste> <chemin relatif a $PROJET/meandre>
# Seul ce que le pilote lit avec la recette du 4 octobre monte : forcages CaSR brut, bases
# des territoires, tables provinciales, cibles NEISIM, fichiers texte des projets PHYSITEL.
set -u
D=${MEANDRE_DATA:-D:/meandre-data}
P=${MEANDRE_PLATFORMS:-C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel}
M=${1:-$D/quebec/alliance-manifeste-province.txt}
: > "$M"
ajoute(){
  local dest=$1; shift
  for f in "$@"; do [ -f "$f" ] && echo "$f $dest/$(basename "$f")" >> "$M"; done
}
ajoute donnees/quebec "$D"/quebec/forcing-*-casr-brut.nc
ajoute donnees/quebec "$D"/quebec/*.duckdb
ajoute donnees/quebec "$D"/quebec/*.parquet
ajoute donnees/quebec/checkpoints-etbench "$D"/quebec/checkpoints-etbench/*
ajoute checkpoints-reference "$D"/quebec/checkpoints-reference/*
ajoute donnees/derives/auxiliaires "$D"/derives/auxiliaires/neisim-*.npz "$D"/derives/auxiliaires/rsesq-puits.parquet
# Projets PHYSITEL : seulement les fichiers texte, lus pour l'occupation du sol, les milieux
# humides et les lacs. Les rasters et fichiers de forme ne sont jamais lus.
find "$P/LN24HA" -type f \
  | grep -vE "simulation/simulation/resultat|/meteo/" \
  | grep -viE "\.(tif|shp|dbf|shx|prj|sbn|sbx|qpj|cpg|db|png|jpg)$|Thumbs" \
  | while read -r f; do echo "$f plateformes/${f#"$P"/}" >> "$M"; done
n=$(wc -l < "$M")
o=$(cut -d' ' -f1 "$M" | while read -r f; do stat -c %s "$f"; done | awk '{s += $1} END {print s}')
echo "manifeste : $n fichiers, $((o / 1000000)) Mo"
echo "  -> $M"
