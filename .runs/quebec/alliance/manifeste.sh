#!/usr/bin/env bash
# ETAPE 2, sur le poste Windows : dresser la liste EXACTE de ce qui doit monter, et sa
# taille. Rien n'est transfere ici ; le fichier produit sert de liste a Globus et de
# controle a l'arrivee (nombre de fichiers et octets attendus).
#
# Principe : ne monter QUE ce dont la flotte a besoin. Les 47 Go de tuiles CaSR brutes ne
# servent qu'a CONSTRUIRE les forcages, deja construits ici : ils restent sur le poste.
set -u
D=${MEANDRE_DATA:-D:/meandre-data}
P=${MEANDRE_PLATFORMS:-C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel}
M=${1:-/d/meandre-data/quebec/alliance-manifeste.txt}
: > "$M"
ajoute(){ for f in $1; do [ -e "$f" ] && echo "$f" >> "$M"; done; }
# Forcages retenus par region (CaSR corrige, plus la variante hybride de la lignee 1.0).
ajoute "$D/quebec/forcing-*-budyko.nc"
ajoute "$D/quebec/forcing-casr-corr.nc"
ajoute "$D/quebec/forcing-vaud.nc"
ajoute "$D/quebec/forcing-*-hyb.nc"
# Bases regionales (reseau, stations, attributs, prelevements).
ajoute "$D/quebec/*.duckdb"
# Champs territoriaux et tables auxiliaires lues au chargement.
ajoute "$D/quebec/*.parquet"
# Module d'evapotranspiration appris, lu a chaque demarrage.
ajoute "$D/quebec/checkpoints-etbench/*"
# Observations auxiliaires (MODIS, GRACE, CanSWE) si elles sont en cache.
ajoute "$D/quebec/aux/*"
# Plateformes Hydrotel : SEULS les fichiers TEXTE de calage. Les rasters (altitude,
# pente, occupation en .tif) et les fichiers de forme pesent 99 % de l'arbre et ne sont
# jamais lus par la recette : 4.4 Go contre 38 Mo une fois ecartes (mesure 2026-09-01).
find "$P/LN24HA" -type f \n  | grep -vE "simulation/simulation/resultat|/meteo/" \n  | grep -viE "\.(tif|shp|dbf|shx|prj|sbn|sbx|qpj|cpg|db|png|jpg)$|Thumbs" >> "$M"
n=$(wc -l < "$M")
o=$(du -ch $(cat "$M") 2>/dev/null | tail -1 | cut -f1)
echo "manifeste : $n entrees, $o"
echo "  -> $M"
