#!/usr/bin/env bash
# ETAPE 1, sur le noeud de connexion de Fir : decouvrir ce que le compte permet, AVANT
# de transferer quoi que ce soit. N'ecrit rien, ne consomme aucune ressource de calcul.
# Sortie a rapporter : nom du compte, cartes disponibles, quotas.
echo "=== comptes et priorite ==="
sshare -U 2>/dev/null | head -20
echo
echo "=== comptes utilisables pour --account ==="
sacctmgr -nP show associations user="$USER" format=Account,Partition 2>/dev/null | sort -u
echo
echo "=== cartes graphiques offertes ==="
sinfo -o "%P %G %D %m %c" 2>/dev/null | sort -u | head -25
echo
echo "=== quotas d'espace ==="
diskusage_report 2>/dev/null || quota -s 2>/dev/null
echo
echo "=== modules disponibles ==="
module -t spider python 2>&1 | grep -E "^python/3\.(11|12|13)" | sort -u | head
module -t spider cuda 2>&1 | grep -E "^cuda/" | sort -u | tail -3
echo
echo "=== roues Python locales (aucun acces Internet sur les noeuds de calcul) ==="
module load python/3.11 2>/dev/null
pip download torch --no-deps --no-index -d /tmp/essai-roue 2>&1 | tail -2
