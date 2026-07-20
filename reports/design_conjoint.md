# Design : entraînement conjoint multi-régions (proposition, à discuter)

Objectif : UN modèle pour le domaine PHYSITEL entier. Un seul encodeur spatial (NeRF) alimenté par les coordonnées et attributs de ~28 000 nœuds, entraîné sur les 178 jauges poolées plus l'ET satellite 8 jours et GRACE de toutes les régions. C'est le test décisif de la thèse de méandre: le continuum de paramètres contre les six calages équifinaux.

## Architecture d'entraînement

1. Un HydroModel unique, encodeur spatial et colonne partagés. Les codes latents z_n restent par nœud (concaténation des 15 vecteurs régionaux, offsets d'index par région).
2. Les graphes de routage restent par région (aucun tronçon ne traverse une frontière PHYSITEL): dans chaque epoch, boucle round-robin sur les régions; pour chaque région, forward complet (colonne + routage opérateur) sur son graphe, loss régionale, accumulation du gradient; un seul optimizer.step() par groupe de régions (ou par epoch, à trancher au pilote selon la stabilité).
3. Loss poolée: somme des loss régionales pondérées par le nombre de jauges valides (sinon SLSO avec 38 jauges serait noyé par la somme des petits). Multi-obj ET/GRACE par région avec les mêmes poids partout.
4. Mémoire: une région à la fois sur GPU (le plus gros graphe, GASP 3917 nœuds, tient largement); les tenseurs de forçage des autres régions restent sur CPU et montent à tour de rôle. Epoch estimé: somme des epochs régionaux, ~45-60 min.
5. Ancrages régionaux en INITIALISATION seulement (Linacre + fonte par région là où validé, spatial_melt partout): le conjoint doit pouvoir s'en éloigner par gradient, c'est le point de la démonstration. Le boréal (leçon SAGU) part sans ancrage fonte.
6. Volume: point fixe par région (lame + ETR simulée, une itération après le premier run conjoint).

## Ce que le conjoint teste précisément

- Les régions pauvres en jauges (CND, LABI) apprennent-elles du pooling? (le NeRF voit leurs attributs, les jauges des voisines contraignent la physique partagée)
- Le trou de l'est se referme-t-il quand la physique est contrainte par 178 jauges au lieu de 16-27?
- L'équifinalité: le conjoint trouve-t-il UN jeu de champs cohérent là où Hydrotel a besoin de 6 calages incompatibles?

## Protocole (règles maison)

- Pilote 3 régions d'abord (SLSO + MONT + GASP: 3 profils, 77 jauges, ~8700 nœuds), verdict held-out 2022-2024 contre les runs mono-région ET contre l'ensemble des 6, AVANT tout passage à 15.
- Critère de succès du pilote: aucune des 3 régions ne régresse vs son meilleur mono-région, et au moins une gagne significativement (sinon le pooling n'apporte rien et on arrête les frais).
- Un changement à la fois ensuite (ajout des 12 régions = un changement).

## Chantier code (estimation honnête)

- MultiBasinTrainer: généralisation du Trainer actuel (boucle régions, états/spinup par région, val par région, checkpointing du modèle partagé). Le gros du travail.
- Chargement: BasinCache par région (déjà là), z_n concaténés (offset d'index simple), forcings par région (déjà construits).
- Diag: métriques par région à chaque val, autopilot global (drift moyen pondéré).
- Risques identifiés: coût CPU-GPU des transferts par région (mitigé par pin_memory), hétérogénéité des périodes d'obs (géré: masques NaN), déséquilibre des tailles (géré: pondération).

Rien ne sera lancé au-delà du pilote 3 régions sans validation ensemble.
