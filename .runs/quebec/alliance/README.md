# Porter la flotte méandre sur une grappe de l'Alliance

Procédure en sept étapes. Les trois premières se font la veille, sans transférer un octet ; les quatre autres se déclenchent quand la connexion est bonne. Chaque étape produit une sortie vérifiable, et aucune n'engage la suivante.

## Ce que la grappe change

Les quinze régions sont indépendantes : sur le poste elles se suivent, sur la grappe elles partent ensemble. Une nuit de vingt heures devient l'attente d'une seule région, environ deux heures, plus le temps de file. Il n'y a pas de facturation à l'heure : ce qui se répartit entre les groupes est une priorité d'ordonnancement calculée sur la consommation passée, et un accès inutilisé ne consomme rien.

## Étape 1, la veille : sonder les trois grappes et en choisir une

Les accès à Fir, Narval et Rorqual sont accordés, et la procédure est identique sur les trois : même ordonnanceur, même pile logicielle. Une seule différence compte : les données ne se partagent pas entre grappes, donc le transfert se fait vers une seule d'entre elles.

Sur le nœud de connexion de chacune, `bash .runs/quebec/alliance/sonde.sh`, une minute et aucune ressource de calcul consommée. Trois critères tranchent, dans cet ordre. Le compte disponible d'abord : une allocation peut n'exister que sur certaines grappes, et sans compte utilisable rien ne se soumet. L'attente en file ensuite, qui varie fortement de l'une à l'autre. Les cartes offertes enfin, sachant que le modèle tient dans huit gigaoctets de mémoire vidéo : les cartes les plus modestes conviennent et se réservent plus vite que les plus grosses.

La sortie donne aussi le nom du compte à passer à `--account`, les quotas d'espace et la présence de PyTorch dans la logithèque locale. La sortie donne le nom du compte à passer à `--account`, les cartes offertes, les quotas d'espace et la présence de PyTorch dans la logithèque locale.
## Étape 2, la veille : dresser le manifeste

Sur le poste, `bash .runs/quebec/alliance/manifeste.sh`. Il liste ce qui doit monter et sa taille, mesurée le 2026-09-01 à 20 Go pour 86 entrées. Les 47 Go de tuiles CaSR brutes ne montent pas : elles ne servent qu'à construire les forçages, déjà construits. Le fichier produit sert de liste à Globus, puis de contrôle à l'arrivée.

## Étape 3, la veille : déposer le dépôt

Le dépôt est petit et n'a pas besoin de Globus : `git clone` depuis le nœud de connexion, ou une archive par `scp`. La chaîne d'entraînement est entièrement portable depuis le 2026-09-01 : elle lit ses racines dans `MEANDRE_DATA`, `MEANDRE_PLATFORMS` et `MEANDRE_RQH`, et ne porte plus aucun chemin absolu.

## Étape 4, au bureau : transférer les données

Par Globus, entre le point de terminaison personnel installé sur le poste et celui de la grappe. Globus reprend tout seul après une coupure, donc le transfert n'a pas à être surveillé. Destination : l'espace de projet du groupe, pas l'espace personnel qui est petit, et pas l'espace de travail temporaire qui est purgé après soixante jours. À l'arrivée, comparer le nombre de fichiers et le volume au manifeste.

## Étape 5, au bureau : bâtir l'environnement

`bash .runs/quebec/alliance/env_grappe.sh` sur le nœud de connexion, une seule fois. Les nœuds de calcul n'ont aucun accès à Internet : tout s'installe depuis la logithèque locale de l'Alliance, et PyTorch en particulier doit venir de là, compilé pour les cartes de la grappe.

## Étape 6 : le contrôle de reproduction, avant toute flotte

Une seule région, zéro époque, sur un modèle dont la valeur est connue :

```
sbatch --account=<compte> --array=0-0 --time=01:00:00 \
  --export=ALL,ETL_EPOCHS=0,ETL_WARM_FROM=.runs/quebec/checkpoints/best-outv-etl-canon.pt,JOINT_FX_SUFFIX=-hyb \
  .runs/quebec/alliance/region.sbatch
```

Le KGE médian attendu est 0,780 sur 16 stations pour la période 2022-2024. Tant que ce nombre n'est pas retrouvé, rien d'autre ne se lance : c'est ce contrôle, à six minutes, qui a évité de publier un rapport faux le 2026-09-01.

## Étape 7 : la flotte

```
sbatch --account=<compte> --array=0-14 .runs/quebec/alliance/region.sbatch
```

Les quinze régions partent ensemble. `squeue -u $USER` suit l'avancement, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS` donne le bilan. Les points de reprise se retrouvent dans `.runs/quebec/checkpoints/` sur la grappe et redescendent par Globus.

## Pièges connus

Les nœuds de calcul sont sans réseau : tout téléchargement échoue silencieusement en fin de tâche. L'espace de travail temporaire est purgé après soixante jours, donc les données de référence vivent dans l'espace de projet. Le disque local du nœud, désigné par `SLURM_TMPDIR`, est vidé à la fin de chaque tâche mais il est bien plus rapide que le système de fichiers partagé : la tâche y copie ce dont elle a besoin au démarrage, ce que fait déjà `region.sbatch`. Enfin, une tâche courte passe avant une tâche longue dans la file : demander six heures plutôt que vingt-quatre raccourcit l'attente.
