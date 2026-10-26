# Prompt : ingérer les barrages dans meandre

Ce fichier est un prompt de démarrage pour une session de travail sur l'ingestion des barrages dans meandre. Il porte l'état des connaissances établi le 2026-08-28, la méthodologie proposée, et les portes de décision qui restent ouvertes. À lire avant d'écrire une ligne de code, et à réviser à chaque verdict.

## Contexte à charger

Lis d'abord `README.md` du même dossier, puis `RAPPORT-VERIFICATION.md`. Les tables sont dans `D:/meandre-data/barrages/data/`. Les scripts `06` à `10` produisent les cinq diagnostics sur lesquels tout ce qui suit repose ; ils se relancent en quelques minutes et n'appellent pas le réseau.

Les faits établis, avec le script qui les produit :

1. Le répertoire est complet et exact. 6 115 barrages, coordonnées, hauteur, capacité de retenue, usage, réseau amont-aval. La catégorie administrative se déduit de la hauteur et de la capacité par la Loi sur la sécurité des barrages sans un seul écart sur 5 964 fiches testables, et le débit spécifique implicite des rattachements assurés vaut 21,6 L/s/km². Deux sources indépendantes se recoupent. (`05`)
2. Le stockage est massivement concentré. 20 tronçons portent 87 % des 364 000 hm³ du Québec méridional, 50 en portent 97 %. Le chantier est à cinquante objets, pas à six mille. Attention à l'agrégation : une retenue est fermée par plusieurs ouvrages qui déclarent chacun la capacité de la retenue entière, donc on prend le maximum par tronçon et jamais la somme. (`06`)
3. Sur 2001-2024, ces grandes retenues ne sont pas calables. Avec les 178 stations CEHQ, 17 des 120 tronçons candidats ont un jaugeage en aval, soit 3 % du stockage. Baskatong en est un cas d'école : la station 040830, 6 768 km², se déverse directement dans le réservoir. Son entrée est mesurée, sa sortie ne l'est pas. (`07`)
4. HYDAT n'y change presque rien sur cette période. 208 stations québécoises couvrent 2001-2024, 134 se rattachent au réseau, et le compte passe de 17 à 30 tronçons pour 5 % du stockage. (`09`)
5. La raison est datée. Toutes les stations en aval des grands réservoirs existent et s'arrêtent en 1993 ou 1994 : Gouin aval, Trenche, La Tuque, Grande-Mère, La Gabelle, Paugan, Rapides Farmers, Manic-2, Manic-5, Outardes-3 et 4, Dozois, Rapide-Sept, Rapides-2, Rapides-des-Îles, Rapides-des-Quinze. Ce sont les stations exploitées par Hydro-Québec, retirées de HYDAT au milieu des années 1990. Sur 1980-1994, 53 des 120 tronçons candidats ont un jaugeage en aval, soit 24 % du stockage, huit fois plus qu'aujourd'hui. Et CaSR démarre en 1980. (`10`)
6. Vingt-sept des 178 stations actuellement utilisées portent plus de quinze jours de débit moyen en stockage amont, treize en portent plus de trente, en ne comptant que les retenues mises en eau avant le début de l'enregistrement. Ces stations sont ajustées aujourd'hui sans aucun organe pour porter la régularisation. (`08`)

Un piège déjà tombé, à ne pas refaire : la station 073801 sur la Romaine semblait porter 460 jours de stockage amont. Elle s'arrête en juin 2014, à la mise en eau de Romaine-2. Son enregistrement est naturel de bout en bout. Toute statistique de régularisation doit filtrer sur l'année de construction contre la période observée.

## Le problème à résoudre, formulé correctement

Ce n'est pas « simuler les barrages ». C'est empêcher le modèle de mentir sur la physique du bassin là où il subit une régularisation qu'il ne peut pas représenter.

Sur les 27 stations concernées, le signal de lâcher est déjà dans l'hydrogramme observé contre lequel meandre s'ajuste. Faute d'organe de stockage, le NeRF l'absorbe dans les paramètres de sol et de routage, c'est-à-dire exactement les paramètres dont le projet revendique l'identifiabilité. Le gain attendu n'est donc pas un KGE plus haut : c'est un champ de paramètres qui cesse d'être contaminé. Si le chantier améliore le KGE, tant mieux ; s'il ne fait que déplacer la variance du sol vers un réservoir explicite, il a déjà atteint son but, et il faut le dire ainsi.

Corollaire opérationnel : les retenues sans station en aval ne coûtent rien et ne rapportent rien. Elles n'entrent pas dans la fonction de coût. Ne pas les modéliser. Manicouagan, Gouin, les Outardes et Baskatong sont hors de portée sur la fenêtre d'entraînement actuelle, et leur donner un module de lâcher serait ajouter des paramètres libres qu'aucune observation ne contraint.

## Méthodologie proposée

### Étape 0, gratuite, à faire avant tout le reste

Quantifier le coût de l'absence. Prendre le champion actuel, et comparer les métriques et les paramètres appris entre les 27 stations régularisées et les 151 autres, après contrôle de la superficie et de la région. Trois signatures à regarder : le biais de gamma (une retenue écrête les pics et soutient l'étiage, donc la variabilité simulée devrait être trop forte), le K_sat de la couche 1 et le K de Muskingum sur les tronçons amont de ces stations, comparés à leurs voisins non régularisés.

Porte de décision. Si aucune signature ne ressort, le chantier n'a pas de justification empirique et il faut s'arrêter là, quel que soit l'intérêt conceptuel. Si une signature ressort, elle devient la métrique de succès du chantier, plus honnête qu'un KGE global.

### Étape 1, le module minimum

Un nœud de stockage sur un tronçon, pas un DZTR complet. Trois paramètres, appris :

- une capacité utile, initialisée à la capacité de retenue du registre et bornée par elle ;
- un temps de vidange, qui donne un lâchage linéaire en fonction du stockage au-dessus d'un seuil ;
- une modulation saisonnière du niveau cible, en sinusoïde annuelle, deux paramètres si l'amplitude et la phase sont libres.

La règle d'Occam s'applique avec force ici. Le DZTR de Yassin et al. 2019 demande une douzaine de paramètres par réservoir, dérivés de séries observées de stockage et de lâchers. Sur 27 stations contraintes uniquement par leur hydrogramme aval, poser douze paramètres par retenue fabrique de l'équifinalité, pas de la physique. Commencer à trois, mesurer, n'ajouter qu'en montrant que l'ajout se paie.

Attention à la différentiabilité : toute commutation de zone doit être lissée, un `min` ou un `max` dur coupe le gradient. Et le stockage doit être borné par une saturation douce, pas par un `clamp`.

### Étape 2, la mutualisation

Ne pas caler cinquante jeux de paramètres indépendants. Faire prédire les trois paramètres par le champ spatial, à partir des attributs du registre : capacité, hauteur, usage déclaré, superficie du bassin versant, année de construction. C'est l'architecture du correcteur d'attributs déjà validée dans le dépôt, et c'est ce qui permet ensuite de placer un réservoir plausible là où aucune station ne contraint, sans avoir ajouté de degré de liberté.

Le registre donne l'usage de chaque ouvrage, et c'est un prédicteur fort : sur les cinquante premiers tronçons, treize sont purement hydroélectriques et le reste mêle régularisation et contrôle des crues. Un réservoir hydroélectrique et un réservoir de villégiature n'ont pas la même signature saisonnière.

### Étape 3, la validation par la fenêtre historique

C'est ce qui distingue une hypothèse d'un résultat. Sur 1980-1994, la sortie de 53 tronçons candidats est observée à la centrale, en journalier, et CaSR couvre la période. Le montage : construire un forçage 1980-1994 pour une ou deux régions test, faire tourner le modèle avec le module réservoir, et comparer le lâcher simulé au lâcher observé aux stations de centrale.

C'est la seule occasion de vérifier l'état interne du réservoir plutôt que son effet net en aval. Sans elle, rien ne distingue une politique de lâcher bien calée d'une erreur qui en compense une autre.

La réserve à porter dans le texte : rien ne garantit que les consignes d'exploitation de 1985 soient celles de 2020. La validation historique établit que le module est capable de reproduire une gestion réelle, pas que les paramètres appris sur 2001-2024 soient les bons.

### Étape 4, ce qui reste ouvert

Deux voies non explorées, à ne pas oublier dans la discussion d'un article :

L'observation spatiale. Le produit JRC Global Surface Water donne l'étendue en eau mensuelle de 1984 à aujourd'hui, ce qui couvre les deux fenêtres, et combiné à une hypsométrie tirée d'un MNT il donne une variation de stockage. ICESat-2 depuis 2018 et Sentinel-3 donnent le niveau directement, et Manicouagan, Gouin, Baskatong et les Outardes sont très au-dessus des seuils de taille. Le bémol québécois est le couvert de glace de décembre à avril, mais le marnage estival et automnal est justement le signal qui compte. Sondage recommandé sur trois réservoirs témoins avant tout investissement.

L'accès aux données d'exploitation. Les séries de niveaux et de lâchers existent chez Hydro-Québec depuis les années 1950 et ne sont pas publiées. Le flux ouvert d'Hydro-Québec diffuse 298 séries horaires de niveau et 72 de débit, dont Baskatong, Cabonga amont et Dozois, mais sur une fenêtre glissante d'une dizaine de jours seulement. Une moissonneuse quotidienne construirait un jeu de validation utilisable dans un an. C'est un coût quasi nul aujourd'hui pour une option ouverte plus tard.

## Ce qu'il ne faut pas faire

Ne pas rattacher automatiquement les grandes retenues. Le rattachement de `04_mapping.py` vise le tronçon le plus proche, pas l'exutoire de la retenue : Cabonga tombe sur le lac Icône. Sur cinquante objets, c'est une après-midi de rattachement manuel, et elle doit être faite avant tout entraînement.

Ne pas confondre capacité de retenue et volume utile. Le registre donne le volume total, hauteur de la retenue multipliée par la superficie du réservoir. C'est une borne supérieure du marnage exploitable, à utiliser comme contrainte dure et non comme valeur cible.

Ne pas ingérer les 6 115 barrages. La table `dams` existe déjà dans les bases régionales, héritée d'une récolte antérieure, avec des rattachements à plus de 70 km. Elle n'est pas utilisable telle quelle et ne doit pas servir de point de départ.

Ne pas laisser le module réservoir libre sur des tronçons sans station en aval. Vérifier avec `07_amont_stations.py` avant d'activer un nœud.

## Critères d'acceptation

Le chantier est un succès si, et seulement si :

1. la signature identifiée à l'étape 0 se réduit significativement sur les 27 stations régularisées, sans dégradation sur les 151 autres ;
2. les paramètres de sol de ces bassins se rapprochent de ceux de leurs voisins non régularisés, ce qui est le vrai objectif ;
3. le lâcher simulé sur la fenêtre 1980-1994 reproduit le lâcher observé aux stations de centrale avec un KGE supérieur à celui d'un modèle sans réservoir, sur au moins une région test ;
4. le nombre de paramètres ajoutés par retenue reste à trois ou quatre, et toute augmentation est justifiée par une amélioration mesurée.

Un KGE global qui monte sans que les points 1 et 2 bougent doit être traité comme un signal d'équifinalité, pas comme un succès.

## Décisions qui appartiennent à Essi

- Étendre ou non la fenêtre de forçage à 1980, ce qui suppose de reconstruire CaSR sur quinze années supplémentaires pour au moins une région test.
- Faire du chantier réservoirs un article distinct, ou un élément de l'article sur l'identifiabilité. Le résultat sur les 27 stations contaminées relève clairement du second.
- Lancer ou non la moissonneuse quotidienne du flux Hydro-Québec, décision à coût quasi nul mais qui ne se rattrape pas rétroactivement.
