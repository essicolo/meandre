# Design : module ET appris (physico-empirique, supervisé MOD16)

Date : 2026-07-21. Statut : spec à valider avec Essi avant tout code.

## Motivation

Le choix entre formules d'ETP (McGuinness, Linacre, Penman) et l'emprunt de coefficients calés aux plateformes Hydrotel (ancrages) sont de la sélection de modèle à l'aveugle alors qu'on possède la variable cible : 1150 composites MOD16A2GF 8-jours par région, ~30M d'observations nœud×composite valides sur les 15 régions (fondations assainies 2026-07-19). La thèse du projet est l'identifiabilité par les données; l'ET est la pièce où la donnée est la plus riche. On remplace le choix de formule par une relation apprise météo + territoire vers ETR, régionale par construction (aucun coefficient par région), ce qui répond aussi au compromis inter-régions qui a cassé le timing du conjoint (pilotes 3/3b/3c : critère échoué, gap GASP = r/timing).

## Étape 1 : banc HORS-LIGNE (régression supervisée pure, pas de simulation hydrologique)

Cible : ET 8-jours MOD16 (mm/8j) par nœud terrestre, les 15 régions.

Entrées du module : séquence météo journalière (P, Tmin, Tmax, Rn, u2, e_a) couvrant la fenêtre 8 jours plus un historique de 60-90 jours (mémoire : état hydrique et phénologique implicites); attributs territoriaux statiques du nœud (les mêmes que le NeRF); doy en sin/cos; latitude seule comme entrée physique (rayonnement extraterrestre, comme McGuinness), PAS de longitude ni de coordonnées identifiantes : le module doit transférer à une région jamais vue.

Architectures comparées :
- (a) GRU petit (hidden 32-64) sur la séquence météo, attributs statiques concaténés au readout.
- (b) MLP sans mémoire sur des agrégats de fenêtre (ablation : valeur de la mémoire).
- (c) Bancs physiques sur les mêmes fenêtres : McGuinness×K_c et Linacre×coeff, avec K_c/coeff ajustés par moindres carrés sur le train (les formules dans leur meilleur jour, pas des hommes de paille).

Splits pré-enregistrés :
- Spatial : leave-region-out, entraîner sur 12 régions, tester sur 3 régions tenues couvrant les classes (gasp = est naturel, sagu = boréal/régulé, mont = sud).
- Temporel : train 2000-2018, val 2019-2021, test 2022-2024, toutes régions.

Métriques par région : R² 8-jours, RMSE, biais annuel (mm/an).

Critère de succès pré-enregistré : le module bat les deux formules calées À LA FOIS en leave-region-out et sur 2022-2024. Sinon on garde les formules et on aura mesuré pourquoi.

Coût : quelques heures GPU max, itérations en minutes. Aucun pilote hydrologique tant que ce banc n'a pas tranché.

## Étape 2 (seulement si succès, validation Essi) : intégration dans la colonne

Le module prédit la demande évaporative (ETP effective ou conductance); la colonne garde l'extraction par couche et les scaling factors sf1-3 (conservation de masse intacte). K_c phénologique neutralisé quand le module est actif (double comptage). Fine-tuning bout-en-bout avec w_et conservé comme contrainte.

## Étape 3 (plus tard) : même patron ailleurs

Fonte (supervision par couvert nival MODIS et relevés nivométriques plutôt que seuils empruntés), stockage (GRACE comme signal d'apprentissage du module, pas seulement pénalité de fin de chaîne).

## Réserves honnêtes

MOD16 est un produit de modèle (PM piloté MERRA + LAI MODIS) : on hérite de ses biais; les tours de flux le valident en boréal (~450 mm/an, cf. reference_casr_wetbias). Mitigation : vérification croisée du biais annuel contre le bilan régional P - Q - dS(GRACE). Fuite potentielle via la météo MERRA de MOD16 : le leave-region-out reste valide car c'est la relation qui doit transférer, pas un niveau local.

## Leçons de méthode qui encadrent ce chantier (2026-07-21)

Chaque hypothèse passe par : 1) diagnostic gratuit sur l'existant, 2) falsification à petite échelle (minutes-heures), 3) pilote complet raccourci seulement si 1 et 2 concordent. Les pilotes conjoints ont coûté 2×20h pour des hypothèses falsifiables à bas prix (la décomposition r/beta/gamma qui a innocenté les z_n était disponible en 5 minutes sur le checkpoint pilote3b). Les ancrages sont gelés : smoke labi+cndc avec ancrages = LABI (boréal) 0.126 à l'epoch 0 contre 0.611 sans, cohérent avec la leçon SAGU (ancrages sud toxiques au boréal), et surtout contraire à la thèse (emprunter le calage d'Hydrotel n'est pas l'identifiabilité par les données).
