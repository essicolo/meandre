# Chantiers à venir de méandre

Ouvert le 2026-09-08. Ce document liste les chantiers identifiés mais non entrepris, avec leur raison d'être, ce qu'ils coûtent et le test qui dira s'ils tiennent. Il ne remplace ni le registre des hypothèses, qui dit ce qui est vrai aujourd'hui, ni le journal des expériences, qui raconte ce qui a été fait. Un chantier n'entre ici qu'avec un critère de réussite mesurable.

## 1. Forçage climatique MRCC6 et projections futures

**Pourquoi.** Le modèle sert aujourd'hui à reconstituer le passé et à naturaliser des débits observés. Le troisième volet de la stratégie de publication, les scénarios, demande de le faire tourner sous un climat qui n'a pas eu lieu. Le Modèle régional canadien du climat de sixième génération, développé au centre ESCER de l'UQAM avec Ouranos et fondé sur le coeur atmosphérique GEM, est la source régionale naturelle pour le Québec.

**Ce qu'il faut comprendre avant de commencer.** Un modèle climatique ne reproduit pas la météorologie d'une journée donnée, il reproduit la statistique d'un climat. Il ne peut donc servir ni à l'entraînement, ni à un calcul d'efficacité contre des débits observés. L'usage correct est en deux temps : entraîner sur une réanalyse, ici les données CaSR corrigées, puis simuler avec les sorties climatiques. Confondre les deux produirait des scores dénués de sens.

**Ce que le chantier demande.** Obtenir les six variables consommées par la colonne verticale au pas journalier sur le domaine québécois, soit la précipitation, les températures minimale et maximale, le rayonnement net, le vent et l'humidité. Les interpoler aux tronçons, ce que la chaîne de préparation du forçage sait déjà faire. Construire une correction de biais contre CaSR sur une période commune : c'est l'étape déterminante, puisque le plafond de performance mesuré sur ce modèle est fixé par le forçage et non par la structure. Vérifier enfin que le régime simulé sur la période historique du modèle climatique, après correction, reproduit les signatures hydrologiques observées.

**Critère de réussite.** Les signatures hydrologiques simulées sous le climat historique du modèle régional, après correction de biais, doivent être compatibles avec celles obtenues sous la réanalyse. Sans cela, aucune projection n'est défendable.

**Ce que la différentiabilité apporte ici.** La dérivée du débit par rapport à chaque paramètre et à chaque variable de forçage est calculable exactement. Une projection peut donc être accompagnée de l'attribution de son changement aux différentes causes, ce qu'un modèle calé de façon classique ne permet pas.

## 2. Fonction objectif non paramétrique

**Pourquoi.** Le diagnostic de septembre 2026 a établi que la composante de variabilité de l'efficacité de Kling-Gupta, seul terme de la perte qui pénalise un hydrogramme figé, est satisfaisable en abaissant la moyenne autant qu'en augmentant la variabilité, puisqu'elle compare des coefficients de variation. Wagener et ses collaborateurs, dans Hydrology and Earth System Sciences en 2026, calent quarante-sept structures sur dix bassins avec huit fonctions objectif et concluent que le choix de la fonction objectif pèse souvent plus lourd que le choix de la structure, et que la version non paramétrique de l'efficacité de Kling-Gupta donne la plus faible erreur globale sur les signatures sensibles. Cette version remplace précisément le rapport des variances par un terme fondé sur la courbe des débits classés, et la corrélation de Pearson par celle de Spearman.

**Ce que le chantier demande.** Implémenter cette version comme terme différentiable, l'apparier à un terme de basses eaux selon la recommandation des auteurs, et juger le résultat sur des signatures hydrologiques plutôt que sur une efficacité globale.

**Critère de réussite.** À efficacité médiane égale ou meilleure sur la période d'évaluation, la part de jours plats doit rejoindre celle de l'observé et le rapport des pointes annuelles doit rester dans l'intervalle de 0,8 à 1,2.

## 3. Plateaux d'hiver

**Pourquoi.** Huit régions sur quatorze échouent au verdict de forme, et leurs suites plates les plus longues sont hivernales, de cinquante-neuf à cent dix-neuf jours. La part imputable au modèle et celle imputable aux observations n'ont pas été séparées : le Centre d'expertise hydrique reconstruit la majorité des débits de janvier et février sous couvert de glace, et l'observé est lui-même plus plat en hiver.

**Ce que le chantier demande.** Reprendre le verdict de forme en distinguant les jours réellement mesurés des jours reconstruits, à partir des remarques accompagnant les débits publiés, puis rejuger la platitude hivernale sur les seuls jours mesurés.

**Critère de réussite.** Savoir si la règle actuelle est trop sévère en hiver, et si oui la corriger, ou constater que le modèle fige réellement son hiver et ouvrir alors un diagnostic de physique.

## 4. Cause du gradient non fini

**Pourquoi.** Des blocs sont écartés à chaque entraînement parce que leur gradient devient non fini. Le filet de sécurité les jette sans perdre l'époque, mais la cause reste inconnue, et les gels de validation observés dans le bras à un pas par bloc suivent chaque fois un épisode de blocs jetés.

**Ce que le chantier demande.** L'instrumentation existe et imprime, avant restauration, les paramètres touchés et la norme du gradient de chaque terme de la perte. Elle n'a pas encore attrapé de bloc fautif sur le sous-bassin d'essai. Il faut la faire tourner sur une région entière, où le phénomène se produit.

**Critère de réussite.** Nommer l'opération et le terme responsables, et proposer un correctif qui supprime le rejet plutôt que de le compenser.
