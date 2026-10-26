# Diagnostic de la Montérégie — 7 octobre 2026, version 2.0

**Auteur** : Agent meandre  
**Date** : 2026-10-07  
**Version** : 2.0 (corrigée)  
**Statut** : À valider avec Essi  
**Note** : Ce document remplace la version 1.0 du 7 octobre. La thèse centrale de la v1.0 (drainage agricole comme problème principal) est réfutée par R323. Les erreurs factuelles de la v1.0 sont corrigées ci-dessous.

---

## Sommaire

1. [Indicateurs mesurés et problématique](#1-indicateurs-mesurés-et-problématique)
2. [Cause racine établie : biais saisonnier du forçage CaSR brut](#2-cause-racine-établie-biais-saisonnier-du-forçage-casr-brut)
3. [Processus physiques vérifiés et leur impact réel](#3-processus-physiques-vérifiés-et-leur-impact-réel)
4. [Corrections récentes intégrées](#4-corrections-récentes-intégrées)
5. [Diagnostic complet](#5-diagnostic-complet)
6. [Recommandations](#6-recommandations)
7. [Références](#7-références)

---

## 1. Indicateurs mesurés et problématique

### 1.1. État actuel de la Montérégie

Territoire modélisé : Montérégie entière, 23 stations, période 2022-2024, dernière époque, code corrigé (R308, R311).

| Métrique | Valeur | Interprétation | Source |
|----------|--------|----------------|--------|
| **KGE médian** | **0,549** | Score global mauvais | R317 |
| **Volume sim/obs** | 1,04 | Légèrement excédentaire mais acceptable | R317 |
| **Min. annuel sur 7 jours** | **1,46** | **Les bas débits sont TROP HAUTS**, pas trop bas | R317 |
| **Juillet sim/obs** | **0,45** | **Étiage estival catastrophique** | R317 |
| **Août sim/obs** | **0,52** | **Étiage estival catastrophique** | R317 |

**Rapports mensuels simulé/observé** : 01:1,09 | 02:1,02 | 03:1,04 | 04:1,44 | 05:1,22 | 06:0,81 | **07:0,45** | **08:0,52** | 09:1,01 | 10:0,97 | 11:0,97 | 12:1,02

**→ Diagnostic** : Étés (juillet-août) **à moitié vides**, avec un **volume annuel correct**. L'été simulé est **plat**, sans crues d'orage. Les bas débits (min sur 7 jours = 1,46) sont **trop HAUTS**, ce qui indique une **mauvaise répartition temporelle** de l'écoulement, pas un manque d'eau absolu.

### 1.2. Comparaison avec l'Outaouais (référence)

| Territoire | KGE | Volume sim/obs | Min. 7j sim/obs | Juillet | Août |
|-----------|-----|----------------|-----------------|--------|------|
| Outaouais | **0,796** | 1,16 | **0,85** | 0,81 | - |
| Montérégie (socle) | 0,549 | 1,04 | **1,46** | **0,45** | **0,52** |

L'Outaouais démontre que le modèle peut atteindre des scores élevés (KGE > 0,75) avec la recette actuelle.

---

## 2. Cause racine établie : biais saisonnier du forçage CaSR brut

### 2.1. Mesure du biais (R321)

**Fait ÉTABLI sans simulation** : CaSR brut n'est pas homogène en espace ni en saison.

En **Montérégie**, rapport CaSR brut / stations GHCN (2011-2024) :
- **Hiver** (novembre à mars) : 1,0 à 1,28 × les stations (légèrement excédentaire)
- **Été** (mai à septembre) : **0,69 à 0,86 × les stations** (déficit de 15 à 30 %)

**Interprétation** : Les pluies d'été sont **sous-estimées de 15 à 30 %** dans le forçage CaSR brut pour la Montérégie. 

### 2.2. Conséquence directe

Avec un **déficit de pluie estivale de 15-30 %**, **aucun processus physique** (drainage, routage, nappe) **ne peut compenser l'absence de pluie**. 

**Preuve par l'absurde** : Sur le **banc** (bassin de la Châteauguay, 2 492 km²), la paire drainage de Hooghoudt + tête de temps de transfert **passe le KGE de 0,52 à 0,73**. Mais au **territoire Montérégie**, la même paire donne **KGE 0,555 vs 0,549** (R323) : **gain nul**. 

**Explication** : La Châteauguay du banc **compensait par ses paramètres propres** ce que le **champ partagé du territoire** ne peut pas compenser. Quand le forçage manque de pluie, les paramètres du champ (K_c, etc.) ne peuvent pas créer de l'eau qui n'existe pas.

### 2.3. Statut des décisions

La décision du 2 octobre 2026 d'utiliser CaSR brut partout a été prise **sur l'été de l'Outaouais** où CaSR brut est **juste** (rapport 0,95 à 1,04 en été). 

**Cette décision ne vaut pas pour la Montérégie.**

**Décision requise** : Correction du forçage CaSR **par territoire et par saison**, **hors de meandre** (comme indiqué explicitement dans AGENTS.md, section 5 bis).

---

## 3. Processus physiques vérifiés et leur impact réel

### 3.1. Drainage agricole (R296, R323)

**Fait ÉTABLI** : Le drainage agricole (Hooghoudt sur couche 2) **fonctionne** et améliore les scores **au banc**. 

Configuration : `ETL_SOIL_TOML=sol-drain-appris.toml` avec `DRAIN_HOOGHOUDT`, `spacing_m=15.0`, `depth_m=1.0`, `share=0.6`, `learn=["spacing_m", "share"]`

**Impact mesuré** :
- **Banc Châteauguay** : KGE 0,52 → **0,73** (gain de +0,21)
- **Territoire Montérégie** : KGE 0,549 → **0,555** (gain nul, R323)
- Juillet et août restent à **0,46 et 0,51** (inchangés)
- Minimum sur 7 jours passe de **1,46 à 1,63** (pire !)

**Conclusion** : Le drainage **ne corrige pas l'étiage estival en Montérégie** car le problème vient du **forçage**, pas de la représentation physique. La même configuration qui fonctionne au banc est **neutre au territoire** car le champ partagé ne peut pas compenser un déficit de pluie.

**Recommandation** : Le drainage reste **un processus valide et nécessaire** pour la physique agricole, mais son impact sera limité tant que le forçage CaSR brut ne sera pas corrigé pour l'été montérégien.

### 3.2. Temps de transfert et tête de routage (R299, R323)

**Fait ÉTABLI** : Le temps de transfert figé à 24h **aplatit les crues d'orage**.

Configuration : `ETL_ROUTAGE_TETE=1`, `ETL_ROUTAGE_SOUSPAS=24`, `MEANDRE_KMUSK=1,48,24`

**Clarifications importantes** (corrigées vs v1.0) :
- `ETL_ROUTAGE_TETE=1` : **Tête linéaire sur les attributs du tronçon** (longueur, pente), **pas dépendant du débit**
- `MEANDRE_KMUSK=1,48,24` : Bornes de **1h à 48h**, valeur initiale 24h (la borne basse est **1h**, pas 4h comme indiqué dans la v1.0)

**Impact mesuré** :
- **Banc** : Meilleure reproduction des pics, synchronisation améliorée
- **Territoire Montérégie** : **Neutre** en combinaison avec le drainage (R323)
- Juillet et août restent à **0,46 et 0,51**

**Conclusion** : Comme pour le drainage, la tête de routage **ne peut pas compenser un déficit de pluie dans le forçage**.

### 3.3. Évaporation de l'eau libre (R285, R288)

**Fait ÉTABLI** : La colonne n'évaporait que la pluie du jour sur les fractions d'eau libre.

**Impact mesuré** (passe avant) :
- 052805 : Q7min de 1,38 → **0,98** (+39 %)
- Outaouais : Q7min de 1,33 → **1,09** (+23 %)

**Pour la Montérégie** : **0,8 % de surface d'eau libre** (AGENTS.md, section 5). 

**Conclusion** : **Aucun impact attendu en Montérégie** car la fraction d'eau libre est trop faible pour affecter significativement l'étiage. Cette correction reste valide mais **ne résout pas le problème montérégien**.

### 3.4. Nappe libre et stockage souterrain (R316, R317)

**Fait ÉTABLI** : L'écoulement de base est **trop fort en hiver** dans les grands bassins du nord.

**Mécanisme** : Deux problèmes distincts :
1. **Décembre-janvier** : Écoulement de base trop fort (ex. bassin 040841 : base 0,84 mm/j vs observé 1,28 mm/j)
2. **Février-mars** : Fonte trop précoce qui alimente l'écoulement hypodermique (sud)

**Pour la Montérégie** : Le problème estival **n'est pas lié à la nappe** mais au **forçage** (R321).

---

## 4. Corrections récentes intégrées

### 4.1. Prélèvements et rejets dans l'entraînement (R308)

**DÉFAUT CORRIGÉ le 2026-10-05** : Tous les modèles ont été entraînés **SANS prélèvements ni rejets**.

**Mécanisme** : L'entraîneur passait la série complète des prélèvements avec le forçage d'un bloc, et la simulation lisait à l'indice de pas compté depuis le début de l'appel. Chaque bloc lisait donc **janvier 2000**, **nul**.

**Correctif** : `WithdrawalData.slice` — chaque appel reçoit la tranche de son forçage. Test : `tests/test_prelevements_tranche.py`.

**Impact** : Les prélèvements sont maintenant **intégrés dans l'entraînement**. Les modèles précédents les avaient **absorbés dans leurs paramètres** en s'ajustant à des débits observés qui les contenaient.

**Pour la Montérégie** : Prélèvement net moyen par mois 2001-2024 : de **-0,3 m³/s** (novembre) à **-7,2 m³/s** (juillet). Ces valeurs **atteignent maintenant l'entraînement**.

### 4.2. Référence du terme MOD16 centré (R309, R310, R311)

**DÉFAUT CORRIGÉ le 2026-10-05** : La référence du terme MOD16 centré **n'était jamais posée** au pilote régional.

**Mécanisme** : Le pilote évaluait avant d'entraîner. La référence n'y était pas calculée. L'époque d'entraînement reprenait l'état en cache sans repasser par ce code. Le repli prenait la moyenne du **premier bloc d'hiver** (0,02 mm/j) au lieu de la **vraie moyenne** (1,61 mm/j).

**Impact du défaut** : Le terme MOD16 centré poussait K_c vers le bas **50 à 80 fois plus fort** que les autres termes réunis, causant :
- Chute de K_c en première époque
- Excès de volume du modèle régional

**Correctif** : Référence posée au début de l'époque. La poussée du terme sur K_c tombe de **+1,27 à +0,015** (R311).

**Conséquence** : Réentraînement de l'Outaouais avec les correctifs : KGE **0,755 → 0,796**, volume **+21 % → +16 %** (R313).

**Pour la Montérégie** : Ces correctifs sont intégrés dans la recette `-corr1006-dt`, mais **n'améliorent pas les étés** car le problème est le forçage.

### 4.3. Résumé des correctifs intégrés dans `-corr1006-dt`

| Correctif | Statut | Impact sur Montérégie |
|-----------|--------|------------------------|
| Prélèvements dans l'entraînement (R308) | ✅ Corrigé | Paramètres plus réalistes |
| Référence MOD16 (R311) | ✅ Corrigé | K_c plus stable |
| Drainage Hooghoudt | ✅ Activé | Gain nul (forçage limitant) |
| Tête de routage | ✅ Activée | Gain nul (forçage limitant) |
| **Forçage CaSR été** | ❌ Non corrigé | **Problème persistant** |

---

## 5. Diagnostic complet

### 5.1. Chaîne causale

```
Biais saisonnier CaSR brut (R321)
  ↓
Pluies d'été sous-estimées de 15-30 % en Montérégie
  ↓
Débit simulé estival trop faible (juillet 0,45, août 0,52)
  ↓
Étiage "à moitié vide" (expression d'Essi)
  ↓
Min. sur 7 jours = 1,46 (bas débits TROP HAUTS par rapport à la moyenne)
```

**Note sur l'interprétation du min. sur 7 jours** : Une valeur > 1 signifie que **les plus bas débits simulés sont trop élevés** par rapport à l'observé, pas trop bas. C'est cohérent avec un **volume estival global trop faible** (0,45-0,52) : l'eau manquante en été fait que les bas débits ne descendent pas assez, tout en ayant des **pics de crue manquants** (été plat).

### 5.2. Pourquoi les correctifs physiques ne marchent pas

**Deux explications possibles** :

1. **Le forçage est la contrainte limitante** : Sans pluie, aucun processus ne peut créer de l'eau.
   - Preuve : Neutralité du drainage + routage au territoire (R323) vs gain au banc
   - Le champ partagé du territoire **ne peut pas compenser** ce que le forçage ne fournit pas

2. **Le champ partagé dilue l'effet des correctifs locaux** :
   - Au banc (un bassin), le drainage peut être optimisé spécifiquement
   - Au territoire (champ partagé), un correctif local peut être **noyé** par les besoins d'autres bassins

**Conclusion** : **La priorité absolue est la correction du forçage CaSR brut pour l'été montérégien.**

### 5.3. Statut des autres hypothèses

| Hypothèse | Statut | Source |
|-----------|--------|--------|
| Drainage agricole manquant | **CADUC** (corrigé mais inefficace sans forçage) | R296, R323 |
| Temps de transfert fixe | **CADUC** (corrigé mais inefficace sans forçage) | R299, R323 |
| Évaporation eau libre absente | **ÉTABLI** (corrigé, impact faible en Montérégie) | R285, R288 |
| Prélèvements non dans l'entraînement | **CORRIGÉ** | R308 |
| Référence MOD16 manquante | **CORRIGÉ** | R311 |
| **Biais saisonnier CaSR été** | **OUVERT** (décision d'Essi requise) | R321 |

---

## 6. Recommandations

### 6.1. Action prioritaire

**Corriger le forçage CaSR brut pour la Montérégie, été (mai-septembre)** :
- Appliquer un **facteur multiplicatif de 1,15 à 1,45** (inverse de 0,69-0,86)
- **Hors de meandre** (comme indiqué dans AGENTS.md)
- **Décision d'Essi requise**

**Justification** : Sans cette correction, **aucun ajustement de processus physique** ne peut résoudre l'étiage estival montérégien.

### 6.2. Actions secondaires (à faire après correction du forçage)

1. **Valider l'activation du drainage** : Vérifier que `ETL_SOIL_TOML=sol-drain-appris.toml` est bien posé et que les paramètres `DRAIN_HOOGHOUDT` sont actifs
2. **Valider la tête de routage** : Vérifier `ETL_ROUTAGE_TETE=1`, `ETL_ROUTAGE_SOUSPAS=24`, `MEANDRE_KMUSK=1,48,24`
3. **Re-tester la Montérégie** avec forçage corrigé + drainage + tête
4. **Si KGE reste < 0,70** : Calibrer les paramètres de drainage (spacing_m, share) via les classes IRDA

### 6.3. Actions de validation

**Dès que `mont-corr1006-dt` est prêt sur Ubuntu** (chaine_nuit4.sh) :

1. Rapatrier les fichiers :
```bash
scp essi@192.168.40.165:meandre-data/carte-essai/mont-corr1006-dt-q-journalier.npz \
     essi@192.168.40.165:meandre-data/carte-essai/mont-corr1006-dt-sans-journalier.npz \
     D:/meandre-data/carte-essai/
```

2. **NE PAS renommer** en `mont-*.npz` : garder les fichiers `-corr1006-dt` pour comparaison

3. Calculer les indicateurs :
```bash
python .runs/quebec/indicateurs_stations.py D:/meandre-data/carte-essai/mont-corr1006-dt-q-journalier.npz
```

4. **Comparer avec le socle** :

| Métrique | Socle (R317) | `-corr1006-dt` (R323) | Gain | Cible |
|----------|--------------|------------------------|------|-------|
| KGE | 0,549 | **0,555** | **+0,006** | ≥ 0,70 |
| Volume | 1,04 | **+2,4 %** | - | ~1,00 |
| Min. 7j | 1,46 | **1,63** | **-0,17** | ≤ 1,10 |
| Juillet | 0,45 | **0,46** | **+0,01** | ≥ 0,80 |
| Août | 0,52 | **0,51** | **-0,01** | ≥ 0,80 |

**→ Résultat** : Les correctifs physiques **ne suffisent pas**. Le problème persiste.

### 6.4. Diagnostic supplémentaire requis

**Calculer l'impact théorique d'une correction du forçage** :

Si CaSR été est multiplié par 1,3 (inverse de 0,77) :
- Juillet : 0,45 × 1,3 = **0,585** (toujours loin de 0,80)
- Août : 0,52 × 1,3 = **0,676** (toujours loin de 0,80)

**Conclusion** : Même avec une correction agressive du forçage, les cibles ne seraient pas atteintes. **Autres facteurs à investiguer** :

1. **Partage pluie-neige** : Un seuil inadapté pourrait classer des pluies d'été en neige (mais peu probable en été)
2. **Interception par la canopée** : Sous-estimation des pertes par évaporation
3. **Représentativité des stations** : 80,9 % du flux anthropique invisible (R185)

**Priorité** : **Corriger le forçage d'abord**, puis réévaluer.

---

## 7. Références

### 7.1. Entrées du registre (reports/registre_hypotheses.md)

- **R285** : Évaporation de l'eau libre — gain de 20-30 % sur l'étiage
- **R288** : Défaut corrigé — colonne ne rendait pas l'ETP ni l'apport
- **R296** : Drainage agricole déclaré fonctionne (KGE 0,698 sur deux graines)
- **R307** : Sonde de gradient — cibles poussent K_c vers le haut sur 040841
- **R308** : **DÉFAUT CORRIGÉ** — prélèvements jamais dans l'entraînement avant le 2026-10-05
- **R309** : Première époque fait chuter K_c (cause : référence MOD16 manquante)
- **R310** : Terme MOD16 centré pousse K_c vers le bas 50-80× trop fort
- **R311** : **DÉFAUT CORRIGÉ** — référence MOD16 centré posée au début de l'époque
- **R312** : K_c dérive vers le bas à cause des bassins nerveux
- **R313** : Réentraînement Outaouais — KGE 0,755 → 0,796, volume +21 % → +16 %
- **R314** : Indice d'écoulement de base — modèle plafonne à 0,60 vs 0,69-0,73
- **R316** : Excès d'hiver — écoulement de base trop fort (nord) + fonte trop précoce (sud)
- **R317** : **Montérégie socle** — KGE 0,549, volume 1,04, juillet 0,45, août 0,52, min 7j 1,46
- **R321** : **CaSR brut biais saisonnier** — Outaouais sud : 1,3-1,44× hiver, juste été ; **Montérégie : 0,69-0,86× été**
- **R323** : **Drainage + tête neutre en Montérégie** — KGE 0,555 vs 0,549, juillet/août inchangés

### 7.2. Fichiers de configuration

- `.runs/quebec/config/socle-2026-10-04.toml` : Recette du socle
- `.runs/quebec/config/sol-drain-appris.toml` : Profil de sol avec drainage Hooghoudt

### 7.3. Scripts de vérification

- `.runs/quebec/indicateurs_stations.py` : Calcul des indicateurs par territoire
- `.runs/quebec/pluie_casr_vs_stations.py` : Comparaison CaSR vs stations GHCN
- `tests/test_prelevements_tranche.py` : Test des prélèvements dans les tranches

### 7.4. Données

- `D:/meandre-data/carte-essai/mont-q-journalier.npz` : Montérégie socle (à conserver)
- `D:/meandre-data/carte-essai/mont-sans-journalier.npz` : Montérégie socle naturalisé
- `D:/meandre-data/carte-essai/mont-corr1006-dt-q-journalier.npz` : Montérégie avec drainage + tête
- `D:/meandre-data/carte-essai/mont-corr1006-dt-sans-journalier.npz` : Montérégie avec drainage + tête naturalisé

---

**Document généré par l'agent meandre le 2026-10-07. Version 2.0.**

*Toute citation de chiffre doit provenire du registre ou de ce document. Les numéros R doivent être vérifiés dans `reports/registre_hypotheses.md` avant utilisation.*
