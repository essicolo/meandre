# État des lieux de méandre — 2026-08-19

Topo demandé par Essi. Tout ce qui suit est MESURÉ ; aucune valeur reprise d'un commentaire ou d'un script (leçon du 0.82 fantôme du 11 août).

## 1. L'état optimal du modèle, aujourd'hui

Il y a DEUX modèles champions, sur deux cas différents, et ils ne se ressemblent pas.

### Québec / PHYSITEL (le cas qui porte l'enjeu) — champion OUTV à 0.7880

Recette exacte, à reproduire telle quelle :

```
ETL_REGION=outv ETL_EPOCHS=30 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb
ETL_INIT_HYDROTEL=sauf_ks     # courbe de rétention d'Hydrotel imposée SAUF K_sat et porosités
ETL_ETP=linacre               # ETP Linacre CALÉE de la plateforme (coefficient par UHRH)
ETL_SEUIL_NEIGE=1             # seuil pluie/neige de la plateforme
ETL_MELT_DIR=<plateforme>     # fontes calées
ETL_AQUIFER=1 ETL_KREC=<calibré> ETL_KGW=0.0645
ETL_WET=0 ETL_WSNOW=0.3 ETL_NO_LATENT=1
+ occupation du sol, milieux humides, phénologie, HGM, lacs (défauts ON depuis le 10 août)
```

| repère (tenu de côté 2022-2024, 16 stations) | KGE médian |
|---|---|
| socle, ZÉRO époque | 0.7389 |
| socle, 30 époques | 0.7810 |
| **socle + aquifère, 30 époques (CHAMPION)** | **0.7880** |
| ENSEMBLE Hydrotel, médian par station | 0.7711 |
| meilleur MEMBRE (MG24HK) | 0.8299 |
| meilleur membre PAR STATION (oracle) | 0.8543 |

**On bat l'ensemble opérationnel de 0.017. Il reste 0.042 jusqu'au meilleur membre.**

Transposition du socle (zéro époque, aucune adaptation) : GASP 0.7749, SAGU 0.7517, SLNO 0.7106, MONT 0.4869. Contre l'ensemble : au-dessus sur OUTV et GASP, en dessous sur SAGU (0.7933) et MONT (0.6631, bassin hyper-régulé, 1.19 barrage par nœud, Hydrotel y plafonne aussi).

### SLSO / cas d'essai — champion zn inchangé à 0.688 / 0.814 groupé

Trois tentatives d'amélioration cette semaine, TOUTES neutres ou négatives (voir §3).

## 2. Le run de la nuit : occupation du sol sur SLSO — RÉGRESSION

30 époques en départ à chaud, occupation PHYSITEL réelle posée (forêt 58.2 %, eau 2.2 %, imperméable 3.8 %, humides 6.7 %) là où la colonne SLSO voyait 100 % de sol nu depuis toujours.

| | validation 2019-2021 | tenu de côté 2022-2024 |
|---|---|---|
| champion zn (sans occupation) | 0.7758 | **0.688** / 0.814 groupé |
| zn-occ (avec occupation) | **0.7788** (époque 3) | **0.6595** / 0.7995 groupé |

**La validation CROISE (+0.003), le tenu de côté RECULE (-0.029).** C'est la signature déjà documentée sur ce cas : la validation 2019-2021 et le tenu de côté 2022-2024 ne classent pas pareil, à cause de la non-stationnarité climatique mesurée le 1er juin (P +28 % JJA, T +1.5 °C DJF). Sélectionner sur la validation SLSO est donc peu fiable, et ce résultat le confirme une fois de plus.

**MAIS les paramètres appris sont physiques**, ce qui est le second juge et il est positif :

| paramètre | valeur apprise | littérature |
|---|---|---|
| K_sat_1 | 80 mm/j | Rawls 1982 loam 60-130 |
| C_f (fonte) | 4.47 mm/°C/j | Hock 2003 forêt 2-4.5 |
| T_melt | -0.49 °C | proche de 0 attendu |
| porosité_1 | 0.402 | loam 0.40-0.46 |
| f_vert | 0.50 / 0.60 / 0.70 | NON collapsé (le collapse à 0.03 était le défaut de mai) |

Lecture : donner le vrai territoire produit des paramètres crédibles mais un score de tenu de côté moindre. C'est exactement l'arbitrage documenté pour la recharge — le débit seul ne récompense pas le réalisme physique.

## 3. Bilan des leviers testés (tous mesurés, un seul changement chacun)

Sur OUTV, contre le socle à 0.7810 :

| levier | résultat | verdict |
|---|---|---|
| aquifère (recharge calibrée) | 0.7880 | **+0.007, SEUL GAIN** |
| MODIS ET rallumé | 0.7744 | neutre (-0.007) |
| GRACE retiré | 0.7616 | **GRACE apporte +0.02** |
| ancrage MG24HK + McGuinness calée | 0.7547 | le calage 0.83 n'est pas transportable |
| taux d'apprentissage réduit | 0.7111 | pire que ne rien apprendre |
| ETP apprise (MLP) | 0.7443 | hérite le biais de niveau de MODIS |
| CaSR brut | 0.7277 | nos corrections valent leur coût |
| recharge libérée (5e-5) | 0.4446 | effondrement, balayage monotone décroissant |

Sur SLSO, contre le champion à 0.688 : forçage fusionné 0.6717 (neutre), occupation 0.6595 (négatif).

**Onze hypothèses réfutées cette semaine.** Le socle actuel a survécu à toutes ses explications concurrentes.

## 4. Ce qui reste ouvert, avec le test qui tranche

1. **Déficit hivernal** (février à 0.69 de l'observé, plus gros écart mensuel ; Hydrotel fait l'erreur inverse à 1.24). L'aquifère n'en a comblé qu'une fraction, la recharge est réfutée comme levier. Piste restante : le taux de vidange k_gw, ou une nappe régionale absente des deux modèles.
2. **Excès d'été résiduel** (juillet à 0.87). Vaut ~0.02 d'après le test saisonnier.
3. **Vitesse** : la carte n'est utilisée qu'à 20-25 %, le goulot est le pilotage Python (9132 pas × ~100 niveaux topologiques). Gain potentiel ×3-4 sur le débit d'expériences, supérieur à tout levier de score restant. Profilage à faire.
4. **MODIS en tendance** : implémenté et testé (5 tests verts), jamais smoké. Aujourd'hui l'ET est ancrée en NIVEAU alors que MODIS est biaisé de +30 % contre les tours de flux.
5. **Recharge comme livrable** : le débit seul la préfère quasi nulle, la réalité québécoise se compte en dizaines à centaines de mm/an. Contrainte indépendante requise.
6. **Bug de sélection corrigé le 18 août** : la tolérance de 0.005 s'appliquait au KGE, jetant les améliorations fines et gonflant les coupes de LR. Tous les scores historiques sont donc des PLANCHERS légers.

## 5. Recommandation

Le chantier occupation SLSO est CLOS sur un résultat négatif en tenu de côté : ne pas remplacer le champion zn. Prochaine priorité, par ordre de rendement attendu : la VITESSE (débloque tout le reste), puis l'hiver sur OUTV (le seul écart gros et localisé), puis MODIS en tendance. La flotte régionale reste verrouillée jusqu'à stabilisation.
