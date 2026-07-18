# Rapport scale-up Québec — semaine du 2026-07-14

Objectif demandé : un modèle méandre pour tout le domaine PHYSITEL (15 régions). Ce rapport donne l'état exact : acquis, verdicts pilotes, recette recommandée, et ce qui reste à décider ensemble avant la flotte.

## 1. Le résultat phare (inchangé et renforcé)

SLSO, held-out 2022-2024 strict, mêmes tronçons, mêmes jours : méandre 0.689 médian bat LES SIX membres de l'ensemble Hydrotel (MG24HK 0.673, LN24HA 0.666, MG24HA 0.651, MG24HQ 0.634, MG24HS 0.629, MG24HI 0.560), avec intervalles quantiles calibrés (cov_90 = 0.905, cov_50 = 0.498 sur 32 092 obs). Forçage CaSR auto-corrigé reproductible, maillage PHYSITEL.

## 2. Infrastructure Québec (prête, committée)

- 15 caches DuckDB régionaux (~28 000 nœuds), 178 stations importées.
- CaSR Québec complet : 19 tuiles, 37 Go, 0 erreur.
- Forçages corrigés par région (dé-crachinage + jour local + volume bilan régional).
- Configs et file d'entraînement, comparateur held-out contre les 6 membres Hydrotel, chaîne quantile.
- Harnais de validation Hydrotel WSL opérationnel sur toute plateforme (18 min le run MONT complet).

## 3. Le banc MONT : de l'écart inexpliqué aux causes attribuées

MONT (23 jauges, basses-terres agricoles) était la région d'échec type. Le harnais a tout disséqué :

| pilote | recette | médian held-out | verdict |
|---|---|---|---|
| v2 | fonte NeRF | 0.573 | base |
| v4 | + volume Budyko | 0.552 | β corrigé, r inchangé |
| v5 | météo krigée MELCCFP | 0.535 | hypothèse forçage RÉFUTÉE |
| v3/v9 | sol Hydrotel GELÉ (LN puis MG24HK) | -0.31 / 0.125 | échec systématique |
| v6 | + ETP Linacre calée régionale | 0.520 | β 1.02, vrai levier volume |
| v7 | + fonte régionale (taux + SEUILS) | **0.592** | RECETTE FINALE |
| v8 | fonte du membre MG24HK isolée | 0.532 | les calages sont des paquets |

Références Hydrotel MONT : meilleur membre 0.758 (MG24HK), médiane des membres ~0.636.

Validations à la décimale au passage : colonne sol (4780 UHRH, 806.8 vs 807.0 mm) et clone Linacre (491.9 vs 491.9 mm/an) contre Hydrotel 4.3.6.

## 4. Les trois lois apprises (généralisables à la flotte)

1. Le calage régional d'Hydrotel ne vit PAS dans bv3c.csv (uniforme partout) mais dans les coefficients d'ETP (Linacre ×0.4-0.5 optimisé par UHRH) et la fonte (taux ET seuils par région).
2. Loi des ancrages : ancrer les PROCESSUS scalaires régionaux (ETP, fonte) fonctionne ; GELER les champs que le NeRF doit apprendre (sol) casse le modèle, car le NeRF compense par le sol les divergences structurelles assumées de méandre.
3. L'ensemble Hydrotel = 6 calages équifinaux INCOHÉRENTS entre régions (MG24HK champion à MONT, banal à SLSO ; MG24HI 2e à MONT, dernier à SLSO ; dispersion 0.17). Argument central pour le continuum différentiable et le papier identifiabilité.

## 5. Recette de flotte proposée (à valider ensemble)

Sol NeRF libre + z_n + ETP Linacre régionale (clone validé, coeff par nœud) + fonte régionale LN24HA (taux/seuils par nœud) + fonte NeRF en modulation + volume bilan cohérent (lame + ETR Linacre) + forçage CaSR corrigé. Puis tête quantile par région (recette éprouvée).

Attendu : les régions type SLSO restent au-dessus de l'ensemble ; les régions type MONT remontent fortement (0.52 → 0.59 démontré) sans encore battre le meilleur membre local ; les régions pauvres en jauges (CND, LABI) profitent des ancrages mais l'entraînement conjoint multi-régions reste LE chantier structurel pour elles.

## 6. Restes ouverts

- r hiver résiduel MONT (0.59 vs 0.75 du meilleur membre) : après ETP et fonte, le prochain suspect est l'acheminement hiver/glace et le tassement du couvert. Non traité cette semaine.
- Entraînement conjoint multi-régions (NeRF partagé, jauges poolées) : la vraie réponse aux régions pauvres et la vision Québec unifiée.
- Régions à rivières régulées (Côte-Nord) : module barrages ou stations filtrées.
- VAUD sans jauge : régionalisation (correcteur d'attributs LOSO-validé disponible).
