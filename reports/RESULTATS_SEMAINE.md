# Résultats de la campagne autonome (semaine du 2026-07-03)

Résumé exécutif. Le détail par expérience est dans `experiment_log.md`.

## Deux contributions principales

### 1. Déblocage du forçage : correction CaSR auto-référencée (volume ET timing)
CaSR (réanalyse ouverte) a le meilleur *timing* mais une distribution biaisée pour l'hydrologie.
Le diagnostic traîné toute la semaine : c'est un problème de **volume ET de timing**. La correction
finale, construite **entièrement depuis l'horaire de CaSR** (aucun produit tiers, `build_casr_corrected.py`) :
- **TIMING** : agrégation sur le **jour local** (décalage UTC-5, EST) au lieu du jour UTC, pour aligner
  P sur le débit CEHQ (mesuré en jour local). Corrige le décalage de frontière (~5h) qui misplace les
  orages de fin de journée. → r monte 0.68→0.90 en training.
- **VOLUME** : dé-crachinage horaire (heures < 0.3 mm/h retirées, jours pluvieux 62%→40%) puis calage
  du total sur le bilan d'eau flux-tower (1147 mm/an = ET 450 + Q 697).

Held-out 2022-24 (non stationnaire, jamais vu) : **pooled 0.814, médian par station 0.678.**
- **Bat QM-v3** sur le médian (0.678 vs 0.634), l'égale sur le poolé (0.814 vs 0.823),
- **et entièrement auto-référencé sur CaSR** — QM-v3 empruntait la forme de distribution de quebec.zarr.

C'est la correction à recommander : auto-cohérente, applicable par quiconque à sa propre CaSR sans
produit propriétaire. **Recommandation : adopter CaSR-corr comme forçage canonique** (remplace QM-v3).
- Construction : `.runs/slso/build_casr_corrected.py`. Cache : `D:/meandre-data/slso/forcing-casr-corr.nc`.
- Config : `slso-casr-corr.toml`. Checkpoint : `best-physitel-hydrotel-casr-corr.pt`.
- QM-v3 (`build_casr_qm2.py`, `slso-qmcasr3.toml`) gardé comme comparatif, mais dépassé (emprunt tiers).
- Livrable Ouranos (méthode générale, FR, autonome) : `notebooks/correction_casr_ouranos.ipynb`.

### 2. Conclusion scientifique : la flexibilité ne généralise pas hors-distribution
Six leviers testés rigoureusement (held-out aveugle 2022-24 non stationnaire, sélection sur dev).
La baseline bien contrainte (médian 0.653, à parité avec Hydrotel brut 0.651) **n'est pas battue**
par l'ajout de flexibilité. Tout ce qui relâche les contraintes SUR-APPREND la période
d'entraînement et généralise moins. **Appuie la thèse physique-différentiable vs ML pur.**

## Solutions à POTENTIEL (gardées, opt-in, pas optimales sur SLSO)

| solution | held-out | statut | comment l'activer |
|---|---|---|---|
| **z_n codes latents** (espace-paramètres) | méd 0.688 = RECORD jaugé | **CHAMPION jaugé** | `[model] use_latent_codes = true` + `latent_mode = "additive"` |
| **CaSR-corr + correcteur attributs** | méd 0.693, RÉGIONALISABLE (LOSO prouvé) | **CHAMPION non-jaugé** | forçage corr + `exp6_attr_transformer.py` REL=1 |
| **CaSR-corr** (volume+timing auto-réf) | pooled 0.814, méd 0.678 | forçage canonique | forçage `forcing-casr-corr.nc` |
| QM-CaSR (v3, emprunte quebec.zarr) | pooled 0.823, méd 0.634 | dépassé (garder comparatif) | forçage `forcing-casr-qm3.nc` |
| **versant UH** (Nash) | méd 0.649 (neutre), +moyenne | garder, plus physique | `[model] use_hillslope_uh = true` |
| **GRU résidu minimal** | +0.006, généralise | garder si bridé | `.runs/slso/exp5_gru_residual.py` |

NB : z_n + correcteur d'attributs NE s'additionnent PAS (même signal, redondance mesurée). Architecture
scale-up QC : z_n où il y a des jauges, correcteur d'attributs ailleurs. Rejets supplémentaires documentés
dans experiment_log : corr2 (volume spatial par bassin, 0.596 — les niveaux ne transfèrent pas) et ETI
cold-start (0.551 — à retenter en warm-start + init Hock, forçage FB déjà construit).

Note GRU : positif SEULEMENT en version minimale (features = Q + forçage, correction bornée ±30%,
early-stop). Enrichir les features (état physique) fait SUR-APPRENDRE (voir rejets).

## Solutions REJETÉES (clairement non fonctionnelles sur SLSO)
Gardées en opt-in (défaut OFF) pour la traçabilité, mais **ne pas activer** :
- `discharge_dependent_celerity` (dqcel) : sur-corrige les pics, casse r (0.524).
- porte de gel continue (`[soil] frozen_gate = true`) : sur-apprend (0.621).
- `et_mode = "oudin"` : recalé par MODIS-ET, pas de gain (0.626). (reste une option ET légitime)
- Hortonien (`use_hortonian`) : le quickflow pointu dégrade r partout, même sur QM propre. **Le
  mécanisme est en cause** → chantier « module Hortonien correct » en cours (router l'excès par le
  versant au lieu d'un pic instantané).
- GRU nourri de l'état physique : sur-apprend (−0.015).

## Mode open data (mis en pause à la demande)
HydroSHEDS + CaSR fonctionne (`.runs/slso-od/`), ré-snapping des jauges validé (aire 1.26→1.01,
médian 0.29→0.41), mais reste sous PHYSITEL (−0.19, qualité du réseau ouvert). Base :
`basin-resnap.duckdb`. Config : `slso-od-resnap.toml`. À reprendre si besoin de reproductibilité
totale du maillage.

## Chantiers ouverts (à décider au retour)
1. **Module Hortonien correct** : FERMÉ (verdict au mieux neutre sur SLSO, voir experiment_log). L'intensité
   horaire réelle bat le proxy DT_eff (série 0.610→0.622→0.636) mais le fast-flow ne devient jamais un
   gain net (r plafonne 0.84 < 0.89 sans-Horton). Réponse lente du sol optimale ici. Opt-in gardé, OFF.
2. **Forçage 100% ouvert auto-référencé** : RÉSOLU par CaSR-corr (§1). Plus besoin de cible quebec.zarr.
3. **Re-calage σ sur CaSR-corr** : la tête de bruit proba est cassée sur ce forçage (cov90 0.23, σ figée
   d'un autre run). Re-caler avant tout usage probabiliste. N'affecte pas le KGE déterministe.
4. ATTENTION chiffre fantôme : il n'existe AUCUN 0.75 held-out. Le 0.749 était le kge_med DEV de
   méandre (période de sélection 2019-21), le 0.761 Hydrotel était mesuré période complète (juin).
   Sur le held-out 2022-24 strict, tout ce qui a été mesuré : Hydrotel brut 0.651, méandre+quebec.zarr
   0.653, méandre+CaSR-corr 0.678 = MEILLEUR médian held-out jamais mesuré sur ce bassin. Les leviers
   restants (volume spatial, fonte ETI, z_n) visent 0.70+, comparé à 0.678, pas à un 0.75 imaginaire.
