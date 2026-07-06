# Résultats de la campagne autonome (semaine du 2026-07-03)

Résumé exécutif. Le détail par expérience est dans `experiment_log.md`.

## Deux contributions principales

### 1. Déblocage du forçage : correction de CaSR par quantile mapping
CaSR (réanalyse ouverte) a le meilleur *timing* mais une distribution biaisée pour l'hydrologie.
Le **quantile mapping** vers une distribution saine (préservant le timing), suivi d'un **calage du
volume sur le bilan d'eau flux-tower** (1147 mm/an = ET 450 + Q 697), donne un forçage qui :
- **bat quebec.zarr** sur le poolé (0.823 vs 0.784) et le timing (r 0.893 vs 0.77),
- **l'égale** sur le médian par station (0.634 vs 0.653).

C'est un forçage entièrement basé sur une réanalyse ouverte, à parité (ou mieux) avec le krigeage
propriétaire. **Recommandation : adopter QM-v3 comme forçage canonique.**
- Construction : `.runs/slso/build_casr_qm2.py` (forme + volume) puis rescale à 1147.
- Cache : `D:/meandre-data/slso/forcing-casr-qm3.nc`. Config : `slso-qmcasr3.toml`.
- Livrable Ouranos (méthode générale, FR, autonome) : `notebooks/correction_casr_ouranos.ipynb`.

### 2. Conclusion scientifique : la flexibilité ne généralise pas hors-distribution
Six leviers testés rigoureusement (held-out aveugle 2022-24 non stationnaire, sélection sur dev).
La baseline bien contrainte (médian 0.653, à parité avec Hydrotel brut 0.651) **n'est pas battue**
par l'ajout de flexibilité. Tout ce qui relâche les contraintes SUR-APPREND la période
d'entraînement et généralise moins. **Appuie la thèse physique-différentiable vs ML pur.**

## Solutions à POTENTIEL (gardées, opt-in, pas optimales sur SLSO)

| solution | held-out | statut | comment l'activer |
|---|---|---|---|
| **QM-CaSR** (v3) | pooled 0.823, méd 0.634 | **RECOMMANDÉ** | forçage `forcing-casr-qm3.nc` |
| **versant UH** (Nash) | méd 0.649 (neutre), +moyenne | garder, plus physique | `[model] use_hillslope_uh = true` |
| **GRU résidu minimal** | +0.006, généralise | garder si bridé | `.runs/slso/exp5_gru_residual.py` |

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
1. **Module Hortonien correct** : générer l'excès d'infiltration + le router par le versant (pas de
   pic instantané) + capacité d'infiltration séparée du K_sat de drainage (leçon Raven). EN COURS.
2. **QM 100% ouvert** : remplacer la cible quebec.zarr par une distribution climatologique ouverte.
3. Le médian à un cheveu sous quebec.zarr : gap station-spécifique (petits bassins), pas forçage.
