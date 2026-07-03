# Journal d'expériences autonomes — semaine grève (début 2026-07-03)

## Protocole
- Un changement isolé à la fois, hypothèse écrite AVANT.
- Screening court sur SLSO, validation 30 epochs si prometteur.
- Sélection checkpoint sur DEV (kge_median). Held-out test 2022-24 gardé AVEUGLE, jamais tuné dessus, juge final seulement.
- GARDÉ si held-out médian s'améliore, ou neutre held-out + plus physique + dev robuste. Sinon JETÉ, raison notée.
- Autopilote réparé (beta_thr 0.10). Forçage quebec.zarr (station-based, timing bon).

## Référence à battre
| modèle | held-out médian | pooled |
|---|---|---|
| méandre v2 (baseline) | 0.653 | 0.784 |
| Hydrotel BRUT MG24HA | 0.651 | — |
| PORTRAIT (OI krigé, leaké, hors-concours) | 0.944 | — |

## Expériences

### EXP-1 : dqcel (célérité dépendante du débit) — pics
- Hypothèse : peak_ratio 0.88 (méandre sous-estime les pointes). Sur quebec.zarr le timing est bon (peak_lag 0), donc accélérer la célérité en crue (K_eff = K·(Qref/(Q+Qref))^dq_beta) relève les pics SANS casser r, contrairement à CaSR où ça amplifiait le bruit convectif.
- Changement : discharge_dependent_celerity=true, dq_beta=0.5. Sinon = config v2.
- Statut : LANCÉ.
