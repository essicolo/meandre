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
- Résultat : REJET. held-out médian 0.524 (vs 0.653), pooled 0.717 (vs 0.784). peak_ratio 0.88→1.20 (sur-tire), r 0.886→0.803 (timing cassé). dev pic 0.685 (vs 0.731).
- Verdict : la célérité dépendante du débit sur-corrige les pics ET dégrade r, même au bon timing. Piste gentler dq_beta=0.25 possible mais le drop de r est structurel. JETÉ.

### EXP-2 : infiltration sol gelé au freshet
- Hypothèse : le Québec est freshet-dominé. Si l'infiltration sur sol gelé est mal gérée (porte de gel trop grossière), les pics de fonte souffrent. Améliorer la porte de gel → pics de freshet → médiane.
- À investiguer d'abord : comment la porte de gel agit dans bv3c2 (frozen gate sur pinf), puis un levier isolé.
- Résultat : REJET (held-out). dev pic 0.740 (vs 0.731, MIEUX) mais held-out médian 0.621 (vs 0.653, PIRE), pooled 0.765 (vs 0.784). Signature sur-apprentissage : améliore le dev, dégrade le held-out non stationnaire.
- Verdict : le gel continu ajuste mieux 2000-2021 mais généralise moins sur 2022-24 (régime réchauffé, moins de gel). Le held-out juge. JETÉ. Leçon : plus physique ≠ mieux généralisant hors régime.

### EXP-3 : ET Oudin 2005 (température-radiation)
- Hypothèse : McGuinness sur-évapore (594 mm/an vs Oudin 404, et ET~593 documenté sur CaSR vs MODIS 450). Moins d'ET → plus de Q → beta 0.92 vers 1.0. Oudin = optimal pluie-débit (27 formules comparées). Ne dépend que de T+lat+doy (compatible quebec.zarr).
- Changement : et_mode mcguinness → oudin. Sinon = config v2.
- Statut : LANCÉ.
- Résultat : REJET (held-out). dev pic 0.741 (vs 0.731) mais held-out médian 0.626 (vs 0.653), pooled 0.761 (vs 0.784). (NB : crash écriture fields = disque C plein, held-out calculé avant, verdict valide.)
- Verdict : Oudin (404 mm/an) réduit bien l'ET mais le held-out médian baisse. La contrainte MODIS-ET (w_et=1.0) recale probablement l'AET quel que soit le PET, donc changer la formule ne libère pas le débit espéré. JETÉ.

### OD-MODE : baseline open-data (HydroSHEDS 6166 + CaSR)
- But : mode entièrement reproductible remplaçant PHYSITEL. Attendu plus bas que 0.653 (CaSR plafonne r à 0.76), mais reproductible.
- Statut : LANCÉ.

### EXP-4 : hydrogramme de versant (use_hillslope_uh)
- Hypothèse : lisser au VERSANT (Nash) façon Hydrotel plutôt que dans le canal Muskingum diffusif. Risque : double-lissage → baisse les pics.
- Statut : en file après OD.
- Résultat OD baseline : FONCTIONNEL mais faible. held-out médian 0.291, pooled 0.475, r 0.562, beta 1.20 / vol_ratio 1.33 (SUR-PRODUCTION 33%).
- DIAGNOSTIC : le mesh HydroSHEDS sur-estime l'aire de drainage de +26% médian (ratio aire_modèle/aire_officielle 1.26, 50/180 stations en mismatch fort). Le modèle croit les bassins plus gros → sur-produit. Problème de SNAPPING jauge→tronçon, pas de physique.
- FIX : ré-accrocher chaque jauge au tronçon dont l'aire accumulée matche l'aire officielle (OD-MODE-v2). Le mode OD est livré et tourne ; la qualité viendra du bon snapping.
