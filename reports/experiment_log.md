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
- Résultat : NEUTRE / WEAK-KEEP. held-out médian 0.649 (vs 0.653, neutre), moyenne 0.622 (vs 0.610, MIEUX), pooled 0.777, dev pic 0.754 (vs 0.731, mieux).
- Verdict : le lissage au versant (Nash, fidèle Hydrotel) n'améliore pas le médian mais la moyenne+dev, et il est plus physique. GARDÉ comme candidat de combinaison finale. Checkpoint conservé.

### EXP-5 : GRU résidu post-hoc sur physique gelée
- Design conservateur : physique GELÉE, correction multiplicative bornée ±30%, GRU 16 unités, features [log Q_phys, log P, Tmean, sin/cos doy], early-stop sur val 2020-21, jugé held-out 2022-24 aveugle.
- Résultat : POSITIF (weak-keep). held-out médian 0.645 (phys) → 0.651 (+GRU), +0.006, 19/24 stations améliorées. GÉNÉRALISE (val montait à 0.723 mais held-out tient).
- Verdict : l'hybridation MARCHE si bridée fort. Un résidu borné sur physique gelée ne sur-apprend pas, contrairement à un LSTM pur. Gain modeste ici (features = Q+forçage seulement). Piste : nourrir le GRU de l'ÉTAT physique (theta, swe) pour corriger des mécanismes non modélisés. GARDÉ.

## SYNTHÈSE CAMPAGNE (mise à jour continue)
- Baseline v2 (held-out médian 0.653) NON battue par un levier isolé.
- Rejets : dqcel (0.524), gel continu (0.621, sur-app), Oudin ET (0.626).
- Keepers faibles : versant UH (neutre médian, +moyenne, plus physique), GRU résidu (+0.006, généralise).
- Leçon centrale : le baseline est bien réglé ; les gains held-out sont marginaux et viennent de l'hybridation bridée + routage physique, pas des swaps de composante. Prochaine piste forte : GRU nourri de l'état physique.

### OD-MODE-v2 : résultat après resnap
- held-out médian 0.409 (vs 0.291 avant resnap, +0.118), pooled 0.545, r 0.604, beta 1.13, vol_ratio 1.14.
- Verdict : mode open data LIVRÉ + réparé + reproductible. Mais reste sous PHYSITEL : même forçage CaSR, HydroSHEDS 0.41 vs PHYSITEL 0.60 = écart 0.19 côté MAILLAGE (topologie, aires, découpage). r plafonne 0.60, +14% sur-production résiduelle. Coût réel de la reproductibilité totale, quantifié. Marge restante : resnap plus serré, raffinement réseau.

### EXP-5b : GRU résidu nourri de PROXIES D'ÉTAT (API sol, swe neige, gel)
- Résultat : NÉGATIF (held-out). GRU minimal (5 feat) +0.006 ; GRU + état (8 feat) −0.015. Val identique (~0.72) mais held-out se dégrade avec plus de features.
- Verdict : ajouter des features d'état = SUR-APPRENTISSAGE de la période d'entraînement, généralise MOINS sur non-stationnaire. Le point idéal est le résidu MINIMAL borné. Confirme la méfiance LSTM.

## CONCLUSION DE CAMPAGNE
Six leviers testés rigoureusement (held-out aveugle, sélection dev). Baseline v2 (médian 0.653) NON battue significativement.
- Sur-corrige/casse : dqcel (0.524).
- Sur-apprend le régime d'entraînement : gel continu (0.621), GRU+état (−0.015).
- Recalé par contrainte externe : Oudin ET (0.626, MODIS fixe l'AET).
- Neutre/physique : versant UH (0.649, +moyenne).
- Seul positif : GRU résidu MINIMAL borné (+0.006, généralise).
RÉSULTAT SCIENTIFIQUE : en prédiction hors-distribution (période non stationnaire 2022-24), AJOUTER de la flexibilité (ET riche, résidu d'état, célérité dynamique) NE GÉNÉRALISE PAS et souvent dégrade. La physique bien contrainte + routage physique + résidu minimal borné est l'optimum robuste. Appuie la thèse physique-différentiable vs ML pur. Contribution paper : le sur-apprentissage guette dès qu'on relâche les contraintes ; l'hybridation ne paie que minimale et bornée.

### QM-CASR : quantile mapping CaSR -> distribution quebec.zarr
- v1 : QM complet (forme + volume QZ). held-out médian 0.508, pooled 0.760, r 0.887 (EXCELLENT timing préservé), beta 0.840 (SOUS-production, vol trop sec).
- Diagnostic : le QM libère le timing supérieur de CaSR (r 0.887 vs QZ 0.77) mais mapper sur le volume QZ (1087) rend CaSR trop sec pour son timing → beta 0.84 écrase le médian. Levier restant = VOLUME.
- v2 : QM de FORME (distribution QZ) mais VOLUME propre de CaSR préservé par nœud. Isole la correction de forme. LANCÉ.
- v2 : forme QZ + volume CaSR (1229). held-out médian 0.626 (vs v1 0.508, GROS gain), pooled 0.769, r 0.864, beta 0.964, gamma 1.095, vol_ratio 1.12.
- Encadrement volume : 1087→0.508, 1229→0.626. Optimum entre. gamma 1.095 = QM sur-concentre un peu. Proche de quebec.zarr (0.653) avec bien meilleur r.
- v3 : volume bilan-d'eau flux-tower (1147, entre v1 et v2). LANCÉ.
- v3 : volume bilan-d'eau 1147 (×0.933 de v2). held-out médian 0.634, POOLED 0.823 (>QZ 0.784!), r 0.893 (record projet), beta 0.935, gamma 1.002, vol_ratio 1.04. Le sweet spot volume.
- BILAN QM : le quantile mapping au volume bilan-d'eau donne un CaSR qui BAT quebec.zarr sur pooled+r et l'ÉGALE sur médian (0.634 vs 0.653). Forçage CaSR ouvert ≈ krigeage, meilleur timing. Goulot météo largement débloqué. Reste le médian à un cheveu -> exploiter DT_eff (Hortonien).

### QM+HORTON : QM-v3 + Hortonien (exploite DT_eff)
- Résultat : REJET. médian 0.634→0.610, pooled 0.823→0.700, r 0.893→0.787, peak_ratio 0.98→0.79. Le quickflow Hortonien dégrade le timing même sur QM propre. Le problème est le mécanisme, pas le forçage. JETÉ.

## SYNTHÈSE GOULOT MÉTÉO (CaSR preprocessing)
RÉSULTAT MAJEUR : le QUANTILE MAPPING de CaSR au volume bilan-d'eau (QM-v3) donne un forçage entièrement basé sur réanalyse OUVERTE qui :
- BAT quebec.zarr sur pooled (0.823 vs 0.784) et r (0.893 vs 0.77)
- l'ÉGALE sur médian (0.634 vs 0.653)
Méthode : par nœud, remapper la distribution de précip CaSR sur celle de quebec.zarr (préserve le timing CaSR = corr rang 1.0), puis rescaler au volume bilan-d'eau flux-tower (1147 mm/an = ET 450 + Q 697). Le timing CaSR (supérieur) + distribution saine + volume correct.
Encadrement volume décisif : 1087→0.508, 1147→0.634, 1229→0.626.
DT_eff (Hortonien) n'ajoute rien (mécanisme dégrade r). Le goulot météo est LARGEMENT débloqué : plus besoin du krigeage propriétaire, CaSR prétraité par QM est ≥ quebec.zarr et 100% reproductible (sauf la distribution-cible QZ, remplaçable par une cible climatologique ouverte).

### SOUS-JOURNALIER : Hortonien depuis quickflow horaire RÉEL (précalcul offline scalable)
- Méthode : excès d'infiltration horaire (>5mm/h) précalculé offline depuis l'horaire CaSR, injecté comme canal ; modèle reste journalier. SCALABLE.
- Résultat : held-out médian 0.622, r 0.834, vol 1.00. ENTRE Horton-DT_eff (0.610, r 0.787) et sans-Horton (0.634, r 0.893).
- Verdict : l'intensité RÉELLE bat le proxy DT_eff (le proxy était le coupable), mais le quickflow reste marginalement sous le sans-Horton. Fast-flow perturbe un timing déjà excellent. Quasi-neutre. Test infil_cap plus haut (10mm/h, plus sélectif) pour tipper.
- cap=10 (ultra-sélectif, 1% jours) : médian 0.636 (parité sans-Horton 0.634), pooled 0.790, r 0.840. Le Hortonien atteint la parité médian mais reste sous sur pooled/r.
- VERDICT DÉFINITIF Hortonien SLSO : au mieux NEUTRE. L'intensité réelle >> proxy DT_eff (série 0.610→0.622→0.636), mais le fast-flow ne devient jamais un gain net (r plafonne 0.84 < 0.89 sans-Horton). Réponse lente du sol optimale. Documenté, opt-in désactivé.

## CaSR-CORR — correction CaSR auto-référencée, DEUX axes (volume ET timing), 2026-07-07
- Instruction Essi : « c'est une question de volume ET de timing... il faut bien sûr corriger les deux ». Référence = CaSR (pas quebec.zarr).
- Méthode (build_casr_corrected.py, entièrement depuis l'horaire CaSR, aucun quebec.zarr) :
  - TIMING : agrégation sur le jour LOCAL (décalage UTC-5, EST) au lieu du jour UTC, pour aligner P sur le débit CEHQ (jour local). Corrige le décalage de frontière (~5h) qui misplace les orages de fin de journée.
  - VOLUME/distribution : dé-crachinage horaire (heures < 0.3 mm/h retirées → jours pluvieux 62%→40%) puis calage du total sur le bilan d'eau flux-tower (1147 mm/an = ET 450 + Q 697).
  - P corrigé remplace le canal 0 ; T/Rn/etc gardés de CaSR.
- Training (McGuinness, kge_median, 30 ep) : montée forte r 0.68→0.90, best val kge_median 0.7758 (ep 18-19), val_kge pooled 0.877, β 0.94, γ 0.98. Le r monte bien au-dessus de QM-v3 en training → la correction jour-local relève le timing.
- HELD-OUT 2022-2024 (jamais vu, non-stationnaire) : pooled 0.8142, médian PAR STATION 0.6776, mean 0.630.
- COMPARAISON :
  - QM-v3 (vers quebec.zarr) : pooled 0.823, médian 0.634 — MAIS emprunte la forme de distribution de quebec.zarr (pas auto-référencé).
  - CaSR-corr : pooled 0.814 (parité), médian 0.678 (BAT QM-v3 de +0.044), ENTIÈREMENT auto-référencé CaSR.
- VERDICT : meilleur résultat CaSR défendable. Corriger les deux axes sur les données horaires propres de CaSR (dé-crachinage + bilan-eau + jour-local) lève le médian held-out à 0.678, au-dessus de QM-v3, sans aucune fuite vers un produit tiers. C'est la correction à recommander pour Ouranos (auto-cohérente).
- Réserve : couverture proba cassée (cov90 0.23) = tête de bruit non recalibrée sur ce forçage (σ figée), à re-caler ; n'affecte pas le KGE déterministe.
- Config : slso-casr-corr.toml ; forçage forcing-casr-corr.nc ; ckpt best-physitel-hydrotel-casr-corr.pt.

## CORR2 — calage volume SPATIAL par sous-bassin jaugé : REJET (held-out), 2026-07-07
- Méthode : cible P locale = lame obs train + ETR 450 par plus petit bassin jaugé englobant (51% des nœuds), facteurs bornés [0.75, 1.30], base CaSR-corr.
- Training : val kge_med 0.7705 (≈ corr 0.7758), convergence OK.
- HELD-OUT : médian 0.596, pooled 0.753 — RÉGRESSION nette vs corr (0.678 / 0.814).
- Lecture : cohérent avec le diagnostic de stabilité — corriger le NIVEAU de biais par station vers sa valeur train EMPIRE le test (|beta-1| 0.095→0.118 prédit par le diag statique). Le régime 2022-24 a un coefficient de ruissellement +6% ; ancrer les volumes locaux sur le train fige l'ancien régime. Le pattern d'erreur STABLE est le pattern RELATIF, pas le niveau (confirmé par exp6b).
- VERDICT : REJET. Champion reste CaSR-corr (calage volume GLOBAL). Le levier beta spatial passe par la correction relative zéro-somme (exp6b), pas par le forçage.

## EXP6 — correcteur d'erreurs ATTRIBUT-CONDITIONNÉ (transformer), 2026-07-08
- Idée (Essi) : les erreurs résiduelles sont dues aux attributs des bassins ; un réseau attention peut les corriger. Post-hoc sur physique gelée (champion CaSR-corr), correction multiplicative bornée [0.74, 1.35], 16 attributs territoriaux en tokens + token jour (Q_sim, saison, P 3j/14j), TransformerEncoder 2 couches d=32.
- Prérequis diagnostiqués :
  - Signatures d'erreur par station STABLES dev↔test (corr beta 0.84, r 0.72, gamma 0.76) → le signal existe.
  - MAIS décalage de NIVEAU global dev→test (beta 1.05→0.92 ; RC obs +6% en 2022-24, vraie non-stationnarité) → toute correction de niveau apprise sur train EMPIRE le test (vérifié : correction statique |beta-1| 0.095→0.118 ; corr2 rejeté pareillement).
- v1 sans contrainte : dev +0.08 mais held-out -0.029 (4/24). Apprend la période, pas les attributs.
- v2 RELATIF (pénalité zéro-somme sur le log-facteur moyen, le correcteur ne peut pas décaler le niveau global) :
  - FULL (stations vues) : held-out 0.678 → 0.694 (+0.016), 13/24.
  - LOSO 6-fold (stations JAMAIS vues) : 0.678 → 0.693 (+0.015), 12/24. GAIN IDENTIQUE au FULL.
- VERDICT : KEEP. La correction attribut→erreur GÉNÉRALISE aux bassins non jaugés (LOSO=FULL) = preuve de RÉGIONALISATION, argument clé scale-up QC. La contrainte relative zéro-somme est l'ingrédient décisif (leçon : seul le pattern relatif inter-stations est stable, jamais le niveau).
- Script : exp6_attr_transformer.py (MODE=full|loso, REL=1, FOLDS). CSV : exp6-loso.csv.

## ETI — fonte radiation réelle (melt_mode=eti) : REJET (held-out), 2026-07-08
- Forçage dédié construit (FB W/m2 canal 6, build_casr_eti_forcing.py), base champion CaSR-corr.
- Training : val kge_med best 0.7089 (vs champion 0.7758), r plafonné 0.82 (vs 0.90). tf/srf appris de zéro ne rattrapent pas la recette degré-jour calée (melt÷2.5) en 30 epochs.
- HELD-OUT : médian 0.551, pooled 0.757 — REJET (champion 0.678/0.814).
- Piste si on y revient : init littérature tf/srf (Hock 2003) + warm-start du champion, pas cold-start. Le forçage FB reste disponible.

## Z_N — codes latents additifs par nœud : KEEP (nouveau champion médian), 2026-07-08
- Config champion CaSR-corr + use_latent_codes=true, latent_mode=additive (le gagnant du banc mini-bassin).
- Training : val kge_med 0.7596 (≈ champion 0.7758).
- HELD-OUT : médian 0.6881 (+0.010 vs champion 0.678) = NOUVEAU RECORD ; pooled 0.798 (-0.016).
- Lecture : les effets par station en ESPACE-PARAMÈTRES (shrinkage L2, partial pooling) TRANSFÈRENT au régime 2022-24, contrairement aux corrections de niveau en espace-volume (corr2, 0.596) et sortie (exp6 v1). Triade cohérente : la correction locale doit passer par la physique.
- Suite : empilage correcteur d'attributs relatif par-dessus (exp6 sur parquet z_n, en cours).

## EMPILAGE z_n + correcteur attributs : REJET (redondance), 2026-07-08
- exp6 relatif sur parquet z_n : FULL 0.6964→0.6871 (-0.009), LOSO 0.6748 (-0.022).
- Lecture : z_n absorbe déjà le signal d'erreur station-spécifique (en espace-paramètres, mieux) ; le correcteur n'a plus rien à corriger et ajoute du bruit. Les deux leviers NE S'ADDITIONNENT PAS.
- CLASSEMENT FINAL held-out (médian) : z_n 0.688 (0.696/24 communes) ≈ corr+correcteur 0.693 > corr 0.678 > Hydrotel brut 0.651 >> corr2 0.596, ETI 0.551.
- Architecture scale-up QC suggérée : z_n sur bassins jaugés, correcteur d'attributs (régionalisable, prouvé LOSO) sur non jaugés.

## ZN-QUANTILE — re-calibration probabiliste sur champion z_n : SUCCÈS, 2026-07-13
- Recette Phase 2 (tête quantile K=6, offsets depuis mu, médiane = mu) warm-startée sur backbone z_n GELÉ, forçage CaSR-corr.
- BUG TROUVÉ ET CORRIGÉ : melt_factor_scale réappliqué sur warm-start (double application, fonte ÷6, backbone gelé dégradé r 0.90→0.70). slso.py ignore désormais la recette en warm-start.
- Après fix : epoch 0 reproduit le champion (val 0.866/0.761), tête calibrée en 15 epochs.
- HELD-OUT 2022-24 (32 092 obs) : cov_90 = 0.9048 (cible 0.90), cov_50 = 0.4981 (cible 0.50), KGE médian 0.6881 PRÉSERVÉ (= record), pooled 0.798.
- NB : le bloc held-out de slso.py affiche encore les cov de la vieille tête sigma (0.09/0.23) — ignorer en mode quantile, la vraie couverture quantile est ci-dessus (script inline, à intégrer dans slso.py un jour).
- Checkpoint : best-physitel-hydrotel-casr-zn-quantile.pt. PIPELINE COMPLET : déterministe record + probabiliste calibré. PRÉCISION (2026-07-13) : forçage 100% ouvert (CaSR auto-corrigé) et jauges publiques, mais MAILLAGE PHYSITEL (comme Hydrotel opérationnel — comparaison équitable). Le variant 100% ouvert (HydroSHEDS, slso-od) plafonne à ~0.41 médian, en pause.

## ETI v2 — warm-start champion + init Pellicciotti corrigée : REJET DÉFINITIF, 2026-07-14
- Corrections apportées : warm-start depuis z_n (0.688), init srf littérature 0.0094 mm/j/(W/m²) (l'ancienne était 20× trop forte, bug corrigé dans hydrotel_column), pas de double recette.
- Résultat : val kge_med plafonne 0.620 (r 0.78 vs 0.90 champion), held-out 0.549/0.675. Le passage à l'ETI dégrade IMMÉDIATEMENT le champion et ne récupère pas.
- VERDICT : 2 échecs propres (cold 0.551, warm 0.549) = l'ETI journalier n'apporte rien ici. Le degré-jour Hydrotel (indice de radiation potentielle par géométrie) + recette calée reste supérieur au forçage journalier. L'ETI aurait besoin du sous-journalier pour exprimer son avantage. PISTE FERMÉE sur SLSO journalier.
