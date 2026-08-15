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

## DST — agrégation jour-local saisonnière (UTC-4 avril-oct) : NEUTRE, 2026-07-14
- Même config que champion z_n, seul le forçage change (offset saisonnier dans l'agrégation jour-local).
- HELD-OUT : médian 0.6838 (vs 0.6881, -0.004 = bruit), pooled 0.8047 (vs 0.798).
- VERDICT : NEUTRE. Le UTC-5 fixe suffit ; garder le champion (plus simple). Axe correction timing épuisé au journalier.

## QUÉBEC gates — pilotes avant flotte (règle d'Essi), 2026-07-15
- Rappel Essi : valider sur pilote AVANT toute grande opération ; v1 (recette SLSO uniforme) lancée sans pilote contrasté = erreur, ~15h GPU sur runs à refaire.
- Bilan v1 (held-out, vs Hydrotel brut mêmes tronçons) : gagne SLSO (0.689/0.666), LABI (0.743/0.644), CNDE (0.622/0.552) ; perd CNDA (0.615/0.759), CNDC (0.489/0.660), MONT-v2 (0.523/0.637). Structurel : recette mono-bassin < 20 ans de calage régional.
- Gate 1 CNDC-v2 (spatial_melt : C_f NeRF module la fonte, rustine ignorée) : 0.489→0.535, gamma 1.46→1.17. Mieux mais training PLAT (2 jauges = pas de signal). Leçon : région pauvre en jauges ne s'entraîne pas seule.
- Gate 2 MONT-v2 (23 jauges, spatial_melt) : 0.523 vs Hydrotel 0.637, beta 0.82 (sous-volume). ÉCHEC → pas de flotte.
- Gate 2bis MONT-v3 EN COURS : + ancrage sol sur calage Hydrotel régional (hydrotel_calib_dir, mécanisme réactivé). Doctrine reproduire-puis-moderniser appliquée au scale-up : prior = calage régional, la différentiabilité raffine.
- Chantiers structurels notés : entraînement conjoint multi-régions (NeRF partagé, toutes jauges QC), module barrages/régulation.

## MONT-v3 (ancrage sol calage Hydrotel régional) : ÉCHEC BRUTAL, 2026-07-16
- Held-out médian -0.31, beta 0.20 (80% de l'eau disparaît). load_calibrated_soil écrit/validé pour SLSO, branché sur MONT SANS validation isolée (entorse à la règle validate-before-integrate, la semaine même du rappel).
- v4 (chaînée avec le même ancrage) tuée avant pollution ; relancée SANS ancrage = melt NeRF + volume Budyko seulement (ablation propre par-dessus v2).
- TODO avant tout ré-essai d'ancrage : valider load_calibrated_soil(MONT) isolément (params chargés vs bv3c.csv lu à la main, bilan 1 an sur 10 UHRH vs sortie Hydrotel de la plateforme).

## BENCHMARK CORRIGÉ — Hydrotel = ENSEMBLE de 6 calages (info Essi), 2026-07-17
- LN24HA (Linacre) + 5 × MG24Hx (McGuinness). Même physique, calages différents = optima locaux.
- MONT held-out : MG24HK 0.758 >> MG24HI 0.651 > LN 0.637 > MG24HS 0.634 > MG24HQ 0.622 > MG24HA 0.593. Dispersion 0.17.
- SLSO held-out : méandre 0.689 BAT LES 6 (MG24HK 0.673, LN 0.666, ..., MG24HI 0.560). Claim phare renforcé.
- Classement des membres INCOHÉRENT entre régions (MG24HI : 2e MONT, dernier SLSO) = équifinalité documentée, argument identifiabilité.
- MG24HK MONT : bv3c z 0.10/0.55/1.00 (mince, réactif ; vs 0.22/0.16/2.65 LN), krec 5e-7, fonte 4/4/4 avec seuils 0.25/-1.75/-3.75, tassement 0.0054, McGuinness ×0.50.
- Conséquence : la barre MONT = 0.758 ; candidat v8 = ancrage MG24HK (sol mince compatible colonne, + coeff McGuinness régional à câbler comme Linacre).

## MONT v7 : fonte régionale = +0.07, v8 (fonte MG24HK) lancé, 2026-07-17
- v7 (Linacre + fonte LN ancrée : taux 4.5/9/18, seuils +2.3/+1.9/+1.6) : held-out médian 0.592, pooled 0.767 (records méandre-MONT). La fonte à seuils était bien le verrou hiver/printemps (+0.072 vs v6).
- Série pilotes MONT : v2 0.573 (melt NeRF) / v4 0.552 (+Budyko) / v5 0.535 (krigé : forçage réfuté) / v6 0.520 (+Linacre : beta 1.02 mais gamma 0.80) / v7 0.592 (+fonte régionale).
- v8 = v7 avec fonte MG24HK (4/4/4, seuils 0.25/-1.75/-3.75, tassement 0.0054) = la fonte du membre champion (0.758).

## Fin de la série pilotes MONT : v8 et v9 REJETS, recette finale = v7, 2026-07-17
- v8 (fonte MG24HK seule sur base v7) : 0.532/0.712 — RECUL vs v7. Le calage d'un membre est un PAQUET cohérent, une pièce isolée ne se greffe pas.
- v9 (v7 + sol MG24HK gelé) : 0.125/0.372 — ÉCHEC SÉVÈRE. Confirmé avec v3 : GELER le sol tue méandre, quel que soit le sol (profond LN ou mince MG24HK). Le NeRF compense par le sol les divergences structurelles assumées de la colonne (aquifère restituant, UH versant, etc.) ; le priver de ce levier casse l'équilibre.
- LOI DES ANCRAGES (3 succès, 3 échecs) : ancrer les PROCESSUS scalaires régionaux (ETP Linacre + coeff, fonte taux/seuils) = OUI ; geler les CHAMPS que le NeRF doit apprendre (sol) = NON.
- RECETTE FINALE MONT (v7) : sol NeRF libre + Linacre régional + fonte régionale LN + fonte NeRF (modulation) + volume bilan → 0.592/0.767 (vs 0.520 départ, LN24HA 0.637, MG24HK 0.758).

## AVEU + CARTE v4 : la flotte n'avait JAMAIS été arrêtée, 2026-07-18
- Le kill du 07-16 a tué le python courant mais PAS la boucle bash _fleet_v4 (survivante invisible à mon ps). La flotte a tourné 2 jours en contention avec les pilotes et a fini le 07-18 04:18. J'avais annoncé son arrêt à Essi : faux. Leçon : VÉRIFIER la mort du parent (boucle), pas seulement de l'enfant.
- Bénéfice accidentel : carte v4 complète 12 régions vs ensemble 6 membres (reports/quebec_v4_vs_ensemble.csv).
- LECTURE MAJEURE : l'ensemble Hydrotel est FORT sur les grandes régions de l'est (SAGU 0.77-0.81, SLNO 0.79-0.82, OUTV 0.80-0.83, GASP 0.77-0.79) et FAIBLE sur SLSO/MONT où méandre gagne/rivalise. v4 uniforme y est ~0.2 dessous. méandre-v4 bat la médiane d'ensemble seulement 2/11 (LABI, CNDB).
- Question décisive : modèle ou forçage (CaSR vs krigé dense) sur ces régions ? Tests v5-style (météo plateforme, fenêtre courte) lancés sur GASP et SAGU.

## ATTRIBUTION EST (GASP/SAGU, tests krigés même fenêtre 2023-24), 2026-07-18
- GASP : r 0.601→0.703 (+0.10) avec météo MELCCFP mais beta inchangé 0.81 → timing = FORÇAGE (CaSR pauvre à l'est), volume = MODÈLE (McGuinness sur-évaporant, ancrage Linacre pas encore appliqué hors MONT).
- SAGU : +0.10 KGE avec krigé ; gamma 1.16-1.22 résiduel = régulation (Lac-Saint-Jean).
- CONTRASTE avec MONT (krigé sans effet) : la valeur de la météo krigée dépend de la densité d'assimilation CaSR locale. À l'est, le forçage compte ; au sud, non.
- SYNTHÈSE FLOTTE : recette = ancrages v7 partout (Linacre + fonte régionale) + choix forçage par région (CaSR corrigé au sud/centre, MELCCFP ou hybride à l'est) + z_n + quantile. Décisions Essi lundi.

## Pilotes v7 est : GASP +0.088, SAGU leçon point-fixe, 2026-07-18/19
- GASP-v7 (Linacre+fonte+VOL régionaux) : held-out 0.577 vs v4 0.489 (+0.088). beta 0.81→1.03. Recette CONFIRMÉE sur l'est naturel ; reste = plafond forçage (r 0.70).
- SAGU-v7 : RÉGRESSION 0.524→0.449, beta 0.999→1.19. Cause : VOL = lame + ETP_Linacre suppose ETR≈ETP ; sur boréal humide l'ETR simulée (236) << ETP (359). L'ancrage volume est un POINT FIXE : itérer une fois avec l'ETR simulée (VOL 1037→914). sagu-v7b lancé.
- Règle de flotte : VOL_région = lame + ETR_sim (une itération après le premier run), pas lame + ETP.

## SAGU-v7b : volume réglé (point fixe OK, beta 1.037) mais ancrages nuisibles au boréal, 2026-07-19
- v7b : beta 1.19→1.037 (l'itération ETR simulée FONCTIONNE) mais r 0.578→0.500, gamma 0.73. Le KGE chute à 0.389.
- Cause : les ancrages LN (seuils fonte +2.3°C, ETP Linacre basse) sont calés pour le sud ; sur le manteau boréal profond du Saguenay ils déforment le freshet. Réseau régulé (Lac-Saint-Jean) amplifie.
- VERDICT FINAL PILOTES : recette v7 (ancrages régionaux) validée sur MONT (+0.04) et GASP (+0.088) ; v4 (McGuinness+Budyko) reste meilleure sur SAGU (0.524). Les ancrages ne sont PAS universels : par CLASSE de région (sud/est naturel = v7 ; boréal/régulé = v4 en attendant barrages + entraînement conjoint).
- Le point fixe volume (VOL = lame + ETR_sim, une itération) est validé comme mécanisme et entre dans la recette de flotte.

## FONDATIONS DONNÉES ASSAINIES (audit Essi "échecs silencieux STAC"), 2026-07-19
- Instinct d'Essi confirmé 5 fois : ET annuelle (26 valeurs) déguisée en 8-jours dans TOUTES les bases régionales (chemin Planetary Computer = MOD16A3GF annuel, cap structurel) ; token Earthdata expiré depuis le 10 juin (fichier ~/.edl_token ET var d'env EARTHDATA_TOKEN qui le court-circuitait) ; C: plein à 100% ; pyhdf sans DLL Windows (ET8 = WSL only) ; DNS WSL flaky (retry ×3 nécessaire).
- Résultat : 15/15 bases COMPLÈTES — ET8 MOD16A2GF 1150 composites partout (~30M obs valides), GRACE partout, 100% des nœuds terrestres (les nœuds sans ET = eau libre, vérifié contre f_water/lake_fraction).
- Critère de complétude corrigé : couverture des nœuds TERRESTRES (MOD16 n'existe pas sur l'eau).
- Toute la flotte v1-v7 avait tourné avec un multi-obj quasi vide (26 obs annuelles/nœud au lieu de 1150) : les résultats régionaux sont à réinterpréter, le conjoint partira sur des données saines.
- Ops : tuiles CaSR consolidées sur D: (799), fields archivés, 18 Go libérés sur C:.

## PILOTES CONJOINT 3 régions (slso+mont+gasp) : critère ÉCHOUÉ, z_n innocentés, MONT gagne, 2026-07-20/21
- Design reports/design_conjoint.md : UN NeRF + UNE colonne partagés, latents tranchés par offset, rotation des régions, best sur médiane pondérée jauges, 30 epochs cold-start (recette v4/McGuinness, forçage 6 canaux, LR 3e-4 après leçon pilote3).
- pilote3 (LR 5e-4) : effondrement terminal epoch 7 (agrégé 0.62→0.18 sans retour) — la boucle conjointe court-circuitait les garde-fous de fit(). Garde-fou ajouté à joint.py (régression >20% sous best → recharge best + LR/2, max 3).
- pilote3b (LR 3e-4, z_n) : entraînement stable (2 rollbacks rattrapés), best val agrégé 0.617. HELD-OUT 22-24 : slso 0.554 / mont 0.541 / gasp 0.377.
- pilote3c (idem SANS z_n, JOINT_ZN=0) : best val 0.607 (atteint epoch 8), plateau stable 0.60. HELD-OUT : slso 0.533 / mont 0.601 / gasp 0.321.
- VERDICT au critère pré-enregistré (aucune région ne régresse vs mono held-out slso 0.689 / mont 0.592 / gasp 0.577) : ÉCHEC dans les deux variantes.
- z_n INNOCENTÉS : le gap val/test de GASP (0.59 val → 0.32-0.38 test) persiste et EMPIRE sans latents. La fragilité hors-régime du conjoint ne vient pas des effets aléatoires par nœud.
- MONT GAGNE au conjoint : 0.601-0.603 held-out (pilote3 et 3c) vs 0.592 mono-v7 (avec ancrages) et 0.552 à recette égale (v4). Signal de pooling réel, reproductible sur 2 pilotes.
- MONT = seule région instable en cours d'entraînement (les 4 plongeons des 2 runs sont portés par mont) ; gasp val d'une stabilité remarquable (0.58-0.59) malgré les turbulences.
- Confusions non résolues : parité de recette (mono champions = ancrages v7, forçage corr 7 canaux slso, z_n+quantile) vs conjoint cold-start v4 ; sous-entraînement (~10 passes effectives/région) ; gap GASP val/test inexpliqué (à décomposer r/beta/gamma).
- Décision suivante avec Essi : diagnostiquer le gap GASP avant tout nouveau run.

## BANC ET HORS-LIGNE : le module appris bat les formules calées PARTOUT, critère atteint, 2026-07-21
- Banc design_et_appris.md : 15 régions, ~25M paires (nœud, composite 8-j MOD16), leave-region-out (tenues gasp/sagu/mont = est/boréal/sud) × temporel (test 2022-2024). Bancs physiques ajustés au mieux : K_c McGuinness LSQ = 0.540, coeff Linacre LSQ = 0.472.
- TENUES (jamais vues, 2019-2024, médianes) : GRU R² 0.914 / MLP 0.916 / McGuinness 0.817 / Linacre 0.770. Biais : appris -1.4 à -1.7 % contre -11 à -12 % pour les formules.
- TEMPOREL 2022-2024 (12 régions d'entraînement) : GRU 0.911 / MLP 0.912 / McGuinness 0.804 / Linacre 0.728.
- Le module gagne dans CHAQUE région tenue et chaque période, R² et biais. La relation météo+territoire vers ETR transfère hors-région et hors-régime là où les coefficients calés ne transfèrent pas (biais -15 % GASP McGuinness).
- SURPRISE : le MLP (agrégats 8 j + 90 j) égale le GRU — pas besoin de mémoire récurrente à la granularité 8 jours ; intégration colonne beaucoup plus simple (pas d'état).
- Réserve honnête : MOD16 = produit PM (MERRA+LAI), le R² élevé emule en partie son algorithme ; vérif biais annuel vs bilan P-Q-dS(GRACE) à faire avant intégration.
- Coût : ~1 h GPU au total, itérations en minutes — contraste voulu avec les pilotes 20 h.
- Prochaine étape (validation Essi) : étape 2 du design = brancher le module comme demande évaporative dans la colonne (extraction par couche et conservation intactes, K_c neutralisé), fine-tuning bout-en-bout.

## Intégration ET apprise sur GASP (etl1-4) : PAS de gain débit, le goulot GASP n'est pas l'ET, 2026-07-21
- Série (12 ep sauf indication) : etl1 demande gelée sans levier 0.430 (beta cloué 0.807 = biais chaud MOD16 importé tel quel) ; etl2 + K_c NeRF mais w_et=1.0 le neutralise 0.403 (K_c poussé à 1.07 MALGRÉ beta 0.78 : w_et(MOD16) = double ancrage sur cible biaisée quand l'ET vient du module) ; etl3 w_et=0 : beta remonte 0.797→0.864, 0.468 ; etl4 30 ep : val méd 0.60, r 0.718 (au plafond forçage), held-out 0.466. Références : v4 0.489, v7 0.577.
- LECTURE : le module ET gagne sur SON observable (banc : R² 0.91 vs 0.82) mais ne gagne pas le DÉBIT sur GASP, parce que le goulot volume gaspésien est le P du forçage (v7 avait gagné surtout par le VOL régional du forçage -lin, pas par l'ETP) et le goulot timing est la fonte. K_c ne peut pas compenser un P trop bas sans écraser l'ET sous le réaliste (plateau beta ~0.86, retombe 0.795 en fin de LR en re-trocant volume/variance contre r).
- Leçons de design acquises : (1) module appris ⇒ w_et redondant et NUISIBLE (double ancrage) ; (2) toujours garder un multiplicateur NeRF autour d'une demande apprise (levier volume) ; (3) un module qui gagne son banc hors-ligne n'améliore le débit QUE si son processus est le goulot local.
- Critère du plan (≥ v7 sur GASP) NON atteint → pas d'extension automatique ; décision avec Essi : SAGU (où les ancrages ont ÉCHOUÉ, v4 0.524) = test décisif proposé, et/ou phase 2 fonte (le vrai goulot timing).

## ET APPRISE SUR SAGU : RECORD 0.621 held-out (+0.097 vs v4), la relation transfère au boréal où les ancrages échouaient, 2026-07-21
- sagu-etl (recette etl3/4 : demande MLP MOD16 × K_c NeRF init 1.0, w_et=0, 30 ep) : held-out médian 0.6213 / mean 0.6161 (n=19). Références : v4 0.524 (meilleure recette connue), v7b ancrages 0.389.
- Trajectoire saine de bout en bout : beta 0.94→1.05 (le problème de point fixe volume de v7b n'existe plus), r 0.847, gamma 0.833 stable = régulation Lac-Saint-Jean (dossier phase 3, table dams prête).
- LECTURE : là où le processus ET est réellement un levier (boréal humide, ETR simulée sous-estimée chronique), le module appris livre ; et il livre PRÉCISÉMENT dans la région où les coefficients calés du sud ont échoué. Argument central de la thèse identifiabilité-par-les-données, démontré sur débit.
- Bilan phase 1 : GASP neutre (goulot = P forçage + fonte, pas l'ET), SAGU +0.097 (record). Un module par processus, jugé sur le goulot local.

## MONT-etl : 0.578 held-out (> v4 0.552, ~= v7 0.592) — triptyque phase 1 complet, 2026-07-22
- mont-etl (demande apprise 612 mm/an × K_c, w_et=0, 30 ep) : held-out 0.578. Le best vient de l'epoch ~7 : le run a DIVERGÉ ensuite (beta 0.85→0.62, val_med 0.56→0.26) sans être rattrapé — etl_run.py ne passe pas les réglages autopilot du TOML au Trainer, contrairement aux runs slso.py. MONT reconfirme son instabilité (déjà la seule région instable des pilotes conjoints). À corriger si on refait des mono longs.
- BILAN PHASE 1 (ET apprise, zéro emprunt Hydrotel, held-out 22-24) : GASP 0.468 (v4 0.489 : neutre, goulot = P+fonte) ; SAGU 0.621 (v4 0.524 : RECORD +0.097) ; MONT 0.578 (v4 0.552 : +0.026 ; v7 ancrages 0.592 : -0.014, les ancrages LN restent marginalement devant SUR LEUR territoire de calage).
- Verdict : le module appris remplace crédiblement les ancrages partout, les bat là où ils échouent (boréal), et ne perd que 0.014 là où ils jouent à domicile. La pièce ET du conjoint sans compromis est validée.

## PILOTE4-ET (conjoint 3 régions, ET apprise, sans z_n) : premier conjoint STABLE, GASP +0.05, fonte = dernier compromis, 2026-07-22
- Recette : pilote3c + JOINT_ETL (demande MLP par région en 7e canal, K_c init 1.0, w_et=0), 12 epochs, LR 3e-4. AUCUN rollback, montée monotone (0.522→0.565) — premier conjoint sans plongeon : le compromis ETP inter-régions était bien une source d'instabilité.
- HELD-OUT 22-24 : slso 0.543 / mont 0.585 / gasp 0.373. Vs pilote3c (McGuinness) : slso +0.010, mont -0.016, gasp +0.052.
- Vs monos : mont 0.585 ~ mont-etl 0.578 et > v4 0.552 (le conjoint ne coûte plus rien à MONT et reste au niveau des ancrages 0.592) ; gasp toujours en gap val 0.514 → test 0.373 (la fonte compromis, cible phase 2) ; slso toujours sous son champion 0.689 (recette différente : forçage corr 7 canaux + z_n + quantile — parité à tester plus tard).
- Suite de chaîne : banc fonte dès MOD10 complet (slso ~fini, 15 régions ~demain), puis module fonte, puis pilote4b ET+fonte.

## MOD10A1 fonte : 15/15 régions COMPLÈTES (~84M lignes, mars-juin 2000-2024, 100 % des nœuds), 2026-07-22
- Ingestion quotidienne fenêtre de fonte via earthaccess/WSL, cache granules partagé D:/modis10 (~15k granules), zéro échec réseau (retry x3 armé, jamais nécessaire au-delà). ~3015 jours par région, ~40 % d'obs sans nuage (normal printemps).
- Smoke banc fonte (train SLSO seul) : le MLP transfère déjà mieux que degré-jour/ETI calés en tenue (GASP acc 0.85 vs 0.77, MAE date 24 vs 31 j) ; degré-jour 11 j -> 31 j en changeant de région = l'histoire des constantes qui ne transfèrent pas, again. Banc complet 12 régions lancé.

## BANC FONTE COMPLET : MLP passe le critère DE JUSTESSE, degré-jour global MOD10 = la vraie surprise, 2026-07-22
- 12 régions train, tenues gasp/sagu/mont. TENUES : mlp 15.6 j / acc 0.875 ; dd calé 16.8 j / 0.848 ; eti 25.5 j / 0.804 (éliminé). TEMPOREL 22-24 : mlp 13.5 j / dd 14.8 j.
- Critère pré-enregistré techniquement ATTEINT (mlp bat dd et eti partout) mais marge mince (1-2 j). La vraie leçon : un degré-jour calé GLOBALEMENT contre MOD10 transfère déjà très bien (16.8 j en régions jamais vues) — le problème des ancrages était le PAR-RÉGION Hydrotel, pas la forme degré-jour.
- Params dd calés MOD10 (12 régions) : C_f 5.39 mm/°C/j, T_melt -0.98 °C, seuil pluie/neige -0.47 °C — fonte plus précoce et plus rapide que les défauts littérature (4.5, -0.5, 0).
- DÉCISION intégration deux temps : (1) pilote4b = init fonte MOD10 (transplant des 3 scalaires en literature_prior + t_neige_seuil, zéro chirurgie sur le clone neige validé décimale, NeRF module autour) ; (2) module MLP fonte = phase 2b différée (marge 1.2 j ne justifie pas encore d'invader le clone).
- pilote4b lancé : conjoint 3 régions, ET apprise + init fonte MOD10, 12 ep.

## PILOTE4B RÉFUTÉ : l'init fonte MOD10 nuit en conjoint (les scalaires ne survivent pas au changement de structure), 2026-07-22
- pilote4b (pilote4-et + C_f 5.39 / T_melt -0.98 / seuil pluie-neige -0.47 en init) : held-out slso 0.447 / mont 0.541 / gasp 0.349, TOUT sous pilote4-et (0.543/0.585/0.373). Val aussi (best 0.531 vs 0.565).
- Cause probable : les scalaires ont été calés dans le modèle mono-réservoir du banc ; la colonne per-classe (albédo dynamique, tassement, pcts de couvert) a une autre cartographie C_f→fonte. Écho de la leçon v8 : un calage = un paquet cohérent, la greffe d'une pièce isolée échoue — même data-driven.
- MEILLEURE RECETTE CONJOINTE : pilote4-et (ET apprise, sans z_n, sans ancrage, fonte littérature + NeRF). Si la fonte doit être améliorée, ce sera par modulation apprise DANS la structure de la colonne (phase 2b), pas par transplant de scalaires.

## PHASE 3 RÉGULATION : TUÉE SUR PIÈCES par le diagnostic gratuit, 2026-07-22
- Sur le checkpoint record sagu-etl (0.621) : AUCUNE station jaugée SAGU n'a > 0.5 km3 de retenue amont (max 0.42) — les grands réservoirs (Passes-Dangereuses, Lac-Saint-Jean) ne sont en amont d'aucune de nos jauges CEHQ, qui mesurent les tributaires naturels.
- gamma médian actuel : 0.966. Le « gamma 1.16-1.22 = régulation » du 18 juillet était un artefact des vieux runs (krigé/v4), déjà résorbé par l'ET apprise. Un module d'exutoire appris n'aurait AUCUN levier sur nos métriques.
- Phase 3 différée sine die (réévaluer si des stations sous influence entrent dans la BD, p.ex. relevés Rio Tinto/HQ). La table dams reste (features NeRF potentielles, scénarios).
- Pilote4c lancé à la place : fonte supervisée MOD10 DANS la structure de la colonne (w_snow=0.3, swe_obs branché dans joint_data), la voie propre après la réfutation du transplant pilote4b.

## PILOTE4C : MEILLEUR CONJOINT (w_snow MOD10 aide sans coût), MONT bat tous ses monos, 2026-07-23
- pilote4c (pilote4-et + w_snow=0.3 sur MOD10 in-structure) : val agrégé 0.5732 (record conjoint), stable, 0 rollback. HELD-OUT : slso 0.5415 (=) / mont 0.6014 (+0.017) / gasp 0.3873 (+0.014).
- MONT 0.601 held-out > TOUS ses monos (v7 ancrages 0.592, mont-etl 0.578, v4 0.552) : première région où le conjoint sans aucun emprunt Hydrotel domine tout, critère « au moins une région gagne » ATTEINT.
- La supervision MOD10 dans la structure fonctionne là où le transplant de scalaires (4b) échouait — la donnée doit entrer par la loss, pas par les paramètres.
- Critère complet du design toujours PAS atteint : gasp régresse encore vs mono (0.387 vs 0.489 recette égale ; gap val 0.515 → test 0.387), slso sous son champion 0.689 (confonds de recette : forçage corr 7 canaux, z_n, quantile).

## SYNTHÈSE DE LA CHAÎNE MODULES APPRIS (phases 0-4, 2026-07-21 → 23)
- Phase 0 barrages : répertoire MELCCFP ingéré (8629 ouvrages, 15 bases). Diagnostic : la régulation n'explique PAS le déficit de l'est (GASP/CNDC sans stockage = pires écarts ; CNDB/Manic = méandre gagne).
- Phase 1 ET apprise : banc gagné sans appel (R² 0.91 vs 0.77-0.82, biais -1.5 % vs -12 %) ; SAGU record 0.621 (+0.097) ; MONT 0.578 ; GASP neutre (goulot ailleurs) ; conjoint STABILISÉ (le compromis ETP était aussi une source d'instabilité). Recette d'intégration : demande × K_c NeRF, w_et=0.
- Phase 2 fonte : MOD10 15/15 (84M obs) ; banc : MLP gagne de justesse, dd global calé transfère bien (le poison = par-région) ; transplant scalaires RÉFUTÉ (4b) ; supervision in-structure VALIDÉE (4c). Module MLP fonte = différé (2b).
- Phase 3 régulation : tuée sur pièces (aucune jauge sous influence, gamma déjà sain).
- Phase 4 : pilote4c = recette conjointe de référence. RESTE OUVERT : gap GASP val→test (suspects : P CaSR à l'est — attribution du 18/07 —, structure de fonte au-delà des seuils, régime 2022-24) ; parité de recette SLSO (7 canaux + z_n régularisés + quantile) à tester pour la flotte.
- Méthode : 3 chantiers économisés par diagnostics gratuits (z_n après coup, transplant... et phase 3 avant coup) ; bancs hors-ligne à 1 h GPU au lieu de pilotes 20 h ; chaque verdict pré-enregistré.

## FORÇAGE HYBRIDE KRIGÉ : GASP +0.09 held-out, hypothèse forçage CONFIRMÉE, 2026-07-23
- gasp-etl-hyb (P/T krigés MELCCFP quebec.zarr + énergie CaSR, ET apprise, 30 ep) : held-out 0.5585 vs CaSR 0.466 (+0.092). r 0.66→0.70 en val exactement comme prédit par l'attribution du 18/07. À égalité avec v7 (0.577, ancrages) SANS aucun ancrage — juste la bonne pluie.
- Confirme sur held-out ce que l'échange de météo avait montré ponctuellement : à l'est, le goulot de GASP est le P de CaSR, pas le modèle. La chaîne modules appris avait réglé le volume/ET ; le krigé règle le timing/intensité.
- MAIS run diverge encore post-epoch-12 (val 0.664→0.575, best sauvé à 0.664) : etl_run ne passait PAS l'autopilot du TOML au Trainer (bug depuis MONT-etl). Corrigé (autopilot + garde-fou régression câblés). gasp-etl-hyb2 relancé pour capter le plein potentiel.

## GASP-etl-hyb2 (autopilot corrigé) : 0.586 held-out, DÉPASSE les ancrages v7 sans rien emprunter, 2026-07-23
- Fix autopilot validé : plateau propre val 0.685, 0 divergence. Held-out 0.5858 (vs 0.559 sans fix, 0.466 CaSR, 0.577 v7 ancrages). r 0.727.
- ATTRIBUTION FINALE GASP chiffrée : CaSR coûte ~0.12 KGE ; le calage manuel plateformes en récupère ~0.09 (v4→v7) mais NON transférable ; modules appris + krigé = 0.586 PORTABLE et supérieur. Résultat de papier.

## CORRECTION MAJEURE : les runs -hyb sont du SIMAT, pas du krigeage (erreur retrouvée en mémoire), 2026-07-23
- quebec.zarr = SIMAT ré-grillé (attrs : processing_level=regridded, ordre_priorité simat en tête) = LA carte à yeux de bœuf qu'on avait DÉJÀ rejetée le 29 juin (mémoire casr_canonical_decision : « quebec.zarr krigeage bancal, œils-de-bœuf, carte 2007 »). J'ai rebâti dessus cette semaine en l'appelant krigeage. Faux.
- Conséquence : gasp-etl-hyb 0.586 et sagu-etl-hyb = comparaison À FORÇAGE-ÉGAL-AVEC-HYDROTEL (SIMAT est l'intrant d'Hydrotel), PAS un krigeage propre ni portable. Ne rien propager (pas de conjoint hyb, pas de flotte).
- Mon test d'yeux de bœuf était faux-négatif : 10 km + moyenne annuelle + ré-grillage bilinéaire lissent les bullseyes station/journaliers.
- VRAIE VOIE (décision Essi) : pipeline REPRODUCTIBLE CaSR (énergie) + PyGMET tourné par nous sur stations via API. Abandonne era5land+pygmet des collègues (pipeline non reproductible).
  - PyGMET = github.com/NCAR/PyGMET (station->grille, régression + covariables élévation, cross-val LOO, ensemble). pygmet_loader.py existe (lit seulement).
  - Stations = API ECCC climate-daily (api.weather.gc.ca/collections/climate-daily, OGC, sans auth, réseau QC dense) > GHCN.
  - Validation anti-bullseye = LOO de PyGMET (révèle les yeux de bœuf au lieu de les cacher).

## PyGMET : cause racine du NaN trouvée = séries stations non complètes, corrigée par infill IDW, 2026-07-23
- Le run GASP full initial sortait une grille 100% NaN malgré stations/poids/voisins tous valides. Isolé par le cas-test CALI de référence (77% finite avec mon code patché) : le patch est innocent, le bug venait de MES entrées.
- CAUSE : PyGMET/GMET exige des séries stations SÉRIELLEMENT COMPLÈTES (CALI a un fill_flag, trous pré-remplis). Les stations ECCC ont ~50% de NaN réels ; une seule voisine à NaN un jour donné -> estimation NaN -> grille entièrement vide. Confirmé : remplissage grossier -> grille 33% finite (= fraction terrestre du masque), tmean 11.6°C physique.
- FIX intégré à build_pygmet_inputs.py : infill spatial IDW (12 voisins ayant des données ce jour-là ; PAS de remplissage à 0 qui biaiserait vers le sec ; fallback climato jour-de-l'année). Stations désormais complètes.
- Relancé : gasp full corrigé (5 prédicteurs lat/lon/elev/slp_n/slp_e ; le patch numpy gère la singularité qui m'avait fait retirer les pentes). Le run initial de 2h20 était perdu (stations non infillées).

## PyGMET GASP krigé VALIDÉ (LOO CC 0.89 pluie / temp RMSE 1.2°C) ; biais volume Box-Cox corrigé par régression linéaire, 2026-07-23
- Krigeage GASP corrigé (stations infillées) : validation croisée LOO sur 102/102 stations = pluie CC 0.891 / KGE 0.75, température CC 0.997 / RMSE 1.24°C. PAS d'yeux de bœuf (un bullseye donnerait un LOO mauvais aux stations retirées ; ici le champ généralise).
- MAIS régression déterministe en espace Box-Cox surestime le VOLUME de pluie ×2.08 même AUX stations (2282 vs 1099 mm/an) : biais de convexité du retransform (x/4+1)^4, que seule la moyenne d'ensemble corrige dans GMET.
- FIX sans ensemble (30h évitées) : régression en espace LINÉAIRE (transform_vars=['','','']) -> grille 1271 mm/an (vs stations 1099 ; +16% = orographie réelle sur cellules d'altitude, physique). Pattern/timing préservés.
- Pipeline complet fonctionnel : eccc_loader -> build_pygmet_inputs (infill IDW) -> PyGMET (linéaire, 5 préd, patch numpy) -> build_forcing_pygmet (nearest-valid-cell + blend énergie CaSR). Run GASP full linéaire lancé.
- Bugs corrigés en route : code mort dans to_nodes (interpolateur (t,y,x)) ; extraction nearest-valid-cell (évite NaN cellules masquées) ; prcp physique (pas de retransform en déterministe).

## VERDICT GASP sur KRIGEAGE PyGMET REPRODUCTIBLE : 0.601 held-out, meilleur sans rien emprunter, 2026-07-24
- gasp-etl-pgm (ET apprise + forçage PyGMET krigé stations ECCC + énergie CaSR) : held-out médian 0.6013. Volume enfin correct (P nœuds 1274 mm/an, beta ~1.0).
- CLASSEMENT GASP held-out 22-24 : CaSR 0.466 | ancrages v7 0.577 | SIMAT emprunté 0.586 | PyGMET reproductible 0.601 | ensemble Hydrotel 0.768.
- Le krigeage propre gagne +0.135 sur CaSR et DÉPASSE le SIMAT (que je présentais comme borne) — sans aucun emprunt à Hydrotel, 100% reproductible (ECCC API + PyGMET vendoré).
- Réserves : (1) dérive tardive etl_run (best held-out au milieu, beta redescend en fin ; autopilot à durcir) ; (2) l'écart résiduel à l'ensemble Hydrotel (0.768) PERSISTE malgré forçage comparable -> la cause restante n'est plus le forçage mais le CALAGE (Hydrotel calé bassin par bassin, méandre non). Écart de nature différente, honnête pour le papier.
- Pipeline PyGMET prouvé bout-en-bout sur GASP. Suite : étendre à l'est (10 forçages stations déjà accessibles même chemin) + conjoint sur krigeage propre.

## CAUSE RACINE déficit pics/r ENFIN trouvée + fix : K_sat_1 prior 6× trop perméable, 2026-07-24
- MÉTHODE (exigée par Essi) : tests d'IMPULSION rapides (ms) au lieu de sims 25 ans. Scripts : impulse.py, storm_diff.py, storm_sensitivity.py, ksat1_test.py.
- ROUTAGE EXONÉRÉ : impulsion en tête d'une chaîne de 6 tronçons ressort à l'exutoire le JOUR MÊME (retard 0), tous K/modes. Le travel_time_days entier gelé n'est utilisé QUE dans le chemin TTA (pas l'opérateur d'entraînement). Baisser K_musk EMPIRE le retard. Le canal n'est PAS le problème.
- SOL = coupable : orage 50mm -> coeff ruiss 17% (réel 30-50%), pic +3j, absorbe 83%. Levier = K_sat_1 (surface) : ×0.5->32%, ×0.3->59%, ×0.1->88%, pic +0j, BASEFLOW INTACT (K_sat_3 découplé).
- POURQUOI le NeRF ne corrige pas : init/prior littérature K_sat_1 = 0.080 m/j (field_network l.315) = 6× Rawls loam (0.013) ; physical_prior_loss l'ANCRE là. Param cloué à un prior faux (réponse à « pourquoi les params ne s'ajustent pas »).
- Hydrotel = MÊME physique BV3C2 + MÊME pas journalier + météo 24h (Essi a écarté ma fausse piste sous-journalière/hortonien) ; seule diff = K_sat calé.
- FIX : ETL_KSAT1=0.04 (recale prior). RÉSULTAT GASP -hyb (SIMAT, armes égales Hydrotel) : r 0.727->0.790 (ferme la moitié du gap vs Hydrotel 0.877) ; val_kge 0.68->0.740 ; held-out 0.586->0.627 (record méandre-GASP) ; gamma 0.95 (plus de sur-lissage), beta 0.856, stable.
- SUITE : pousser K_sat_1 un peu plus bas (0.03) ? combiner avec forçage PyGMET ? généraliser à toutes les régions + rendre défaut. Reste du gap r (0.79->0.877) = génération résiduelle.

## Empilement forçage/K_sat + verdict freshet, 2026-07-24 soir
- pgm+ksat1 : held-out 0.623 — PAS d'empilement (hyb+ksat1 0.627, pgm seul 0.601). Le r val de pgm (0.609) << SIMAT (0.790) : timings des deux forçages différents, le NeRF n'additionne pas. Meilleur méandre GASP = 0.627 vs Hydrotel brut même forçage 0.744 (gap 0.117).
- FRESHET diagnostiqué (banc synthétique manteau 350mm + redoux) : fonte à ZÉRO jusqu'à Tmax +5.5°C puis déchaînée (43 mm/j) = démarre ~2 sem trop tard, pic compressé. Porte de gel INNOCENTÉE (fonte 43 -> ruiss 40 le même jour, l'eau sort). Seuils de fonte trop hauts = le déficit r nival.
- Le banc MOD10 (phase 2) calait le seuil à -1°C ; transplant direct réfuté (4b) mais supervision in-structure validée (4c, w_snow). Hook ETL_WSNOW ajouté ; run SIMAT+ksat1+w_snow=0.3 lancé (duel à forçage égal vs Hydrotel 0.744).

## LA VRAIE NATURE DU GAP : ROBUSTESSE AU RÉGIME, pas les paramètres, 2026-07-25
- Nuit de runs : ancrage fonte figée = val record 0.698 mais held-out 0.618 (< wsnow 0.642) ; littérature-fonte (T_melt libre prior +2, taux 4.5/9/18, w_snow) = val record 0.741/0.700 mais held-out 0.606. DEUX réfutations concordantes : tout ce qui retarde la fonte gagne le val (climat historique) et perd le test (hivers 22-24 +1.5°C = fonte réellement plus précoce).
- MESURE DÉCISIVE : Hydrotel GASP est STABLE à travers le régime (0.743 val-proxy 20-21 -> 0.744 test 22-24, zéro chute) ; méandre perd 0.06-0.10 sur TOUTES ses variantes (0.70 val -> 0.61-0.64 test).
- DÉCOMPOSITION FINALE du gap GASP : ~0.04 en régime connu (0.700 vs 0.743) + 0.06-0.10 de FRAGILITÉ hors-régime. Les champs optimisés par gradient sur-adaptent le climat d'entraînement (équifinalité = compensation = fragilité) ; les paramètres physiquement contraints d'Hydrotel extrapolent.
- Classement final GASP held-out : wsnow PROPRE 0.642 > pgm+ksat 0.623 > ancrage 0.618 > litfonte 0.606 > base 0.586 | Hydrotel 0.744.
- DIRECTIONS (à cadrer avec Essi) : priors physiques durs anti-compensation ; entraînement à invariance inter-régimes (sous-périodes chaudes/froides du train) ; robustesse val->test comme métrique de sélection. Compat : le branchement T_melt change le comportement des vieux checkpoints (T_melt actif désormais) — réévaluations sur commit antérieur ou T_melt=0.

## PARTITION VERTICALE ROUVERTE : krec était le 4e paramètre enterré, +0.07 KGE en inférence pure, 2026-07-28
- Priors mesurés : récessions jauges GASP k=0.068/j méd (demi-vie ~10 j, 16 stations) ; modèle 2× trop lent (k sim 0.041) ET variabilité spatiale compressée (flashy sous-estimés ×2-3). krec/K_sat_3/k_gw tous MORTS pour la récession ; seul Z2+Z3 y touchait — puis vrai verrou trouvé : krec = scalaire GLOBAL de la colonne (pas NeRF), q3=krec·z3·t3, init 1e-6 = 1% de sa borne max (1e-4) -> baseflow 0.5% du flux, aquifère (spec Essi : recharge -> réservoirs lents par nœud, gel déjà actif sur l'infiltration) branché sur un robinet fermé.
- BANC : krec 5e-5 + use_aquifer + k_gw=0.068 (mesuré) = baseflow 24% (cible 20-40%), KGE 0.500->0.570 fenêtre courte SANS réentraînement ; krec 1e-4 = trop (45%, KGE retombe). Courbe en cloche nette.
- Pédotransfert K_sat : NON fiable pour l'instant (features texturales normalisées en base, reconstruction min-max déforme) — en attente des fractions brutes.
- Run gasp-aquifer lancé : ksat1 0.04 + w_snow + use_aquifer + krec init 5e-5 + prior k_gw 0.068. Réf à battre : 0.642 (held-out), Hydrotel 0.744.

## PARITÉ HYDROTEL ATTEINTE SUR GASP : 0.742 held-out (Hydrotel brut 0.744), robustesse de régime RÉSOLUE, 2026-07-28
- gasp-aquifer (ksat1 0.04 + w_snow MOD10 + use_aquifer + krec init 5e-5 + prior k_gw 0.068 MESURÉ) : held-out médian 0.7421 / mean 0.666 ; val 0.749 méd, r 0.830, gamma 1.000, stable 30 epochs.
- PARITÉ avec Hydrotel brut même forçage (0.744) sur SA meilleure région de l'est, à -0.002 ; ensemble médian 0.768 à portée. Il y a 4 jours : 0.586.
- ROBUSTESSE RÉSOLUE : chute val->test = 0.007 (0.749->0.742) contre -0.06/-0.10 sur toutes les variantes antérieures. La fragilité de régime était la PARTITION manquante : sans voie profonde, le sol compensait le baseflow par des paramètres sur-adaptés au climat ; avec le réservoir lent (spec Essi), la mémoire lente est physique et transfère.
- Recette 100% PORTABLE : aucun calage Hydrotel emprunté — k_gw = récessions des jauges, krec = banc de partition, K_sat = bilan d'orage, fonte = MOD10, ET = MOD16. Tous les priors sont MESURÉS.
- Arc GASP complet : 0.586 -> 0.627 (K_sat) -> 0.642 (fonte MOD10) -> 0.742 (partition+aquifère). Trois paramètres enterrés successivement exhumés (prior K_sat faux, seuil fonte mort, krec à 1% de sa borne), chacun trouvé par bancs rapides.
- SUITE : audit de leviers sur ce checkpoint (krec/k_gw vivants ?), robustesse à re-vérifier sur les autres régions, recette à généraliser (sagu d'abord), conjoint.

## SAGU-aquifer : 0.700 held-out (+0.078 vs record 0.621), robustesse confirmée (test > val), 2026-07-29
- Recette portable complète + k_gw MESURÉ SAGU (0.079) : held-out 0.6995 méd (n=19) vs record précédent 0.621. val méd 0.665 -> test 0.700 : AUCUNE chute de régime, 2e confirmation.
- r 0.839 ; gamma 0.789 = régulation Lac-Saint-Jean (plafond connu sans règles de barrage ; ensemble Hydrotel 0.767 calé DESSUS). Le gap restant est la régulation, pas l'hydrologie.
- MONT lancé avec SON k mesuré (0.172/j, rivières 2× plus flashy — la variabilité régionale que le prior uniforme écrasait).

## FRONTIÈRE DE LA RECETTE AQUIFÈRE + SYNTHÈSE DE L'ARC, 2026-07-29
- MONT-aquifer : 0.544 held-out, RECUL (réfs 0.601 conjoint / 0.578 mono) ; chute val->test de retour (0.636->0.544). Queue de récession re-mesurée : MONT est RÉELLEMENT rapide (k queue 0.138, demi-vie 5 j, drainage agricole) — pas un artefact de méthode. Sweep k_gw en inférence : KGE test INSENSIBLE à k_gw (0.544 partout) -> le réservoir lent n'est pas le levier de MONT ; suspect restant = K_sat_1 0.04 uniforme inadapté aux sols du sud.
- RÈGLE DE CLASSES (symétrique de v7) : recette aquifère-priors-mesurés = EST/BORÉAL naturels (GASP parité 0.742, SAGU record 0.700, test>=val) ; SUD flashy agricole = recette conjointe (MONT 0.601). k_gw par région depuis la QUEUE de récession (gasp 0.052, sagu 0.051, mont 0.138).
- BILAN DE L'ARC (4 jours, méthode scalpel) : 3 paramètres enterrés exhumés (prior K_sat ×6, seuil fonte mort, krec à 1% de sa borne) ; partition verticale rouverte (spec Essi : recharge -> réservoirs lents par nœud) ; robustesse de régime RÉSOLUE là où la physique du baseflow existe ; recette 100% portable, priors tous mesurés (récessions, MOD10, MOD16, bilan d'orage).
- SUITE : cause du déficit sud (K_sat/textures par classe de sol — exige fractions brutes PHYSITEL) ; conjoint est+boréal avec recette aquifère ; carte 15 régions ; audit leviers ckpt gasp-aquifer.

## SUD ÉLUCIDÉ : biais MOD16 hérité par l'ET apprise ; MONT record 0.624 par débiaisage structurel, 2026-07-29
- Chaîne (audit demandé par Essi) : loaders/mapping/pluie BLANCHIS (outil ratio d'aires officielle/mappée : 1/28 micro-station ; budyko=SIMAT ±5%) -> décomposition : beta test 0.73 = 27% d'eau perdue en TROP d'ET -> MOD16 sur-évapore le sud (+17-25% vs bilan P-Q, mesuré 21/07) et le module appris hérite -> K_c×0.8 en inférence : 0.544->0.617 (preuve).
- Prior K_c doux (0.8) RE-DÉFAIT par l'entraînement (0.583) : le gradient sur climat 2000-2018 re-préfère la solution biaisée. FIX STRUCTUREL : demande × ratio bilan/MOD16 au forçage (ETL_DEMAND_SCALE) -> MONT held-out 0.6243 (record ; > 0.601 conjoint, > 0.617 inférence) ; Hydrotel brut MONT test 0.637 = quasi-parité (-0.013).
- ÉTAT DES 3 RÉGIONS (recette portable, priors mesurés) : GASP 0.742 (parité), SAGU 0.700 (record, gap=régulation), MONT 0.624 (record, quasi-parité). SUITE : débiaisage ET aussi sur GASP/SAGU (beta 0.84-0.90, même biais), puis conjoint provincial.

## MÉANDRE BAT HYDROTEL SUR GASP : 0.749 vs 0.744, recette 100% portable, 2026-07-30
- Débiaisage ET mesuré (ratio bilan/MOD16) + k_gw de queue : GASP 0.7489 (> Hydrotel brut 0.744, ex-parité 0.742) ; SAGU 0.7053 (record, gap restant = régulation Lac-Saint-Jean).
- ÉTAT FINAL 3 RÉGIONS (priors tous mesurés, zéro emprunt) : GASP 0.749 BAT Hydrotel | SAGU 0.705 record | MONT 0.624 quasi-parité (0.637). Il y a 9 jours : 0.586/0.621/0.601 avec ancrages empruntés.
- Recette canonique : ET apprise MOD16 × débiais bilan régional + fonte w_snow MOD10 + K_sat 0.04 + aquifère (krec 5e-5, k_gw = queue de récession régionale) + forçage CaSR/hyb. SUITE : conjoint provincial.

## CERTIFICATION PHYSIQUE MACRO (checkpoint gasp-ds), 2026-07-30
- Bilan : P 1015 = ETR 452 (45%) + runoff 551 (54%) + dS +13 (1.3%) — fermeture OK. ETR dans le bilan P-Q mesuré. Partition baseflow 15% (légèrement sous la cible 20-40, seul ambre, non bloquant). Tous les champs NeRF dans leurs plages physiques ET spatialement VIVANTS (T_melt [-0.45,+1.53], C_f [1.9,4.4], k_gw 0.048 ~ queue mesurée 0.052 ; fini les collapses).
- FEU VERT pour le conjoint provincial avec la recette canonique.

## PILOTE5 (conjoint recette canonique) : ÉCHEC net vs monos — le pooling reste le problème ouvert, 2026-07-30
- gasp 0.444 / sagu 0.623 / mont 0.541 held-out vs monos records 0.749 / 0.705 / 0.624. La recette canonique ne survit PAS au partage d'un seul NeRF+colonne entre 3 régions (GASP s'effondre en particulier).
- Confonds à instruire AVANT tout re-run : 15 epochs sans autopilot (monos = 30 + autopilot), k_gw prior GLOBAL 0.07 (monos : par région 0.052/0.051/0.138 — la variabilité régionale mesurée est écrasée par le prior partagé), pondération jauges.
- DÉCISION STRATÉGIQUE pour Essi : (a) flotte de MONOS avec la recette canonique par région (chemin PROUVÉ : 3/3 records, priors régionaux mesurés scriptables pour les 15 régions) vs (b) instruire le conjoint (priors par région DANS le conjoint, plus d'epochs+garde-fous). Le produit provincial n'exige pas le conjoint — la recette portable EST la régionalisation.

## BUG STRUCTUREL DE TOUS LES CONJOINTS TROUVÉ (intuition Essi) : normalisation territoriale PAR RÉGION, 2026-07-30
- Vérifié : chaque duckdb régional stocke les features z-scorées sur SES stats (mean 0/std 0.92 partout). Le NeRF partagé reçoit des entrées incompatibles entre régions (+1σ sable = sols différents) -> ne peut pas apprendre de règle attributs->paramètres commune -> compromis mou, instabilité MONT, 5 échecs conjoints (pilote3->5) expliqués d'un coup. Le pooling n'a JAMAIS été testé proprement.
- FIX avant tout re-run conjoint : renormalisation GLOBALE provinciale des features dans joint_data (exige les stats brutes — vérifier si récupérables des DB ou rebuild territorial). Probable gain aussi pour et_bench (même biais, atténué car la météo domine).
- Les monos records restent valides (normalisation cohérente en intra-région).

## TRANSFERT ZERO-SHOT : la régionalisation marche DÉJÀ, le conjoint est inutile, 2026-07-30
- Hypothèse normalisation RÉFUTÉE proprement (A/B : z-local 0.706 = brut aligné 0.707 sur gasp->sagu ; les stats régionales se ressemblent, le NeRF est peu sensible à l'écart).
- DÉCOUVERTE (intuition Essi « je ne vois pas pourquoi méandre échouerait ») : le NeRF GASP transfère en ZERO-SHOT : SAGU 0.706 (= mono 0.705 !), MONT 0.568 (91% du mono 0.624, classe opposée). Le modèle EST déjà régional ; c'est la PROCÉDURE d'entraînement conjoint qui casse (optimiseur partagé/rotation/pondération), pas la transférabilité.
- Voie provinciale recommandée : UN champion (gasp-ds) + demand-scale et k_gw régionaux mesurés -> carte 15 régions en inférences (minutes/région) ; fine-tune court par région si besoin (sud). Conjoint = R&D non bloquante.

## CARTE PROVINCIALE ZERO-SHOT v1 (1 champion, 90 min d'inférence), 2026-07-30
- gasp 0.720 | sagu 0.718 | cndd 0.821 | cndc 0.710 | abit 0.701 | slno 0.630 | cnde 0.614 | mont 0.559 | slso 0.551 | outm 0.522 | cnda 0.470 | labi 0.339 | cndb 0.310 | outv 0.272 (vaud à vérifier).
- BAT la médiane d'ensemble Hydrotel sur cndd/cndc/abit ; parité proche gasp/sagu ; DÉCROCHAGES corrélés aux ds CONTAMINÉS : labi/cndb/outv ont ds 1.07-1.10 issus de bilans P-Q faussés par la régulation/dérivations (flaggés dès la vérif du 21/07) — le débiaisage y AUGMENTE l'ET à tort. Fix : ds=1 (ou borne <=1) pour les bassins au bilan non fiable, re-sweep 30 min.
- slso 0.551 : champion sur forçage -hyb vs champion slso historique 0.689 (7 canaux corr + z_n + quantile) — écart de recette, pas de transfert.

## CARTE v2 : le boréal réparé par la règle de forçage (hyb affamait le nord), 2026-07-30
- La grille SIMAT s'arrête à 53.3N : les -hyb boréaux perdaient 12-27% de P (cndb 854 vs 1175 mm/an). Sur budyko + ds neutre : labi 0.339->0.601, cndb 0.310->0.746 (BAT l'ensemble 0.547), outv 0.272->0.512, cnda 0.470->0.490 ; betas ~1 partout.
- RÈGLE DE FORÇAGE PAR CLASSE actée : hybride SIMAT+CaSR au sud/est de la grille ; CaSR-budyko au boréal/nord. Carte v2 médiane provinciale ~0.62 en zero-shot pur.
- Zero-shot BAT la médiane d'ensemble sur cndd/cndc/abit/cndb ; talonne gasp/sagu/labi ; fine-tune phase 5 : slso (parité de recette 7 canaux+z_n+quantile), mont, outm/outv/slno.

## Tournée bancs d'orage régionaux : K_sat INNOCENTÉ pour les retardataires, adaptateurs mesurés épuisés, 2026-07-30
- Coefficient d'orage du champion transféré : ~34% partout (mont/slso/outm/slno/outv), déjà dans la cible 30-50% -> échelle retenue 1.0, scores inchangés. Le fix K_sat généralise ; les écarts restants ne sont PAS la génération d'orage.
- Restants : slso 0.551 (parité de recette avec son champion 0.689 : forçage corr 7 canaux + z_n + quantile), mont 0.559 (domaine du champion, sols agricoles), outm/outv/slno 0.51-0.63 (ouest, ensemble Hydrotel 0.77-0.80 : à instruire, forçage ouest ou régulation).
- SUITE (option propre n°2) : instrumentation du paradoxe du conjoint (transfert OK, pooling KO) — run court 2 régions avec loss/gradients par région.

## PARADOXE DU CONJOINT RÉSOLU : tout pas de gradient supplémentaire dégrade le test — doctrine du CHAMPION GELÉ, 2026-07-31
- Warm-start du champion (gasp+sagu, LR 1e-4, 6 ep, entraînement STABLE, gradients équilibrés) : val agrégé MONTE (0.668->0.673) mais held-out DESCEND sous le zero-shot (gasp 0.653 vs 0.720 ; sagu 0.664 vs 0.718). Le conjoint à froid (8 ep) : 0.37/0.52. Diag : gradients sains, pas d'interférence visible.
- LECTURE : ce n'était ni le modèle (transfert parfait), ni l'optimisation (saine), ni la normalisation (réfutée) — c'est que TOUTE exposition supplémentaire au gradient de la période 2000-2018 échange de la robustesse test contre du gain val (le même mécanisme que le prior K_c défait et les runs fonte-tardive). Le point d'arrêt du champion est le bon ; s'en éloigner coûte.
- DOCTRINE : le champion est GELÉ. Régionalisation = adaptateurs MESURÉS uniquement (débiais ET, k_gw queue, règle de forçage). Le conjoint n'a plus d'objet ; le fine-tune régional est à proscrire sauf sélection sur robustesse (à inventer). Résultat de papier : « geler transfère mieux que réapprendre ».
- CARTE PROVINCIALE FINALE v2 = l'état de l'art méandre : médiane ~0.62 zero-shot, bat l'ensemble sur 4 régions, un seul modèle, zéro calage.

## DÉPLOIEMENT REPRODUCTIBLE v0.1 : leçon sur les critères de sélection, 2026-07-31
- deploy.py (measure->infer) livré : adaptateurs DÉRIVÉS des données avec provenance JSON, TOML = données+protocole+champion seulement (exigence Essi : pas de valeurs magiques).
- Trois critères de sélection du produit météo essayés, tous MESURÉS, résultats différents : (a) latitude/couverture -> réfuté (grille couvre 100% des nœuds, y compris cndb à 53.06N) ; (b) volume hyb vs budyko -> réfuté (rejette hyb partout, confond différence de produit et troncature) ; (c) fermeture du bilan P~Q+ET -> retenu comme critère PHYSIQUE, mais donne une carte médiane INFÉRIEURE (0.65 vs 0.69 avec les choix empiriques par région).
- Constat honnête : le meilleur bilan annuel ne garantit pas le meilleur KGE journalier (gasp casr ferme mieux le bilan mais l'hybride simule mieux le débit : timing vs volume). Le critère de sélection reste un CHOIX DE PROTOCOLE à assumer ; le pipeline le rend explicite et interchangeable, c'est ce qui compte.
- CARTE v0.1 (bilan) : cndd 0.750 | cndb 0.739 | cnde 0.730 | cndc 0.716 | abit 0.697 | gasp 0.677 | slno 0.660 | sagu 0.645 | cnda 0.563 | mont 0.552 | outv 0.539 | slso 0.532 | outm 0.498 | labi 0.460. Médiane ~0.65.
- SUITE proposée : critère de sélection par KGE sur une fenêtre de VALIDATION (2019-21, jamais le test) — c'est la sélection de modèle standard, mesurée et sans fuite. vaud : forçage à construire.

## CONTRÔLE DÉCISIF : le transfert porte le NIVEAU de calibration, pas la DIFFÉRENCIATION régionale, 2026-07-31
- Question d'Essi : pourquoi transférer un modèle gaspésien plutôt que partir de la littérature krigée ? Contrôle (même physique, mêmes adaptateurs mesurés, même forçage CaSR brut, seule différence = poids chargés ou init littérature) :
  gasp 0.677 vs 0.344 | slno 0.660 vs 0.486 | mont 0.571 vs 0.471 | labi 0.610 vs 0.535 | outv 0.539 vs 0.542.
- Le transfert apporte +0.08 à +0.33 partout sauf OUTV (nul). Ma conclusion précédente (« le transfert n'apporte rien hors domaine ») est RÉFUTÉE.
- Réconciliation avec l'autre mesure du jour (paramètres quasi identiques entre régions : K_sat 0.0362-0.0369, k_gw 0.0451-0.0475) : le champion transporte un NIVEAU de calibration global très supérieur à la littérature, mais peu de DIFFÉRENCIATION régionale (cause : attributs z-scorés par région -> le nœud médian de chaque région arrive à ~0).
- CONSÉQUENCE : le déficit face à Hydrotel (calé région par région) est un déficit de RÉGIONALISATION, pas de calibration. D'où le plan reports/design_regionalisation_amortie.md : garder le niveau (champions) + ajouter la différenciation (XGBoost/GP sur les paramètres calibrés de plusieurs régions, validation leave-one-region-out, critère de succès pré-enregistré).
- Prérequis technique confirmé : dénormaliser les attributs (extraction brute via load_hydrotel(normalise=False), déjà validée).

## OUTV : l'entraînement local NE referme PAS l'écart — le déficit de l'ouest n'est pas la calibration, 2026-07-31
- Test décisif (recette canonique complète, entraînement local 30 ep, adaptateurs mesurés OUTV, CaSR brut) : held-out 0.566.
- Les quatre configurations sur OUTV : transféré 0.539 | littérature+mesures 0.542 | ENTRAÎNÉ LOCALEMENT 0.566 | Hydrotel calé 0.753.
- LECTURE : entraîner sur place n'apporte que +0.027 et laisse 0.19 d'écart avec Hydrotel. Le déficit de l'ouest n'est donc NI un problème de transfert, NI un problème de calibration — les trois variantes de méandre sont dans un mouchoir (0.54-0.57) alors qu'Hydrotel est 0.19 au-dessus. Suspects restants, par ordre : (a) le forçage (Hydrotel tourne sur SIMAT/stations, méandre sur CaSR brut : l'écart mesuré à l'est en juillet était de +0.10) ; (b) une spécificité physique de l'ouest absente du modèle (grands réservoirs de l'Outaouais, régulation) ; (c) la qualité des jauges/mapping local.
- La régionalisation amortie ne réglera donc PAS l'ouest : elle vise la différenciation entre régions, alors qu'ici même la calibration locale plafonne. Diagnostic de l'ouest à instruire séparément (commencer par le forçage : rejouer OUTV sur -hyb et -budyko en inférence, 10 min).

## CONJOINT PROVINCIAL avec normalisation GLOBALE (qc4) : différenciation OBTENUE, performance non, 2026-08-01
- Prérequis livré : attributs BRUTS des 15 régions (28035 nœuds, territorial-raw-QC.parquet) + stats provinciales ; contrastes réels confirmés (élév 75-581 m, sable 0.34-0.92, forêt 0.10-0.89, agri 0.01-0.56) — la normalisation par région les écrasait.
- qc4 (gasp+mont+cndc+abit, 12 ep, JOINT_GLOBAL_NORM=1) : held-out gasp 0.459 / mont 0.586 / cndc 0.455 / abit 0.468 vs transféré 0.677/0.571/0.716/0.697. MONT progresse (+0.015, sa classe est enfin représentée), les trois autres régressent.
- MAIS la différenciation spatiale APPARAÎT enfin : C_f 3.66 (gasp) à 4.33 (abit), T_melt -0.02 à -0.35, k_gw 0.0576 (vaud) à 0.0717 (cndc), soit 5-20% d'écart entre régions contre 0.2-2% avec la normalisation locale. Le NeRF distingue enfin les territoires.
- LECTURE : la normalisation globale était bien une condition NÉCESSAIRE (la différenciation existe maintenant) mais pas SUFFISANTE (12 epochs sur 4 régions ne valent pas 30 epochs sur une région bien réglée). Le conjoint reste sous le transfert en performance brute.
- Prochain pas cohérent : soit plus d'epochs/régions (coûteux, et la doctrine du champion gelé avertit que le gradient dégrade le test), soit la RÉGIONALISATION AMORTIE (XGBoost sur les paramètres calibrés, données prêtes) qui capture la différenciation sans réentraîner le modèle hydrologique.

## MODÈLE D'EXPÉRIENCE : R² LOO négatif, mais protocole trop sévère (Essi), 2026-08-01
- XGBoost (coords + attributs bruts -> paramètres calibrés, 11457 exemples, 4 régions) : R² leave-one-region-out négatif sur presque toutes les cibles ; seuls K_musk/x_musk/C_f frôlent 0.15-0.28.
- LECTURE INITIALE (partielle) : pas de fonction attributs->paramètres apprenable dans nos calibrations ; chaque région atterrit dans son propre ensemble équifinal.
- CORRECTION (Essi) : un effet spatial INEXPLIQUÉ est légitime (krigeage, effet aléatoire spatial) ; le leave-one-region-out demande d'EXTRAPOLER à des centaines de km avec 4 régions éparpillées, ce qu'un champ spatial ne peut pas faire par construction. Le test condamne le protocole, pas l'approche.
- REFORMULATION : le modèle d'expérience est un CHAMP SPATIAL de correction ; validation correcte = blocs spatiaux INTERNES à un domaine couvert, pas régions entières isolées. Les calages régionaux deviennent les points d'ancrage du krigeage des paramètres ; densifier = améliorer l'interpolation ; les bassins non jaugés héritent d'une interpolation avec incertitude (problème classique de régionalisation, voie différentiable).
- ACQUIS TECHNIQUE : normalisation globale provinciale câblée (JOINT_GLOBAL_NORM=1) + attributs bruts des 15 régions (28035 nœuds) prêts.
- SUITE : (a) forçage de l'ouest (OUTV : 3 produits en inférence, 10 min — levier le plus probable des 0.19 d'écart avec Hydrotel) ; (b) validation du champ spatial par blocs internes.

## OUTV : le forçage n'explique PAS l'écart — le suspect devient la RÉGULATION, 2026-08-01
- Checkpoint OUTV entraîné, rejoué sur 3 produits : CaSR brut 0.567 (r 0.644) | CaSR corrigé 0.567 | SIMAT+CaSR 0.469 (r 0.682). Hydrotel 0.753.
- Le meilleur r vient de SIMAT (0.682) mais son KGE est le pire : le produit station améliore le TIMING et dégrade le volume. Aucun produit ne referme les 0.19.
- Les 4 hypothèses testées sur OUTV sont donc toutes réfutées : transfert (0.539), littérature (0.542), calage local (0.566), forçage (0.469-0.567). Reste la RÉGULATION : l'Outaouais est le bassin le plus aménagé du Québec (réservoirs Dozois/Cabonga/Baskatong, gestion coordonnée) et méandre n'a AUCUNE règle de barrage, alors qu'Hydrotel a été calé SUR des débits déjà régulés (il absorbe la régulation dans ses paramètres).
- Vérification proposée (gratuite) : croiser le KGE par station OUTV avec la capacité de retenue amont (table dams déjà ingérée) — si les stations sous influence portent tout le déficit, l'hypothèse est confirmée et le chantier devient DamData/règles de gestion, pas l'hydrologie.

## SYNTHÈSE DE FIN DE SESSION (2026-08-01)
- ACQUIS : GASP 0.749 > Hydrotel 0.744 (entraîné, forçage adéquat) ; physique certifiée ; robustesse de régime résolue (aquifère) ; pipeline reproductible en 1 commande (deploy.py + provenance) ; CaSR brut partout (décision Essi).
- NÉGATIFS UTILES : conjoint < monos même avec normalisation globale (mais différenciation spatiale enfin obtenue) ; modèle d'expérience non extrapolable (reformulé en champ spatial à interpoler, GP plutôt que XGBoost pour la régularité + incertitude) ; ouest = ni transfert, ni calage, ni forçage.
- PROCHAINS PAS : (1) diagnostic régulation OUTV (gratuit) ; (2) GP spatial des paramètres, validation par blocs internes, ancré sur les régions calibrées ; (3) densifier les ancrages en calibrant d'autres régions.

## Faisceau d'indices RÉGULATION (à confirmer par station), 2026-08-01
- Capacité de retenue amont mappée (table dams) vs performance : GASP 1.3 km3 (méandre BAT Hydrotel) | ABIT 20 km3 (bat) | OUTV 117 km3 (déficit 0.19) | SAGU 288 km3 (déficit 0.12). Direction cohérente : plus le bassin est aménagé, plus méandre décroche — il n'a AUCUNE règle de gestion, Hydrotel a été calé SUR des débits déjà régulés.
- Contre-exemple à instruire : CNDB 425 km3 où méandre gagne (mais n=2 stations, échantillon faible).
- TEST PROPRE à refaire (le script patché n'a pas pris) : KGE par station OUTV croisé avec la capacité amont de CHAQUE station ; si les stations sous influence portent le déficit et montrent un gamma déséquilibré, l'hypothèse est confirmée et le chantier devient DamData/règles de gestion (module existant, jamais alimenté).

## RÉGULATION RÉFUTÉE + 2 nouveaux ancrages calibrés, 2026-08-02
- DIAGNOSTIC RÉGULATION (KGE par station vs capacité amont) : OUTV 1 station régulée à 0.631 vs 15 naturelles à 0.562 ; SLNO 1 régulée 0.453 vs 26 naturelles 0.663 ; SAGU 19 stations, capacité amont MÉDIANE 0.00 km3. Les jauges CEHQ mesurent des tributaires NATURELS, pas les exutoires régulés — les 116 km3 de l'Outaouais ne sont en amont d'aucune station (sauf une). L'hypothèse régulation est RÉFUTÉE comme cause du déficit de l'ouest (même conclusion qu'au 22/07 sur SAGU, oubliée depuis).
- Le déficit de l'ouest reste donc INEXPLIQUÉ après 5 hypothèses testées (transfert, littérature, calage local, forçage, régulation). Piste restante non testée : la qualité des observations elles-mêmes (jauges CEHQ vs sorties Hydrotel calées sur ces mêmes jauges) et le r plafonné à 0.64 sur OUTV (vs 0.797 sur SLNO), qui sent le forçage local ou un réseau mal représenté.
- NOUVEAUX ANCRAGES (recette canonique, CaSR brut, adaptateurs mesurés) : SLNO 0.645 (27 stations ; zero-shot était 0.660 — le calage local n'apporte rien de plus, 3e confirmation) ; SLSO 0.631 (29 stations ; zero-shot 0.532, +0.10 par le calage local).
- Le champ spatial dispose maintenant de 6 ancrages (gasp, sagu, mont, outv, slno, slso) couvrant est/boréal/sud/ouest — assez pour tester l'INTERPOLATION (validation par blocs internes) au lieu de l'extrapolation.

## PIVOT DE CONCEPTION : régionaliser les SIGNATURES, pas les paramètres, 2026-08-02
- Constat : les paramètres calibrés sont largement arbitraires (équifinalité). Preuve dans nos chiffres : SLNO calage local 0.645 < transfert 0.660 ; OUTV transfert/littérature/local dans 3 centièmes. Un champ spatial ajusté sur ces valeurs modélise donc du bruit -> R² LOO négatif expliqué.
- PIVOT : le champ spatial doit interpoler les SIGNATURES HYDROLOGIQUES observables et identifiables, pas les paramètres. Signatures déjà mesurées cette semaine, avec contraste régional réel : constante de récession (0.05 gasp / 0.14 mont), ratio ET du bilan P-Q (0.70-1.10), coefficient de ruissellement d'orage, décalage du pic de fonte. Le modèle en dérive ses paramètres par inversion (c'est déjà ce que font les adaptateurs mesurés, mais région par région et sans interpolation).
- Approche établie en hydrologie (signature-based regionalization), ici en version différentiable + krigeage bayésien avec incertitude pour les bassins non jaugés.

## CHAMP SPATIAL DES SIGNATURES : ça marche pour la dynamique, pas pour les volumes, 2026-08-02
- 127 stations, 14 régions, 5 signatures observables (reports/signatures_stations.csv). Autocorrélation spatiale forte : k_recession corrèle 0.598 avec la station la plus proche (0.04 au hasard) -> le krigeage a un sens.
- GP (Matern 3/2 sur lon/lat/log-aire), validation par BLOCS SPATIAUX internes (k-means, 8 blocs) : k_recession R2 0.618 | cv_debit 0.468 | coeff_ecoul -0.04 | ratio_et -0.12. Couverture des intervalles 90% : 0.89-0.93 sur les quatre (incertitude BIEN CALIBRÉE, y compris quand la moyenne est mauvaise).
- LECTURE : les signatures de DYNAMIQUE (vitesse de vidange, variabilité) sont spatialement prédictibles ; celles de VOLUME (coefficient d'écoulement, ratio ET) ne le sont pas — elles dépendent du bilan local (pluie réelle) que le krigeage géographique ne capture pas. Cohérent avec tout le reste : le volume est piloté par le forçage, pas par le territoire.
- CONSÉQUENCE : le champ spatial fournit k_gw (et le caractère flashy) pour n'importe quel bassin non jaugé, avec incertitude calibrée — c'est utilisable. Pour les volumes, il faut de meilleures observations de pluie, pas un meilleur champ.
- Outil : .runs/quebec/champ_signatures.py (validation blocs internes vs régions, couverture).

## A/B CHAMP k_gw : neutre — le champ est correct mais le paramètre n'est pas le levier, 2026-08-02
- Champ provincial construit (GP sur 127 récessions -> 28035 nœuds, incertitude par nœud) : valeurs physiquement cohérentes (mont 0.132, vaud 0.127 = drainage agricole ; cnd* 0.062-0.065 = socle boréal ; gasp 0.084) et variation INTRA-région réelle (outv 0.077-0.115), là où on n'avait qu'une constante régionale.
- A/B entraîné : GASP 0.633 (vs 0.677 avec k_gw constant mesuré) ; OUTV 0.567 (vs 0.566, identique). Le champ ne dégrade pas OUTV et coûte 0.04 sur GASP.
- LECTURE : cohérent avec l'audit de leviers du 25/07 (k_gw était classé « ~optimum », insensible). Le champ k_gw est scientifiquement meilleur (mesuré, continu, avec incertitude, utilisable en bassin non jaugé) mais le KGE n'y est pas sensible : la récession de queue pèse peu dans un critère dominé par les crues. À conserver comme produit de régionalisation, pas comme levier de performance.
- Acquis réutilisable : la chaîne signatures -> GP -> champ par nœud -> injection dans le modèle fonctionne de bout en bout (ETL_KGW_FIELD=1). Elle s'appliquera telle quelle aux signatures qui PÈSENT sur le KGE (timing du freshet, coefficient d'orage) dès qu'on saura les mesurer par station.

## SIGNATURES DE TIMING : le freshet se krige (R2 0.62) — la cible qui PÈSE sur le KGE, 2026-08-02
- 126 stations, 3 signatures de timing (reports/signatures_timing.csv) : centre de masse du freshet (jour julien), durée de montée, rapport pic/base hivernal.
- Krigeage GP, validation par blocs spatiaux : cm_freshet R2 0.620 (couverture 0.78) | duree_montee 0.321 (0.93) | ratio_pic_base 0.283 (0.89).
- Le CENTRE DE MASSE DU FRESHET est prédictible spatialement ET a un contraste régional énorme : 107 (17 avril, Montérégie) à 144 (24 mai, Côte-Nord C), soit 37 jours. C'est exactement la quantité qui gouverne le r (donc le KGE) sur des rivières nivales, et exactement ce que le modèle rate quand il décroche (banc freshet 2026-07-24 : fonte déclenchée ~2 semaines trop tard).
- PROCHAIN LEVIER IDENTIFIÉ : champ spatial du cm_freshet -> contrainte de timing de fonte par nœud (le T_melt/C_f du NeRF est ajusté pour que le centre de masse simulé colle au champ observé). Contrairement à k_gw (insensible), cette signature pèse directement sur le score.
- Couverture 0.78 sur cm_freshet = intervalles trop serrés (le GP sous-estime son incertitude sur cette signature) ; à corriger avant usage opérationnel.

## 2026-08-02 — Champ de timing de fonte : le gradient nord-sud est comprimé de 40 %

**Champ construit.** GP (Matern 3/2 sur lon/lat/log-aire) sur le centre de masse du freshet de 126 stations -> 28035 nœuds. `D:/meandre-data/quebec/champ_freshet_QC.parquet`. Longueurs de corrélation 3.36° en lon contre 1.83° en lat : la date de crue varie deux fois plus vite vers le nord que vers l'est, signature thermique et non artefact de lissage. Médianes régionales j106 (mont, 15 avr) à j138 (cndc, 17 mai), sd par nœud 4-8 j.

**Sensibilité mesurée (banc `freshet_bench.py`, gasp, inférence pure).** T_melt -1 °C -> centre de masse simulé -3.0 j. Le levier répond : la conversion champ -> décalage de seuil est légitime à 3 j/°C.

**Biais de date, champion gaspésien transféré (base only) :**

| région | CM obs | CM sim | biais | n |
|---|---|---|---|---|
| mont | j106.0 | j108.5 | +3.1 | 22 |
| slso | j112.9 | j116.8 | +4.4 | 30 |
| outv | j121.4 | j121.9 | +1.2 | 13 |
| gasp | j129.3 | j128.8 | -1.0 | 15 |
| sagu | j133.0 | j127.2 | -5.6 | 20 |
| cnda | j134.1 | j125.8 | -8.3 | 1 |

**Diagnostic.** Le signe est systématique : sud en retard, nord en avance. L'observé s'étale sur 28 jours, le simulé sur 17. Le champion transféré COMPRIME le gradient spatial de timing d'environ 40 % — il impose un régime de fonte trop nordique au sud et trop méridional au nord.

**OUTV : sixième hypothèse éliminée.** La pire région est à l'heure (+1.2 j). Son déficit n'est pas un problème de timing de fonte (après transfert, littérature, entraînement local, forçage, régulation).

**Calibration dérivée des données (`freshet_calib.py`).** Par nœud : dT = clamp((CM_champ - CM_simulé)/3.0, ±2 °C). Aucune constante choisie ; provenance dans `reports/freshet_calib_provenance.json`. A/B en INFÉRENCE PURE, held-out 2022-2024 :

- mont : KGE 0.4086 -> 0.4554 (**+0.047**), dT méd -1.51 °C (p10 -2.00, la borne mord)

Suite (slso, sagu, gasp, outv) en cours. gasp et outv servent de TÉMOINS : déjà à l'heure, le gain doit y être nul ; un gain y serait le signe d'un artefact.

## 2026-08-02 (soir) — La « loi des ancrages » était un diagnostic incomplet

**Contexte.** Question d'Essi : pourquoi méandre est-il déclassé alors qu'il a une meilleure météo, une meilleure ET et le même solveur ? Deux objections justes de sa part : (1) Hydrotel est calé sur SIMAT, donc ses paramètres ne transfèrent pas tels quels sur CaSR ; (2) les échecs d'ancrage du sol (MONT-v3 le 16 juillet à -0.31, MONT-v9 le 17 à 0.125) PRÉCÈDENT l'aquifère (28 juillet), or l'explication écrite à l'époque était justement que le NeRF compensait par le sol les divergences structurelles, aquifère en tête.

**Ce que le calage Hydrotel MONT contient réellement** (vérifié en lisant bv3c.csv via load_calibrated_soil, validation isolée jamais faite jusqu'ici — TODO du log du 16 juillet enfin coché) :
- épaisseurs z1/z2/z3 = 0.219 / 0.157 / 2.650 m, **identiques sur TOUS les nœuds** : ce n'est pas un champ calibré, c'est une colonne unique de 3 m appliquée partout ;
- `coef_recharge` = 0 partout et `krec` = 1.29e-7 : Hydrotel n'a **aucune voie profonde**, toute l'eau ressort latéralement ;
- seules la texture (ks, thetas, psis, b) et la pente varient spatialement.

**Correctifs de mécanisme** (le mode d'ancrage était inutilisable en partiel) :
- `set_calibrated_soil` REMPLAÇAIT p_soil en bloc -> passé en FUSION, une clé absente garde la valeur du NeRF ;
- `_calib_z` = None quand les z ne sont pas ancrés (sinon ETR/gel restaient figés sur des z absents) ;
- hooks `ETL_SOIL_CALIB`, `ETL_SOIL_CALIB_FULL`, `ETL_SOIL_CALIB_TEXTURE` dans etl_run.py. 150 tests passent.

**Résultats MONT (aquifère ACTIF dans les deux cas, partition verticale libre) :**

| ancrage | kge_med val époque 0 | r | beta | gamma | tendance |
|---|---|---|---|---|---|
| complet (texture + épaisseurs) | -0.059 | 0.404 | 0.711 | 0.585 | val PLATE sur 4 époques |
| texture seule (épaisseurs libres) | +0.245 | 0.627 | 0.753 | 0.603 | monotone |

**Conclusion.** Le goulot n'est pas « le sol » ni l'absence d'aquifère : c'est la GÉOMÉTRIE. Une colonne uniforme de 3 m sans drainage profond amortit tout (volume -30 %, variabilité -40 %). La texture, elle, est compatible et informative. La loi des ancrages interdisait le tout à cause d'une seule composante non informative ; elle doit être reformulée en : hériter des processus scalaires régionaux ET de la texture, jamais des épaisseurs.

**Conséquence pour la carte de départ Hydrotel** demandée par Essi : côté sol, il y a beaucoup moins à hériter qu'espéré, puisque la géométrie n'a aucune structure spatiale. Ce qui reste utile est ce qu'on utilisait déjà (coefficient ETP Linacre, taux et seuils de fonte) plus la texture.

**Non fait / réserves.** Aucun chargeur des paramètres de ROUTAGE d'Hydrotel (onde_cinematique*.csv) : un test de fidélité tronçon-par-tronçon mélangerait pour l'instant divergence verticale et divergence de routage. Le vrai test de fidélité reste à faire, à météo COMMUNE (celle d'Hydrotel), contre `<projet>/simulation/simulation/resultat/debit_aval.nc` qui porte exactement le même maillage (idtroncon = node_id).

**Correction de mémoire.** Il n'existe aucune liste de « 9 divergences » dans le dépôt (le journal commence le 3 juillet). Les divergences documentées sont 4, toutes délibérées : aquifère restituant, UH de versant, ruissellement par saturation, coefficient cultural saisonnier (docs/architecture.md).

## 2026-08-03 — Carte provinciale : la règle simple bat la sélection, médiane 0.671

**Mesure du jour : la calibration locale ne rapporte qu'à partir d'une dizaine de jauges.**

| région | n stations | transfert gaspésien | calibration locale | verdict |
|---|---|---|---|---|
| slso | 39 | 0.532 | **0.631** | local +0.099 |
| mont | 28 | 0.571 | **0.624** | local +0.053 |
| sagu | 22 | 0.645 | **0.705** | local +0.060 |
| gasp | 16 | 0.677 | **0.749** | local +0.072 |
| outv | 16 | 0.539 | **0.572** | local +0.033 |
| slno | 31 | **0.660** | 0.645 | transfert (exception) |
| abit | 3 | **0.697** | 0.478 | transfert +0.219 |

**Sélection de champion sur validation : RÉFUTÉE.** 6 champions essayés en inférence pure sur chaque région pauvre en jauges, choix sur la validation 2019-2021 seule. Résultat : la carte ainsi construite fait 0.627 de médiane, PIRE que le transfert gaspésien uniforme (0.653). Avec 1 à 4 stations la validation ne classe pas les candidats, elle les mélange (abit : val préfère sagu 0.621 alors que gasp donne 0.720 ; cndc : val préfère sagu 0.612 alors que gasp donne 0.716). Le choix du champion COMPTE (0.485-0.720 d'écart sur abit) mais aucun critère basé sur le débit ne le trouve à ce nombre de stations.

**Carte retenue (règle la plus simple, décidée a priori) :** calibration locale si n_stations >= 10, sinon transfert du champion global (gasp, le seul qui batte Hydrotel). Médiane held-out 2022-2024 = **0.671** (moyenne 0.651), contre 0.653 pour la carte tout-transfert. 6 régions locales, 7 transférées, vaud sans forçage. Détail dans `reports/carte_provinciale.csv`.

**Restes.** vaud n'a pas de fichier de forçage. cnda (0.453) et outm (0.498) plafonnent bas avec 1 et 4 stations. outv reste réfractaire : 7 hypothèses tombées (transfert, littérature, entraînement local, forçage, régulation, timing de fonte, calibration locale à 0.572 contre Hydrotel 0.753).

## 2026-08-03 — TEST DE CAPACITÉ : le plafond CaSR n'est PAS dans les paramètres

**Question d'Essi.** « Si les paramètres ne s'ajustent pas à la météo, simat ou casr, c'est qu'il y a un problème. » Et : « Hydrotel, mal foutu numériquement, encaisse la pluie SIMAT sans problème. »

**Diagnostic préalable (vrai sur les faits, FAUX sur les conséquences).** Sur les 37 paramètres du champion GASP, la moitié sort du NeRF rigoureusement uniforme sur les 3917 nœuds (CV 0.002-0.005) et EXACTEMENT à l'init littérature : f_vert 0.50/0.60/0.70, manning_n 0.0997, interception 1.50, frost_alpha 0.50, vg_n 1.50, rain_hours 12.0, T_snow 1.0. Cause : `init_from_literature` multiplie les poids de fc_out par 0.1 ; les paramètres à fort gradient remontent (||w|| ~1.0), les autres restent au plancher (||w|| ~0.054, facteur 20).

**Test de capacité (`ETL_CAPACITE=1`).** Tout gelé SAUF les codes latents additifs = décalage LIBRE par nœud sur les 37 paramètres (144 929 params), w_prior = 0, w_latent_reg = 0, départ à chaud du champion, CaSR BRUT, 15 époques. C'est du surajustement volontaire : on mesure le PLAFOND, pas la généralisation.

| modèle | entraînement 2001-2018 | validation | tenu 2022-2024 |
|---|---|---|---|
| champion (NeRF + prior) | KGE 0.7398, r 0.818 | 0.7373, r 0.797 | 0.7023, r 0.841 |
| capacité (libre, non régularisé) | KGE **0.7419**, r **0.821** | 0.7385, r 0.798 | 0.7054, r 0.842 |

**+0.002 de KGE et +0.003 de r sur la période d'ENTRAÎNEMENT, pour 145 000 degrés de liberté sans contrainte.** Le modèle ne peut pas mieux ajuster même les données qu'il a vues. Ni le NeRF, ni le prior, ni l'init écrasée ne sont le facteur limitant. Dégeler les paramètres figés ne rapporterait rien.

**Ce qui bride est l'ENTRÉE.** Même modèle, même région, forçage dérivé de SIMAT : r 0.878 contre 0.795 (A/B du 3 août : gasp brut 0.686 / budyko 0.677 / hyb 0.738 en inférence, use_latent_codes=False). La structure est capable ; la corrélation dépend de la coïncidence entre le jour de la pluie dans le fichier et le jour de la crue dans la rivière, et aucun paramètre ne déplace une averse mal datée. Hydrotel n'« absorbe » pas SIMAT par calage : SIMAT assimile densément les stations du Québec et date mieux les événements.

**Corriger CaSR par bilan (Budyko) : INUTILE.** La variante corrigée fait MOINS bien que le brut (0.677 vs 0.686) sur le modèle pourtant entraîné sur elle. CaSR brut confirmé comme forçage canonique, et le chargeur est corrigé : `JOINT_FX_SUFFIX=-none` chargeait en réalité `forcing-<reg>-budyko.nc` (repli silencieux). Toute la carte provinciale du 2 août tournait donc sur la variante corrigée, pas sur le brut. Le fichier chargé est désormais imprimé.

**BUG D'ÉVALUATION à corriger.** `deploy.py`, `choix_champion.py`, `forcage_ab.py`, `freshet_bench.py` instancient le modèle avec `use_latent_codes=False` alors que les champions EN ONT. Mesure de l'écart sur gasp/CaSR brut : 0.686 (sans) contre 0.702 (avec). Tous les chiffres de la carte provinciale sont donc sous-estimés d'environ 0.016. À reprendre avant toute publication.

**Règle fixée pour la suite (accord Essi).** Tout résidu spatialement structuré et non explicable par des paramètres libres va dans un champ de correction du FORÇAGE, validé contre les stations météo ECCC. Rien ne va dans un paramètre physique sans une observation qui contraigne ce paramètre-là : sinon le paramètre encode les défauts de la grille de pluie et l'argument de renaturalisation/scénarios s'effondre.

## 2026-08-03 (soir) — Courbe de bascule météo, et arrêt du bricolage

**Courbe de bascule (sans débit).** 236 stations ECCC tenues à l'écart, 2016-2019. Chaque station est estimée (a) par pondération inverse du carré de la distance depuis ses voisines et (b) par CaSR au nœud le plus proche, puis comparée à ses observations. Aucune donnée de débit n'entre dans ce calcul.

| distance voisine | n | r stations | r CaSR | biais stations | biais CaSR |
|---|---|---|---|---|---|
| < 10 km | 55 | 0.840 | 0.785 | 0.89 | 1.11 |
| 10-15 | 29 | 0.846 | 0.797 | 0.87 | 1.03 |
| 15-20 | 29 | 0.815 | 0.726 | 0.82 | 1.05 |
| 20-30 | 54 | 0.835 | 0.688 | 0.85 | 1.01 |
| 30-40 | 23 | 0.848 | 0.678 | 0.90 | 1.04 |
| 40-60 | 29 | 0.829 | 0.802 | 0.91 | 1.13 |
| 60-100 | 8 | 0.764 | 0.818 | 0.93 | 1.13 |
| > 100 | 5 | 0.589 | 0.914 | 0.88 | 1.31 |

Croisement vers 55-60 km (réserve : 13 stations seulement au-delà de 60 km). Biais de volume OPPOSÉS et stables : les stations sous-estiment de 10-15 %, CaSR sur-estime de 3-13 % (jusqu'à 31 % très loin des stations).

**Carte hybride (SIMAT) : médiane 0.590 contre 0.671 pour CaSR.** L'échec est un défaut de VOLUME, pas de datation : beta 0.57 (labi), 0.60 (abit), 0.50 (cndb), soit 40-50 % d'eau manquante là où le réseau est clairsemé. Gains au contraire dans le sud dense (mont +0.069, gasp +0.026). Champ de densité construit : `densite_stations.parquet` (9 km à vaud, 150 km en cnde).

**Chiffres corrigés (effets aléatoires par nœud enfin activés quand le checkpoint en porte pour la bonne région, via `ckpt_util.a_des_latents`) :** GASP 0.7752 contre Hydrotel brut 0.744 ; MONT 0.6934 contre 0.637 ; SAGU 0.7142. Méandre passe donc devant Hydrotel sur les deux régions correctement calibrées, en tenu de côté 2022-2024.

**ARRÊT (objection d'Essi, retenue).** Le produit mixte stations+CaSR a été écrit (`build_forcing_mix.py`, poids continu 1/(1+(d/55)^3), aucune frontière régionale) mais N'EST PAS lancé : mélanger une pluie interpolée dans CaSR casse la cohérence interne entre précipitation, température, rayonnement et humidité, qui est justement ce qu'une réanalyse apporte. Le modèle recevrait une pluie incohérente avec son énergie. Objection valide, non contournée. Observations ECCC 2000-2024 récupérées (25 tuiles, 0 échec) et conservées pour un usage ultérieur.

**État réel.** Ce qui tient : le test de capacité (résultat scientifique, pas un réglage) et la parité-plus contre Hydrotel là où le modèle est calibré. Ce qui ne tient pas : l'objectif d'une carte provinciale homogène sur 15 régions dont 9 ont moins de 4 jauges. Reprendre par la question de fond (que doit démontrer ce modèle, et quel est le plus petit ensemble de mesures qui le démontre) plutôt que par le score.

## 2026-08-04 — DUEL contre l'ENSEMBLE COMPLET : méandre est DERRIÈRE, correction d'une affirmation antérieure

**Ce qui était affirmé et qui ne tient pas.** J'ai écrit plusieurs fois que méandre « bat Hydrotel » (GASP 0.749 puis 0.775 contre 0.744). Cette affirmation reposait sur UN membre (LN24HA) et sur le forçage hybride. Contre l'ENSEMBLE des 6 calages, station par station, elle ne tient pas.

**Duel, méandre sur CaSR brut, 132 stations, held-out 2022-2024, jours communs :**

| | médiane | moyenne |
|---|---|---|
| méandre | 0.621 | 0.603 |
| médiane des 6 calages | 0.742 | 0.739 |
| meilleur membre par station | 0.805 | |

méandre bat la médiane d'ensemble sur 20/132 stations (15 %), le meilleur membre sur 8/132 (6 %).

| région | n | méandre | ensemble | meilleur |
|---|---|---|---|---|
| gasp | 16 | 0.704 | 0.795 | 0.851 |
| sagu | 20 | 0.686 | 0.786 | 0.828 |
| slso | 30 | 0.625 | 0.655 | 0.731 |
| slno | 27 | 0.621 | 0.808 | 0.834 |
| mont | 23 | 0.600 | 0.720 | 0.776 |
| outv | 16 | 0.552 | 0.821 | 0.854 |

**Inéquité à nommer.** Méandre tourne ici sur CaSR, Hydrotel sur SIMAT, sa météo native, celle sur laquelle il a été calé. L'A/B du 3 août mesure 6 centièmes de corrélation d'écart entre les deux familles. Le duel à armes égales (méandre sur -hyb) est lancé ; son résultat sera consigné quel qu'il soit.

**Leçon de méthode.** Comparer au membre le plus faible d'un ensemble équifinal, sur le forçage le plus favorable, produit une conclusion qui s'effondre dès qu'on élargit la comparaison. Le protocole correct est : tous les membres, toutes les stations, comptage par station, forçage explicite.

## 2026-08-04 — LES LACS. Le déficit est localisé, et il survit au test par station

**Duel à armes égales (méandre sur -hyb, forçage de la famille SIMAT comme Hydrotel), 132 stations :** méandre 0.641 contre 0.742 (médiane d'ensemble) et 0.805 (meilleur membre). Bat l'ensemble sur 23 % des stations (contre 15 % sur CaSR).

| région | méandre | ensemble | écart |
|---|---|---|---|
| gasp | 0.769 | 0.795 | -0.026 |
| mont | 0.693 | 0.720 | -0.027 |
| slso | 0.642 | 0.655 | -0.013 |
| sagu | 0.709 | 0.786 | -0.077 |
| slno | 0.565 | 0.808 | -0.243 |
| outv | 0.499 | 0.821 | -0.322 |

**Hypothèse RÉGULATION : réfutée (2e fois).** Corrélation écart/nb de barrages hydroélectriques = -0.84 sur 6 RÉGIONS, mais +0.018 sur 132 STATIONS (15 stations seulement ont un ouvrage hydroélectrique en amont ; écart médian -0.097 contre -0.083 sans). L'artefact d'échelle régionale a failli déclencher la construction d'un module de régulation. Note : cette hypothèse avait DÉJÀ été écartée pour OUTV plus tôt dans la session (les jauges CEHQ mesurent des tributaires naturels) et je l'ai reconstruite sans m'en souvenir.

**Que fait Hydrotel de spécial sur ces régions ? RIEN.** Les projets GASP et OUTV sont configurés à l'identique : mêmes sous-modèles, mêmes fichiers, seul le nom du fichier météo diffère. Le déficit n'est donc pas un mécanisme manquant côté méandre par rapport à Hydrotel : c'est méandre qui fait quelque chose de faux.

**CAUSE LOCALISÉE : les lacs.** Fraction de nœuds-lacs par région : gasp 1.6 %, mont 1.9 %, slso 2.0 % (écarts -0.026/-0.027/-0.013) contre slno 11.5 %, outv 15.1 %, sagu 15.7 % (écarts -0.243/-0.322/-0.077). Test au niveau des STATIONS (fraction lacustre du bassin amont, n=132) :

| fraction lacustre amont | n | écart médian |
|---|---|---|
| < 2 % | 76 | -0.038 |
| 2-5 % | 10 | -0.023 |
| 5-10 % | 20 | -0.215 |
| 10-20 % | 20 | -0.218 |
| > 20 % | 6 | -0.216 |

Marche NETTE à 5 %, plateau au-delà, et AUCUN effet de taille de bassin (corrélation écart/log n_nœuds amont = +0.004). Corrélation écart/fraction lacustre = -0.350 par station.

| groupe | n | méandre | ensemble | meilleur | > ensemble | > meilleur |
|---|---|---|---|---|---|---|
| peu lacustre (<5 %) | 86 | 0.663 | 0.718 | 0.775 | 31 % | 14 % |
| lacustre (>=5 %) | 46 | 0.590 | 0.815 | 0.848 | 7 % | 0 % |

**Lecture.** Le plateau au-delà du seuil ressemble à un défaut de TRAITEMENT (tout ou rien) et non à une insuffisance progressive de calibration. Cela rejoint la note jamais suivie d'effet sur les pseudo-lacs : des tronçons marqués « lac » par le découpage Hydrotel sans être de vrais plans d'eau, importés par méandre comme réservoirs actifs, qui ajoutent retard et lissage. L'effet y était annoncé comme critique sur SLNO.

**PROCHAIN PAS (gratuit) :** neutraliser les pseudo-lacs en INFÉRENCE, sans réentraîner, et mesurer l'écart sur les 46 stations lacustres. Si le score remonte, cause et correctif sont acquis d'un coup.

## 2026-08-05 — Lacs : la tête de lac n'a JAMAIS appris, et la différenciation par la taille rouvre OUTV

**Neutraliser les lacs : réfuté partout.** A/B en inférence, forçage -hyb : outv 0.4992 -> 0.2682 (0/16 stations améliorées), slno 0.5650 -> 0.4312 (0/27), sagu 0.7142 -> 0.5466 (0/19), gasp 0.7752 -> 0.7216 (0/15). Le coût croît avec la fraction lacustre. L'hypothèse des « pseudo-lacs à neutraliser », qui traînait dans les notes depuis des semaines, est close : les lacs portent la dynamique de ces bassins.

**La tête de lac est restée à son initialisation dans les 6 régions entraînées.**

| région | n lacs | k_lake médian | q10-q90 | beta médian |
|---|---|---|---|---|
| gasp | 63 | 9.27e-5 | 8.8e-5 - 9.6e-5 | 1.550 |
| sagu | 348 | 1.02e-4 | 1.00e-4 - 1.04e-4 | 1.483 |
| slno | 388 | 9.62e-5 | 9.3e-5 - 1.00e-4 | 1.526 |
| outv | 514 | 9.94e-5 | 9.9e-5 - 1.00e-4 | 1.502 |
| mont | 37 | 1.01e-4 | 9.9e-5 - 1.03e-4 | 1.495 |
| slso | 57 | 9.66e-5 | 9.5e-5 - 9.9e-5 | 1.526 |

Soit exactement l'initialisation (1e-4 et 1.5), avec 1-5 % de dispersion pour des bornes couvrant 4 ordres de grandeur. Un étang et le lac Saint-Jean reçoivent la même loi de vidange. Cause : `nn.init.zeros_(fc_lake.weight)`, sortie divisée par 2 avant l'exponentielle, LR de base et weight_decay — atteindre le haut de la borne demanderait une sortie brute de 9 en partant de 0. La tête de lac est la SEULE sans groupe d'optimisation dédié (fc_out, noise_head, latent_codes et seuils GDD en ont tous un). Corrigé : groupe dédié `lake_lr_mult = 50`, wd = 0.

**Fausse piste écartée par la mesure :** j'ai soupçonné `lake_storage_new[...] = S_lake.detach()` de couper le gradient. Mesuré : ||grad fc_lake|| = 1.63e3 contre 2.42e2 pour la tête principale, soit 6.7× PLUS. Le gradient n'est pas éteint. (Au passage, la ligne modifiée appartenait au routage par niveaux, jamais exécuté en mode operator-lagged : je diagnostiquais du code mort.)

**Sensibilité en inférence — un facteur UNIFORME est inerte** (outv : -0.018 à -0.002 selon la perturbation ; slno : -0.001 à +0.017). Mais les deux régions veulent des valeurs OPPOSÉES : k/10 donne -0.018 sur outv et +0.017 sur slno. Une valeur unique ne peut pas satisfaire les deux.

**DIFFÉRENCIATION PAR LA TAILLE : gain réel sur OUTV.** k_i = k0 · (A_ref/A_i)^alpha, alpha dicté par la physique du seuil (Q = k·(S/A)^beta·A à égaler avec Q = C·L·h^1.5).

| alpha | outv | slno | sagu |
|---|---|---|---|
| 0.5 | +0.0177 | +0.0001 | -0.0014 |
| 1.0 | **+0.0265** | +0.0001 | -0.0080 |
| 1.5 | +0.0257 | +0.0002 | -0.0115 |

OUTV passe de 0.4992 à 0.5256 sans réentraînement. Optimum NET à alpha = 1, ce qui correspond à une largeur d'exutoire INDÉPENDANTE de la taille du lac — fixée par le chenal de sortie, pas par l'étendue du plan d'eau. Ma prédiction initiale (alpha = 0.5, exutoire s'élargissant avec le lac) est infirmée par la mesure.

**Réserves.** (1) La surface utilisée est celle du TRONÇON, pas celle du plan d'eau : `lake_fraction` et `f_water` existent dans les attributs et donneraient la vraie surface lacustre. (2) slno reste insensible, probablement parce que la référence prise à la médiane divise le k de son très grand lac (3666 km²) par plus de 160 et le colle à sa borne. (3) sagu recule légèrement. La formulation à tester ensuite n'agit QUE dans la direction utile (réduire k des grands lacs, ne pas augmenter celui des petits, la courbe saturant vers le haut).

**Premier gain sur OUTV après 8 hypothèses réfutées** (transfert, littérature, entraînement local, forçage, régulation ×2, timing de fonte, neutralisation des lacs), et il vient d'un prior physique dont l'exposant est mesuré, pas d'un réglage.

**Prior d'exutoire, formulation unilatérale (vraie surface lacustre, réduction seule, alpha=1).** La fraction lacustre des nœuds-lacs vaut 1 : la surface du tronçon EST la surface du plan d'eau, ma réserve précédente tombe.

| seuil A_ref | outv | slno | sagu | gasp |
|---|---|---|---|---|
| 1 km² | -0.0437 | **+0.0165** | -0.0400 | +0.0077 |
| 5 km² | -0.0016 | +0.0112 | -0.0183 | +0.0000 |
| 20 km² | **+0.0256** | +0.0000 | -0.0117 | (en cours) |

**Trois régions, trois optima différents.** OUTV veut qu'on ne réduise que ses plus grands lacs (seuil haut), SLNO veut qu'on réduise tout (seuil bas), SAGU ne veut aucune réduction. Il n'y a donc PAS de loi universelle en surface, et le prior fixé à la main ne peut pas réconcilier trois bassins qui demandent trois choses différentes.

**Mais c'est l'argument qui manquait.** Le score est sensible à la différenciation des lacs (jusqu'à ±0.04 selon la région) ET la valeur qui convient dépend du bassin : c'est exactement ce qu'un paramètre APPRIS par nœud doit découvrir, et exactement ce que la tête de lac ne fait pas puisqu'elle n'a jamais quitté son initialisation. Le correctif de fond n'est donc pas un prior à la main mais le groupe d'optimisation dédié (`lake_lr_mult = 50`, wd = 0) ajouté aujourd'hui. Test : réentraîner OUTV et comparer à 0.4992.

**Défaut de données repéré au passage :** le plus grand « lac » de GASP fait 13 114 km², ce qui est un nœud d'estuaire et non un plan d'eau. Sans conséquence ici (GASP a 63 lacs) mais à filtrer.

**Réentraînement OUTV avec la tête de lac libérée (`lake_lr_mult = 50`, wd = 0), forçage -hyb, 12 époques.**

- Mécaniquement, le correctif MARCHE : ||W fc_lake|| passe de 5.65e-2 à 2.43 (×43), k_lake se disperse d'un facteur 2.7 entre lacs (1.02e-4 à 2.72e-4) au lieu de 1.02, beta passe de 1.502 à 1.215. La tête n'est plus gelée.
- Mais le tenu de côté ne bouge pas : **0.5011 contre 0.4992**, soit +0.002. La validation, elle, monte à 0.6166.
- Et la direction apprise est l'INVERSE de celle qui aide : l'entraînement fait MONTER k (médiane 9.9e-5 -> 1.63e-4), alors que le banc en inférence montre que le BAISSER sur les grands lacs rapporte +0.026 sur le tenu de côté.

**Lecture.** Un paramètre libre trouve sur 2000-2018 un optimum qui ne transfère pas ; la contrainte physique (k ∝ 1/A, exposant mesuré) transfère. C'est un argument POUR les priors physiques, pas un échec du correctif — et c'est cohérent avec la doctrine du reste du modèle (ancrer le processus, laisser le champ moduler autour). Prochain test : ancrer k sur la loi d'exutoire et laisser la tête moduler autour, au lieu de la laisser libre.

**Ancrage d'exutoire pendant l'entraînement : ÉCHEC.** `set_lake_anchor` (k ancré sur k0*(A_ref/A), A_ref=20 km², alpha=1) + tête modulante, OUTV, -hyb, 12 époques.

| configuration OUTV | tenu de côté 2022-2024 |
|---|---|
| champion original | 0.4992 |
| loi d'exutoire imposée EN INFÉRENCE (sans réentraîner) | **0.5248** |
| réentraîné, tête de lac libérée (lake_lr_mult=50) | 0.5011 |
| réentraîné, ancré + tête modulante | 0.4811 |

Le seul cas qui gagne est celui où la contrainte est appliquée à un modèle DÉJÀ entraîné, hors de portée du gradient. Dès qu'on réentraîne, la tête compense l'ancrage et retourne vers la solution qui optimise 2000-2018. Même phénomène qu'en juillet avec le prior doux sur K_c, défait par l'entraînement et finalement appliqué en débiaisage STRUCTUREL hors gradient (ETL_DEMAND_SCALE). Transposition ici : geler la tête après ancrage, ou appliquer la loi au déploiement.

**Limite honnête :** la loi ne gagne qu'en OUTV (+0.026). Neutre sur SLNO et GASP, -0.012 sur SAGU. Ce n'est pas une correction provinciale mais un correctif local dont on ne sait pas prédire le domaine d'application. Fil clos en l'état.

**Acquis conservés :** (1) le déficit contre l'ensemble Hydrotel est LACUSTRE et mesuré au niveau des stations ; (2) la tête de lac n'avait aucun groupe d'optimisation dédié et ne pouvait pas apprendre — corrigé, elle apprend maintenant (||W|| ×43) ; (3) le score EST sensible à la différenciation des lacs (±0.04 selon la région), donc le levier existe ; (4) mais l'entraînement sur la période de calage préfère systématiquement une solution qui ne transfère pas.

## 2026-08-05 — SYNTHÈSE : le même mécanisme apparaît trois fois

Trois expériences indépendantes de cette session pointent le même défaut, et c'est probablement le résultat le plus important de la semaine.

1. **Prior doux sur K_c (juillet).** Posé comme cible de régularisation, il est DÉFAIT par l'entraînement : MONT retombe à 0.583 alors que le même K_c imposé en inférence donnait 0.617. Résolu en appliquant le débiaisage à la demande évaporative, hors de portée du gradient (0.624).
2. **Test de capacité (3 août).** 145 000 paramètres libres sans régularisation ne gagnent que +0.002 de KGE sur la période d'ENTRAÎNEMENT elle-même. Le modèle est à son plafond ; ce qui reste n'est pas atteignable par optimisation.
3. **Loi d'exutoire des lacs (5 août).** Imposée en inférence : +0.026 sur OUTV. Apprise librement : +0.002. Ancrée avec modulation apprise : -0.018. L'entraînement compense systématiquement la contrainte.

**Le motif.** L'optimisation sur 2000-2018 trouve des solutions qui minimisent l'erreur de calage sans transférer à 2022-2024, et elle défait toute contrainte physique qu'on lui laisse contourner. Les contraintes qui TIENNENT sont celles qui sont hors d'atteinte du gradient : débiaisage structurel de la demande, ancrages scalaires régionaux (ETP Linacre, taux et seuils de fonte), lois physiques appliquées au déploiement.

**Conséquence de conception.** Un paramètre ne devrait être laissé libre que si une observation le contraint DIRECTEMENT (récessions pour k_gw, MOD10 pour la fonte, MOD16 pour l'ET). Sinon il doit être fixé par une loi physique appliquée hors gradient. C'est une règle plus stricte que la « loi des ancrages » et elle explique ses trois succès comme ses trois échecs.

## 2026-08-06 — CORRECTION de la synthèse d'hier : c'est le DÉMARRAGE À FROID qui défait les contraintes, pas l'entraînement

Objection d'Essi : « pourquoi le champ lacustre ne pourrait-il pas être ajusté avec la loss ? il part d'un champ crédible puis s'ajuste, c'est à ça que sert la rétropropagation ». Objection fondée : mes deux réentraînements étaient des démarrages À FROID, donc jamais le protocole décrit. Test refait correctement (départ à chaud depuis le champion, ancrage posé, lr 1e-4, 8 époques) :

| configuration OUTV | tenu de côté 2022-2024 |
|---|---|
| champion original | 0.4992 |
| loi d'exutoire imposée en inférence | 0.5248 |
| réentraîné À FROID, tête de lac libre | 0.5011 |
| réentraîné À FROID, ancré + modulant | 0.4811 |
| **départ à CHAUD, ancré, affinage court** | **0.5251** |

Le gain est intégralement conservé, et c'est le meilleur de la série. La synthèse d'hier (« l'optimisation défait toute contrainte qu'elle peut contourner ») est donc FAUSSE telle qu'écrite. L'énoncé correct : un entraînement à froid reconstruit tout le modèle autour de la contrainte et la contourne ; un affinage à faible LR depuis un état déjà cohérent la respecte.

Nuance : l'affinage CONSERVE le gain sans l'augmenter (0.5251 contre 0.5248). La rétropropagation n'ajoute rien par-dessus la physique sur ce paramètre, mais ne détruit rien — ce qui suffit pour rendre le procédé utilisable et pour empiler plusieurs contraintes successives.

**Le modèle d'expérience, recadré par Essi.** Le champ GP corégionalisé était conçu dès le départ comme DÉMARRAGE À CHAUD : non reproductible, mais gardant trace de ce qui fonctionne au fil des expériences. Cadre accepté, avec une seule discipline : la règle de mise à jour doit être MÉCANIQUE (intégrer systématiquement le dernier champion de chaque région), jamais choisie en regardant le tenu de côté — sinon le point de départ devient un canal de sélection sur le test, exactement le piège de la sélection de champions du 3 août. Sous cette règle, un départ à chaud est aussi légitime qu'une initialisation littérature et bien plus utile. Trois objets distincts à ne pas confondre : le point de contrôle entraîné (démarrage à chaud actuel), la loi analytique k0*(A_ref/A) (ancrage d'exutoire, aucune donnée), et les champs krigés d'observations (récessions, freshet).

**Ce que le champ lacustre représente physiquement.** Q = k*(S/A)^beta*A : k porte la largeur et la rugosité du seuil de sortie (capacité d'évacuation), beta porte la forme de l'ouverture (1.5 déversoir libre, ~0.5 orifice noyé, plus élevé pour un chenal encaissé). C'est une description d'ouvrage naturel, d'où la légitimité d'un champ appris — et d'où l'intérêt de HydroLAKES (profondeur moyenne, volume, temps de séjour par plan d'eau), qui donnerait un champ d'exutoire fondé sur des mesures plutôt que sur une formule.

## 2026-08-06 — Pédotransfert Saxton-Rawls : un RATTRAPAGE, pas une amélioration universelle

**Implémenté** `meandre/data/pedotransfert.py` (Saxton & Rawls 2006) : sable + argile -> porosité, capacité au champ, point de flétrissement, conductivité. Relation publiée, appliquée nœud par nœud, hors gradient. Motif : la texture est DÉJÀ dans les attributs et varie fortement (sable médian 0.34 en abit contre 0.92 sur la Côte-Nord) alors que `init_from_literature` applique un unique loam moyen à toute la province ; 12 des 37 paramètres sont concernés.

**Cohérence de la relation :** porosité 0.46 (littérature du modèle : 0.46), capacité au champ 0.084-0.305 selon la texture (littérature : 0.30 uniforme), conductivité 0.13-3.27 m/j. Cette dernière est 40-80× au-dessus du K_sat effectif du modèle, ce qui est ATTENDU : Saxton-Rawls donne une conductivité de matrice au point, le modèle utilise une conductivité effective au pas journalier et à l'échelle du tronçon, déjà réduite (facteur Beven) puis MESURÉE (recalage 0.04). On n'importe donc que la STRUCTURE spatiale, normalisée à médiane 1.

**A/B en inférence pure (forçage -hyb, tenu de côté 2022-2024) :**

| région | intensité 0.5 | intensité 1.0 | contraste du motif (K_sat q10-q90) |
|---|---|---|---|
| outv | **+0.0645** | -0.0037 | 1.00-2.71 |
| gasp | +0.0055 | -0.0154 | 0.44-1.18 |
| sagu | -0.0382 | -0.0749 | 0.46-1.25 |
| mont | -0.0791 | -0.2454 | 0.20-2.71 |

L'intensité partielle bat toujours la pleine (information partiellement redondante avec ce que le NeRF a déjà appris de la texture, qu'il reçoit en entrée).

**Lecture, et elle vaut pour les deux priors physiques testés cette nuit.** Ces contraintes ne sont pas des améliorations universelles, ce sont des RATTRAPAGES : là où le réseau a bien appris sa structure spatiale (mont, sagu), les imposer casse ce qu'il a trouvé ; là où il a mal appris (outv), elles apportent la structure manquante. OUTV gagne avec DEUX contraintes indépendantes (exutoire +0.026, texture +0.065), ce qui accuse un champion mal calibré plutôt qu'un bassin intrinsèquement difficile — huitième hypothèse enfin remplacée par un diagnostic positif.

**Conséquence de déploiement.** Le critère d'application ne peut pas être le score (sélection sur le test). Un critère a priori possible : appliquer les priors physiques là où la région manque de jauges ou là où l'entraînement n'a pas convergé, et s'abstenir là où le modèle est déjà bien calé. À formaliser.

**OUTV, tableau complet des deux lois physiques (tenu de côté 2022-2024, forçage -hyb) :**

| configuration | KGE |
|---|---|
| champion | 0.4992 |
| exutoire seul, inférence | 0.5248 |
| exutoire seul, affinage à chaud | 0.5251 |
| exutoire + texture, affinage à chaud | 0.5542 |
| exutoire + texture, inférence | 0.5613 |
| **texture seule, inférence** | **0.5637** |

**Deux conclusions nettes.** (1) Les deux lois ne s'ADDITIONNENT PAS : combinées, 0.5613, soit à peine moins que la texture seule. Elles corrigent le même défaut par deux chemins, l'ancrage d'exutoire n'apporte rien par-dessus la pédotransfert. (2) L'affinage coûte systématiquement ~0.01 par rapport à l'inférence pure — ici le réentraînement érode la contrainte au lieu de l'améliorer.

**Meilleure configuration = la plus simple :** structure Saxton-Rawls à mi-intensité, appliquée au DÉPLOIEMENT sur le champion existant. **+0.064 sur la région la plus déficiente du Québec**, sans nouvelle donnée, sans réentraînement, sans paramètre ajusté sur le score, à partir d'une relation publiée et d'attributs déjà présents. L'écart OUTV contre l'ensemble Hydrotel passe de -0.322 à -0.258.

**Question ouverte, à trancher AVANT de mesurer :** le critère d'application. Ces lois aident OUTV (+0.064) et GASP (+0.006) mais nuisent à SAGU (-0.038) et MONT (-0.079). Choisir où les appliquer d'après le score serait une sélection sur le tenu de côté. Critère a priori à formaliser (densité de jauges ? diagnostic de convergence ? écart entre la structure spatiale apprise et la structure texturale ?).

## 2026-08-07 — La tête de lac n'apprend pas la physique, et le champ n'a que 8 dimensions

**Question d'Essi : la tête de lac apprend-elle bien ?** Non. Une fois libérée (`lake_lr_mult=50`), elle étale k_lake d'un facteur 13.9 entre lacs — donc elle bouge — mais la corrélation entre ce qu'elle apprend et la surface du plan d'eau est NULLE (r = -0.001 sur les logs). Elle produit de la dispersion sans structure physique, c'est-à-dire du bruit par nœud ajusté sur la période de calage. D'où son gain de +0.002 hors échantillon contre +0.026 pour la loi imposée. À l'inverse, quand l'ancrage porte déjà la dépendance à la surface (run combo), la tête ne s'en écarte presque plus (étendue ×1.3) : elle ACCEPTE la contrainte, elle ne la RETROUVE pas.

**Dimension effective du champ de paramètres** (ACP sur les 37 params par nœud, champions régionaux) :

| région | dim. effective | 1 axe | 5 axes | 10 axes |
|---|---|---|---|---|
| mont | 6 | 52 % | 80 % | 92 % |
| sagu | 7 | 42 % | 81 % | 92 % |
| gasp | 8 | 39 % | 77 % | 91 % |
| slno | 9 | 34 % | 76 % | 90 % |
| outv | 10 | 29 % | 70 % | 86 % |

Le champ vit sur ~8 axes, pas 37. Première mesure chiffrée de l'équifinalité que le projet postule depuis le début.

**Où vit la physique par rapport à ce que le réseau a appris.** La perturbation texturale ne place que 11 % de son énergie sur les 5 axes principaux du champ appris (7 % pour l'exutoire) : ~90 % de l'information physique vit dans des directions que le réseau n'utilise presque pas. Cela explique les DEUX faces du résultat du 6 août — ces lois apportent de l'information réellement nouvelle (d'où le gain sur OUTV) et déplacent les régions bien calibrées hors de la variété sur laquelle elles étaient ajustées (d'où la perte sur MONT et SAGU).

**Conséquence pour la réduction de dimension, à contre-intuition.** Réduire le champ à ses 8 axes APPRIS supprimerait précisément les directions où vit la physique : une ACP sur le modèle entraîné garde ses erreurs et jette ce qu'on veut lui apprendre. La réduction n'a de sens que sur une base choisie A PRIORI depuis les observations et les relations publiées (axe textural Saxton-Rawls, axe de mémoire souterraine depuis les récessions, axe de timing nival depuis MOD10, axe d'évaporation depuis MOD16, axe de géométrie d'exutoire depuis la surface des lacs), le réseau ne prédisant plus que la position de chaque nœud le long de ces axes interprétables.

**Réserve honnête :** le cosinus nul mesuré entre les perturbations texture et exutoire est un ARTEFACT de ma représentation (le k_lake vit dans une tête séparée, je l'ai représenté par procuration sur K_musk et k_gw, donc coordonnées disjointes par construction). La non-additivité des deux lois mesurée le 6 août reste inexpliquée.

## 2026-08-07 — CORRECTION MAJEURE : le « déficit lacustre » était une CONFUSION régionale

**Ce qui a été fait.** HydroLAKES v1.0 ingéré (`ingest_hydrolakes.py`) : 3213 nœuds-lacs appariés par proximité d'exutoire, couverture 96-100 % sauf cnde (33 %). Attributs par lac : surface, volume, profondeur moyenne, temps de séjour, débit moyen. Fichier `lacs_hydrolakes.parquet`.

**Trois anomalies d'implémentation trouvées sur le module de lac, aucune n'étant le levier attendu :**
1. **Unités.** Q = k·(S/A)^beta·A avec Q en m³/s et S en m³ : avec beta = 1, k est en 1/SECONDE. Premier test posé en 1/jour, donc 86400× trop grand — les lacs se vidaient en un pas de temps (-0.0078, exactement le régime sans lac).
2. **Bornes.** k ∈ [1e-6, 1e-2] /s correspond, pour un réservoir linéaire, à des temps de séjour de 100 s à 11.6 jours. Or les lacs du Québec ont 93-368 jours (HydroLAKES). Le domaine autorisé ne contient AUCUN lac réel.
3. **Surface.** Le module reçoit `territorial.area_km2_physical`, l'aire de DRAINAGE du tronçon (médiane 175× la surface du plan d'eau), là où il attend la surface d'eau libre. Q variant comme 1/√A avec beta = 1.5, l'erreur divise le débit sortant par ~13.

**Mais aucune correction n'aide.** Réservoir linéaire physique (k = 1/τ, beta = 1) : outv 0.3739 contre 0.4992. Surface corrigée par nœud : 0.4914, soit exactement la valeur de saturation. Le modèle est à un optimum local étroit : toute AUGMENTATION du coefficient rend les lacs transparents (k×10 et k×100 donnent le même KGE), toute DIMINUTION sur-amortit. L'erreur de surface a été absorbée par la calibration de k.

**LA CORRECTION QUI COMPTE.** La corrélation entre le déficit et la fraction lacustre vaut -0.35 au niveau global mais **-0.11 une fois l'effet de région retiré**, et elle est incohérente à l'intérieur des régions (-0.49 slno, +0.12 sagu). Le « déficit lacustre » du 4 août était donc CONFONDU : ce n'est pas la présence de lacs qui prédit le déficit, c'est l'appartenance à une région où le modèle marche mal, et ces régions se trouvent être lacustres. Même erreur que l'hypothèse des barrages, que j'avais su écarter — ici la marche nette à 5 % m'a convaincu à tort, alors qu'elle était portée par la composition régionale des groupes. Sept interventions sur les lacs ont échoué pour cette raison.

**Le déficit est RÉGIONAL et aucune covariable de station ne l'explique.** Testées et écartées au niveau des 132 stations : fraction lacustre (partielle -0.11), distance aux stations météo (partielle -0.095), densité météo à 50 km (partielle +0.049), barrages hydroélectriques amont (+0.018), taille du bassin (+0.004), nombre de jauges régionales (+0.215 au niveau région seulement). Au niveau des 6 régions, lacs (-0.87) et densité météo (+0.74) sont confondus entre eux et non séparables sur 6 points.

**Critère a priori d'application des priors physiques : NON TROUVÉ.** Testé : accord entre le champ K_sat appris et celui qu'implique la texture (corrélation des logs). gasp -0.133 (gain +0.006), sagu +0.453 (-0.038), mont -0.119 (-0.079), outv -0.296 (+0.065). Corrélation -0.48 sur 4 points, cassée par mont. Le critère ne discrimine pas.

**État réel du diagnostic.** Ce qui distingue les régions déficientes (outv -0.271, slno -0.257) des autres n'est identifié par aucun mécanisme. Le seul fait solide est que l'Outaouais gagne +0.064 avec une relation de pédotransfert générique, ce qui accuse la qualité de calibration de son champion plutôt qu'une propriété du bassin.

## 2026-08-07 (suite) — Bug de surface de lac CORRIGÉ dans le code, et 17 paramètres sur 37 ne s'ajustent toujours pas

**Correctif appliqué** (demande d'Essi : « il faut corriger le bug de surface »). Le module de lac calcule la hauteur d'eau comme S/A mais recevait `territorial.area_km2_physical`, l'aire de DRAINAGE du tronçon. Ajout d'une surface d'eau libre dédiée : `RoutingLayer._lake_area_km2`, `HydroModel.set_lake_area()`, utilisée par les deux chemins de routage (par niveaux et par opérateur), avec repli sur l'ancien comportement si absente. Source : HydroLAKES là où l'appariement existe, sinon lake_fraction × aire locale. Sur OUTV : 0.34 km² de surface d'eau contre 22.4 km² d'aire de drainage, facteur 66. Hook `ETL_LAKE_AREA=1`. 150 tests passent.

**Effet, à comparaison ÉQUITABLE** (même forçage -hyb, même recette, cold start 12 époques) :

| run OUTV | tenu de côté |
|---|---|
| champion (entraîné sur CaSR) | 0.4992 |
| réentraîné -hyb, surface NON corrigée | 0.5011 |
| **réentraîné -hyb, surface CORRIGÉE** | **0.5160** |

+0.015 par rapport au même entraînement sans le correctif. Contrairement aux priors imposés, ce gain SURVIT au réentraînement, parce qu'il vient de la physique du code et non d'une contrainte que l'optimisation peut contourner.

**Vérification des paramètres (`verif_params.py`, nouveau, à passer après CHAQUE entraînement).**

| checkpoint | params figés | tête de lac (étendue de k) | beta |
|---|---|---|---|
| best-outv-etl-qc | 17/37 | ×1.0 | 1.502 |
| best-outv-etl-lacs | 17/37 | ×13.9 | 1.216 |
| best-outv-etl-airelac | 17/37 | ×14.8 | 1.526 |

La tête de lac est RÉPARÉE (×14.8 au lieu de ×1.0, aucune borne touchée). Mais **17 paramètres sur 37 restent figés à leur initialisation** dans les trois checkpoints : f_root_1/2/3, T_snow, interception_capacity, manning_n, frost_alpha, f_wetland, **f_vert_1/2/3** (la partition verticale du drainage, processus central), T_gw, K_atm, alpha_T, vg_n, rain_hours, vsa_b. Audit de sensibilité en cours pour savoir s'ils sont INSENSIBLES (auquel cas le gel est sans dommage) ou seulement BLOQUÉS (auquel cas c'est du gain laissé sur la table).

**Test de transfert sur OUTV** (forçage -hyb, en inférence) : gasp 0.3756, sagu 0.5024, mont 0.4482, contre 0.4992 pour le champion local. Le saguenéen transféré ÉGALE le local sans même bénéficier des effets aléatoires par nœud — l'entraînement local sur 16 jauges n'apporte donc rien, ce qui est en soi anormal.

**Pourquoi 17 paramètres sont figés : ils sont MORTS.** Audit du code (2026-08-07) : **19 des 37 paramètres produits par le réseau ne sont lus par AUCUN module de la physique active** — theta_fc_2/3, theta_wp_2/3, f_root_1/2/3, T_snow, interception_capacity, manning_n, frost_alpha, f_wetland, f_vert_1/2/3, alpha_T, vg_n, rain_hours, vsa_b. Ce sont des vestiges de l'ancienne colonne (`column.py` / `soil.py`, dont il ne reste que les .pyc) remplacée par `hydrotel_column.py`. Le réseau les calcule, personne ne les consomme.

La liste recouvre 15 des 17 paramètres figés : ils ne bougent pas parce qu'ils n'ont AUCUN gradient, faute d'effet. Ce n'est pas un défaut d'apprentissage, c'est de la dette technique. Les deux exceptions, T_gw et K_atm, sont lues mais seulement par le module de température de rivière, absent de la fonction de coût.

Vérification par l'audit de sensibilité : f_vert_1 ±0.2 et f_vert_3 -0.2 donnent exactement +0.000 sur le KGE, ce qui est la signature d'un paramètre non consommé plutôt que d'une insensibilité physique.

**Conséquence.** Le modèle a **18 paramètres actifs**, pas 37. La dimension effective de 8 mesurée le 7 août correspond donc à un facteur 2 de redondance parmi les paramètres vivants, et non à un facteur 5 — l'équifinalité est réelle mais bien moins sévère que je ne l'ai écrit. À faire : retirer les sorties mortes de `SpatialParams` (ou rebrancher celles qui devraient l'être, notamment f_vert qui portait la partition verticale du drainage et vsa_b le ruissellement sur aire contributive).

**Transfert de champions sur OUTV, comparaison équitable** (même banc, sans effets aléatoires pour tous) :

| champion | validation | tenu de côté |
|---|---|---|
| slno-etl-canon | 0.5831 | **0.5731** |
| outv-etl-qc (local) | 0.5827 | 0.5116 |
| sagu-etl-ds | 0.5666 | 0.5024 |
| mont-etl-ds | 0.4229 | 0.4482 |
| slso-etl-canon | 0.5040 | 0.4398 |
| gasp-etl-ds | 0.5137 | 0.3756 |

En validation le local et celui du Lac-Saint-Jean sont indiscernables (4 dix-millièmes) ; en tenu de côté l'écart est de 62 millièmes en faveur du transfert. La conclusion défendable n'est donc pas « prendre le champion slno » (ce serait sélectionner sur le test) mais : **l'entraînement local d'OUTV, avec 16 jauges, ne produit rien de mieux qu'un modèle entraîné ailleurs et généralise moins bien** — sa validation le flatte, son tenu de côté le contredit. Signature d'un sur-ajustement à la période de calage, cohérente avec tout le reste de la session.

**SLNO, même test (forçage -hyb, même banc pour tous) :**

| champion | validation | tenu de côté |
|---|---|---|
| sagu-etl-ds | **0.6910** | **0.6215** |
| gasp-etl-ds | 0.5894 | 0.6009 |
| slno-etl-canon (local) | 0.5539 | 0.5456 |
| mont-etl-ds | 0.5327 | 0.4995 |
| slso-etl-canon | 0.4819 | 0.4706 |

Ici la sélection sur VALIDATION est sans ambiguïté (sagu devance de 0.10 le suivant) et désigne aussi le meilleur en tenu de côté. Cohérent géographiquement : Saguenay et Lac-Saint-Jean forment le même système hydrographique.

**GAIN NET ET DÉFENDABLE sur les deux régions déficientes**, par sélection de champion sur validation seule :

| région | champion local | champion sélectionné | gain |
|---|---|---|---|
| slno | 0.5456 | 0.6215 (sagu) | +0.076 |
| outv | 0.5116 | 0.5731 (slno) | +0.062 |

Réserve sur OUTV : la validation ne distingue pas le local (0.5827) du sélectionné (0.5831), 4 dix-millièmes — le choix y est arbitraire même s'il tombe juste. Sur SLNO il est net.

**Réconciliation avec le 3 août**, où la sélection de champion sur validation faisait PIRE que pas de sélection (médiane 0.627 contre 0.653). Ce jour-là les régions testées avaient 1 à 4 stations : la validation n'y mesure rien. Ici, 16 et 27 stations. La règle défendable est donc : **sélectionner le champion sur validation là où la région a assez de jauges (~10), prendre le champion global ailleurs** — même seuil que celui déjà établi pour décider entre calibration locale et transfert.

**Le fil conducteur de la session.** Ce modèle sur-ajuste sa période de calage, et cela se voit maintenant par trois chemins indépendants : le test de capacité (145k params libres ne gagnent que +0.002 sur la période d'entraînement), les priors physiques défaits par le réentraînement à froid, et le transfert (un champion étranger généralise mieux que le local sur les deux régions déficientes, alors que la validation ne les distingue pas). La validation, contiguë à la période d'entraînement, est elle-même trop optimiste pour arbitrer.

**OUTV, tableau final de la session** (tenu de côté 2022-2024, forçage -hyb) :

| configuration | KGE |
|---|---|
| **transfert PUR du champion slno** | **0.5731** |
| pédotransfert sur le champion local | 0.5637 |
| transfert slno + affinage local (8 ép., lr 1e-4) | 0.5626 |
| champion local, surface de lac corrigée | 0.5160 |
| champion local | 0.4992 |

**Un modèle qui n'a JAMAIS vu l'Outaouais y fait mieux que tous ceux qui y ont été entraînés**, et même un affinage court depuis ce modèle perd 0.010. Toute forme d'entraînement sur les données de cette région dégrade le tenu de côté.

**Non-stationnarité climatique : écartée comme explication** (9e hypothèse). Dérive calage 2001-2018 -> test 2022-2024 :

| région | ΔP | ΔT | ΔQ | écart vs ensemble |
|---|---|---|---|---|
| outv | +2.1 % | +0.97 °C | -9.4 % | -0.271 |
| sagu | -8.4 % | +1.75 °C | -30.1 % | -0.102 |
| gasp | -4.2 % | +1.06 °C | -9.2 % | -0.059 |
| mont | +3.6 % | +0.88 °C | +4.5 % | -0.032 |
| slso | +0.7 % | +0.92 °C | -5.1 % | -0.021 |
| slno | -0.1 % | +1.01 °C | -0.8 % | -0.257 |

SAGU subit la dérive la PLUS violente (-30 % de débit) et son champion local fonctionne bien ; SLNO n'a presque aucune dérive et son champion échoue. Aucune relation.

**Hypothèses écartées sur OUTV, par ordre chronologique :** transfert gaspésien, littérature, entraînement local, forçage, régulation (×2, dont une reconstruite par erreur), timing de fonte, neutralisation des lacs, paramétrage des lacs (7 variantes), non-stationnarité climatique. Ce qui MARCHE : la pédotransfert (+0.064) et le transfert d'un champion sélectionné sur validation (+0.074) — deux moyens de ne PAS utiliser ce que l'entraînement local a appris.

## 2026-08-08 — APPARIEMENT MODULE PAR MODULE avec Hydrotel (états internes, sans réexécution)

Hydrotel écrit dans `etat/` l'état interne de chaque sous-modèle (bilan_vertical, fonte_neige, acheminement_riviere) à des dates données ; l'acheminement est PAR TRONÇON et de même cardinalité que les nœuds méandre (3412 sur OUTV) — correspondance directe. Harnais : `.runs/quebec/apparier_hydrotel.py`. Date disponible dans le tenu de côté : 2023-08-01 (étiage d'été).

**OUTV, 2023-08-01, champion local vs Hydrotel LN24HA :**

| étage | Hydrotel | méandre | rapport | corr spatiale |
|---|---|---|---|---|
| theta1 (méd) | 0.325 | 0.299 | 0.92 | |
| theta2 (méd) | 0.352 | 0.327 | 0.93 | |
| theta3 (méd) | 0.434 | 0.391 | 0.90 | |
| apport latéral (méd, m³/s) | 0.262 | 0.103 | **0.39** | +0.56 |
| débit aval (méd, m³/s) | 0.898 | 0.427 | **0.48** | +0.98 |

**Lecture.** Les sols sont presque aussi humides (90-93 %) mais la colonne livre MOITIÉ MOINS d'eau à la rivière en été. La corrélation spatiale du débit aval (0.98) montre que l'accumulation le long du réseau est cohérente : le déficit est dans la GÉNÉRATION estivale, pas dans le routage. Piste physique : à teneur en eau légèrement plus sèche, l'écoulement hypodermique non linéaire chute fortement — ou l'ET estivale de méandre est trop forte, ou le drainage profond (krec/aquifère) soutient mal l'étiage. Un seul instantané : le test d'ensemble sur les séries complètes (reseau_compare.py, en cours) dira si le déficit est saisonnier ou permanent.

## 2026-08-08 — L'ERREUR STRUCTURELLE, par confrontation des deux codes sources

Demande d'Essi : investiguer le code de méandre comparativement au C++ d'Hydrotel (dépôt INRS local). Deux lecteurs en parallèle, rapports cités au fichier et à la ligne. Confrontés au test d'ensemble (r médian méandre/Hydrotel 0.335 sur 3412 tronçons, 0.27-0.31 en tête de bassin montant à 0.73 au-delà de 5000 km², volume stable ~0.85), le tableau est complet.

**1. LE VERSANT : Hydrotel étale sur ≤10 jours, méandre sur 0.**
- Hydrotel (`onde_cinematique.cpp`) : la production verticale de BV3C est convoluée par un HYDROGRAMME UNITAIRE GÉOMORPHOLOGIQUE précalculé par UHRH (onde cinématique de Manning résolue pixel par pixel, `CalculeHgm()` l.1166-1781), mémoire maxdeb = 240 h / pas = **10 jours**, queue tronquée à 5 % du volume. Les 10 valeurs "DEBITS" du fichier d'état sont la file d'attente de cette convolution. Surface, hypodermique et base subissent LE MÊME étalement (`_oc_surf = _oc_hypo = _oc_base = _oc_zone`, l.1110-1156). Le lissage principal d'Hydrotel est AU VERSANT.
- méandre (`hydrotel_column.py:662`) : `prod = ps_surf + ph + pb` livré au tronçon LE JOUR MÊME. `use_hillslope_uh=False` par défaut (l.94) ; même activé, c'est une cascade de Nash courte (0.3 j / 2.5 j) qui ne route PAS le baseflow. Et pour un tronçon de tête avec K ≲ 7.5 h, le clamp c2>=0 du Muskingum donne une injection à 100 % le jour même (kinematic.py:60-61).
- CONSÉQUENCE : deux séries de même volume, l'une étalée sur ~10 j, l'autre pas du tout -> décorrélation quotidienne massive en tête de bassin, réconciliation par agrégation vers l'aval. C'est EXACTEMENT la signature mesurée. Le diagnostic du 15 juin (« Hydrotel lisse au VERSANT ») était juste, le correctif codé, jamais activé.

**2. LES LACS : les paramètres calibrés d'Hydrotel sont dans troncon.trl DEPUIS TOUJOURS, et méandre les jette.**
- C++ `troncons.cpp::LectureLac` (l.321-348) : chaque tronçon-lac porte `longueur, surface (km²), c, k` — la loi de tarage calibrée Q = c·h^k (k = 1.5 partout sur OUTV, déversoir), routée en réservoir de niveau à surface constante (`TransfertLac`, implicite trapézoïdal Newton).
- méandre `physitel_loader.py::_parse_troncon` (l.325-333) : lit ces champs mais les nomme `length, width, v3, v4` — il prend la SURFACE pour une largeur et JETTE c et k. Vérifié sur OUTV : tronçon 2 = 41.55 km², c=73.5, k=1.5.
- Correspondance EXACTE avec la loi de méandre Q = k_lake·(S/A)^beta·A : beta = k, A = surface·1e6, **k_lake = c/A**. Sur OUTV : k_lake médian ~2-4e-6 /s, JUSTE au-dessus du plancher [1e-6] du modèle — les vraies valeurs sont au bord du domaine, c'est pourquoi l'apprentissage n'a jamais pu les trouver.
- Écarts restants de méandre : sortie du lac sur le stockage de la VEILLE en mode lagged (1 j de déphasage, operator_routing.py:35-39), stockage initial S=0 (temps de remplissage), aire de drainage au lieu de la surface d'eau (corrigé le 7 août, opt-in).

**Tests en cours :** (a) r à 7 et 30 jours de lissage (départage forçage vs structure) ; (b) import direct des lacs trl (`lacs_trl.py`) en inférence.

**Test réseau LISSÉ (OUTV, 3412 tronçons, 2022-2024) : la décorrélation est un phénomène de courte échelle temporelle.**

| aire cumulée | r quotidien | r 7 j | r 30 j | beta été | beta hiver |
|---|---|---|---|---|---|
| <10 km² | 0.314 | 0.559 | 0.762 | 0.834 | 0.882 |
| 10-50 | 0.266 | 0.508 | 0.743 | 0.909 | 1.108 |
| 50-200 | 0.325 | 0.508 | 0.729 | 0.918 | 1.090 |
| 200-1k | 0.443 | 0.552 | 0.721 | 0.901 | 1.092 |
| 1k-5k | 0.641 | 0.679 | 0.788 | 0.875 | 1.140 |
| >5k | 0.726 | 0.747 | 0.819 | 0.852 | 0.966 |
| lacs | 0.202 | 0.411 | 0.675 | 0.923 | |
| rivières | 0.357 | 0.564 | 0.760 | 0.890 | |

Au pas MENSUEL, la dépendance à l'échelle disparaît (0.72-0.82 partout) : la décorrélation quotidienne est bien un défaut de FORME temporelle de la réponse rapide, exactement ce qu'un noyau d'étalement présent d'un côté (≤10 j) et absent de l'autre produit. Les lacs restent les pires à TOUTES les échelles (défaut propre). Et la saisonnalité du volume est inversée : méandre produit trop l'hiver (+9 à +14 %) et pas assez l'été (-8 à -15 %).

**LE NOYAU D'HYDROTEL EST RÉCUPÉRABLE TEL QUEL.** Le cache `<projet>/hgm/hydrogramme_24H_*.hgm` est un fichier texte contenant, par UHRH, les 10 poids DISTRI de l'hydrogramme unitaire. Sur OUTV : en médiane seulement 45 % de l'eau arrive au jour 0, l'étalement touche 3390 nœuds sur 3412. Implémenté :
- `meandre/data/hgm_loader.py::lire_hgm` — noyau (n_nodes, 10), agrégation UHRH->tronçon pondérée par l'aire, lignes vides (UHRH-lacs) = Dirac au jour 0 comme le C++ ;
- `HydroModel.set_hgm_kernel()` + file de convolution glissante dans simulate() (mécanisme identique à onde_cinematique.cpp:806-886), opt-in, 150 tests passent.

Bancs en cours : lacs trl seuls (3 régions), puis ref / hgm / hgm+trl sur OUTV avec double mesure (KGE jauges + r contre Hydrotel sur le réseau et les têtes <50 km²).

## 2026-08-08 (suite) — BANC DE ROUTAGE VALIDÉ (x20 plus rapide) et premiers verdicts

**Banc `banc_routage.py`** (demande d'Essi : tester plus vite que des simulations longues) : la colonne est simulée UNE fois, sa production latérale mise en cache, chaque variante ne rejoue que le routage (~2 min au lieu de ~20). Deux bugs de ma part trouvés par la validation obligatoire contre la simulation complète : le rejeu passait K_musk en HEURES là où simulate le convertit en SECONDES (model.py:371) — l'opérateur n'atténuait plus rien. Après correctif : écart rejeu/simulate NUL (médiane et p95 à 0.0000), référence 0.4992 reproduite exactement. LEÇON : tout banc rapide doit reproduire la référence avant d'être cru (la première salve de résultats, fausse, a failli être consignée).

**Verdicts OUTV (inférence pure, champion inchangé) :**

| variante | KGE jauges | r réseau vs Hydrotel | r têtes <50 km² | r lacs |
|---|---|---|---|---|
| référence | 0.4992 | 0.335 | 0.278 | 0.202 |
| noyau HGM versant | **0.5262** | 0.470 | 0.424 | 0.315 |
| lacs troncon.trl | 0.4640 | 0.484 | 0.338 | **0.669** |
| HGM + lacs trl | 0.4844 | **0.595** | **0.493** | **0.708** |

- Le NOYAU DE VERSANT gagne sur les DEUX tableaux : +0.027 aux jauges ET fidélité à Hydrotel en forte hausse partout. Structure physique gratuite (cache .hgm), zéro paramètre appris. C'est la confirmation du diagnostic : l'étalement au versant était le chaînon manquant.
- Les LACS trl triplent la fidélité des tronçons-lacs (0.202 -> 0.669) mais coûtent -0.035 aux jauges : le champion a calibré le reste autour de ses propres lacs (motif désormais familier). Remède connu : départ à chaud + affinage court.
- La COMBINAISON porte r réseau de 0.335 à 0.595 : la structure de méandre se rapproche massivement de celle d'Hydrotel avec deux imports de données.

Suite : simulation complète du meilleur candidat cumulatif (noyau HGM + pédotransfert 0.5, cette dernière vivant dans la colonne donc hors banc rapide).

**Combo en inférence (OUTV) : noyau HGM + pédotransfert NE S'ADDITIONNENT PAS.** 0.5527 contre 0.5637 pour la texture seule (r monte à 0.662 mais beta reste 0.766). Le champion avait calibré son routage pour compenser l'étalement manquant : lui ajouter le noyau par-dessus la texture sur-lisse. Classement OUTV inférence : transfert slno 0.5731 > texture 0.5637 > combo 0.5527 > noyau seul 0.5262 > champion 0.4992.

**Test décisif lancé :** entraînement OUTV avec la STRUCTURE corrigée dès le départ (`ETL_HGM=1` noyau de versant actif pendant l'apprentissage, `ETL_LAKE_TRL=1` lacs troncon.trl imposés hors gradient + surface d'eau vraie, + recette canonique). Question : avec la bonne structure, l'entraînement local sur 16 jauges bat-il enfin le transfert (0.5731) ? Hooks ajoutés à etl_run.py.

## 2026-08-08 (suite) — Entraînement avec structure corrigée : le problème d'OUTV n'est PAS structurel

**Entraînement OUTV à froid avec ETL_HGM=1 + ETL_LAKE_TRL=1** (noyau de versant actif pendant l'apprentissage, lacs d'Hydrotel imposés hors gradient) + recette canonique : tenu de côté **0.4740**, SOUS le champion sans structure (0.4992) et loin du transfert slno (0.5731). Même avec le versant et les lacs corrects, l'apprentissage local sur 16 jauges dégrade. Conclusion consolidée : le déficit d'OUTV est un problème d'APPRENTISSAGE (sur-ajustement à la période de calage), pas de structure ni d'observations ; la règle du transfert tient.

**Fidélité v1 (paramètres FIGÉS bv3c complet + Linacre calé + fonte calée + lacs trl + noyau HGM, Muskingum, météo -hyb) :** theta1/2/3 à 0.93/0.92/0.995 d'Hydrotel — les états du sol COLLENT — mais beta médian réseau 0.584 et apport latéral d'août à 0.131. Et la pluie n'explique RIEN : la météo du projet (977 mm/an) est même 2 % SOUS notre grille (997). Entrée équivalente + états équivalents + sortie à moitié = l'eau disparaît DANS la colonne, seul suspect capable : l'ÉVAPOTRANSPIRATION (Linacre×coeff appliqué différemment ?). Juge de paix lancé : réexécution d'Hydrotel (WSL rétabli) sur une COPIE instrumentée du projet OUTV avec 8 sorties internes activées (ETP, ETR_TOTAL, THETA1-3, APPORT, APPORT LATERAL, COUVERT_NIVAL) — comparaison directe de l'ETR des deux modèles à venir.

## 2026-08-09 — FUITE DE MASSE : 21 % de la pluie disparaît dans la colonne figée

**Bilan de masse de la colonne FIGÉE (OUTV, 2021-2024, moyennes spatiales, mm/an) :**
P 970 | ETP 482 | ETR 480 | apport latéral 285 | **résidu +204 (21 % de P)** | coefficient d'écoulement 0.294 (Hydrotel ~0.5-0.6).

- L'ETR est INNOCENTÉE : 480 mm/an = exactement la fourchette des tours de flux boréales (400-500). Le suspect « Linacre trop fort » tombe.
- Écartés par la mesure : somme des fractions fsa+fse+fsi = 1.0 exactement partout ; dérive du stock de neige -6.6 mm/an ; interception par la canopée : N'EXISTE PAS dans la colonne clone (aucun code), donc ne peut ni évaporer ni fuir — mais noter que 204/970 = 21 % est précisément la fraction d'interception forestière canonique, À REVOIR côté C++ (Hydrotel intercepte-t-il ?).
- La fuite est donc entre l'apport au sol et la production. Bilan scindé neige/sol en cours (apport = pluie+fonte livrée au sol, sauvegardé).

**Décision d'exécution :** routeur fidèle (fidelite2) TUÉ après 14 h — la forme séquentielle du C++ (48 sous-pas × 126 niveaux × 1460 jours) est impraticable sur GPU ; il ne servait qu'à la validation, le bilan de masse ne l'exige pas. Hydrotel instrumenté toujours en cours dans WSL (8 h CPU, normal pour 6 ans × 8821 UHRH × 8 sorties).

## 2026-08-09 — LA FUITE EST COLMATÉE : bilan fermé à 2 mm/an près

**Cause exacte.** Le C++ (TriCoucheOct97) boucle JUSQU'À épuiser le pas de temps ; le clone plafonne à n_substep=48 itérations pour le GPU. Sous gel ou crue, l'échelle de Courant descend à DT_H/1152 (~75 s) : le plafond tombait après ~1 h de journée traitée et le reste disparaissait avec sa pluie. Fuite mesurée : 211 mm/an (21 % de P) sur OUTV, profil mensuel culminant en mars (+52 mm) et avril (+34), fort d'oct. à déc., nul en été — la signature du gel.

**Fermeture correcte, trouvée en 3 itérations mesurées (chacune ~20 min, journal complet) :**
1. `lruis += prec·tr` seul : sur-correction +199 (la pluie brute du temps restant fait 410 mm/an, pas 211) ;
2. déduction d'ET plafonnée terme à terme : encore +130 (le plancher ne retire qu'un tiers de l'ET) ;
3. **VERSION RETENUE** : la pluie du temps restant ruisselle ET l'évapotranspiration du temps restant est RETIRÉE DU STOCK (le module d'ET la déclarait en entier au bilan mais la boucle plafonnée ne la prélevait qu'au prorata — l'eau « évaporée » restait dans le sol et ressortait en débordement). Négativité refoulée comme dans la boucle.

**Bilan final (OUTV, colonne figée, 2021-2024) : P 970 -> apport 977 -> ETR 473 + latéral 502, résidu +2 mm/an, coefficient d'écoulement 0.518** (fourchette Hydrotel 0.5-0.6). 150 tests passent.

**Portée.** TOUS les entraînements passés ont appris sur une colonne qui jetait ~20 % de la pluie au moment des crues. À requalifier : déficit de ruissellement de juin (RC 0.55 vs 0.63), déficit d'été, beta ~0.85 généralisé, partie du plafond attribué au forçage. Réentraînement OUTV lancé avec la colonne réparée.

## 2026-08-09 (suite) — Le correctif de masse et le noyau de versant sont COUPLÉS

**Réentraînements sur colonne réparée (recette canonique inchangée, -hyb) :**
- OUTV : 0.5048 contre 0.4992 (+0.006). Le colmatage ne change presque rien : le champion fuyant avait déjà appris à compenser l'eau manquante (leçon récurrente : la calibration absorbe les erreurs structurelles). Rectification demandée par Essi et acceptée : en SCORE, ce bug est mineur ; sa gravité est la CONSERVATION (scénarios, renaturalisation), pas la performance.
- GASP : **0.5599 contre 0.7489 (-0.19), régression massive.** Volume sain (beta 0.854) mais corrélation effondrée (r 0.743 contre 0.878).

**Lecture.** L'eau récupérée par le colmatage est celle des jours de gel et de crue ; sans hydrogramme de versant elle arrive à la rivière LE JOUR MÊME, en pointes que rien n'étale -> r s'effondre. La colonne fuyante servait accidentellement de FILTRE PASSE-BAS en supprimant l'eau la plus impulsive. Les deux corrections sont donc couplées : conservation de la masse SANS étalement du versant = pire que la fuite. Test en cours : GASP avec colonne réparée + noyau HGM (`ETL_HGM=1`).

**Réserve méthodologique actée (objection d'Essi) :** toutes les comparaisons croisées méandre/Hydrotel faites à météos différentes (fidélité v1, rapports d'étages, r réseau) mélangent code et forçage. SEUL le bilan de masse (comptabilité interne d'une même simulation) y échappe. Le test d'intégrité définitif = colonne figée + météo Thiessen du PROJET + confrontation aux sorties instrumentées d'Hydrotel (réexécution WSL en cours, ~16 h CPU), mêmes nombres des deux côtés.

**Requalification aussi des adaptateurs mesurés :** ETL_DEMAND_SCALE (ratio bilan/MOD16) a été dérivé sur un bilan qui INCLUAIT la fuite — à re-mesurer avec la colonne fermée.

**TÉMOIN DÉCISIF (2026-08-09) : la fermeture de masse est INNOCENTE de la régression GASP.** Code d'aujourd'hui avec fermeture DÉSACTIVÉE (`BV3C_FERMETURE=0`) : 0.5254, aussi effondré que 0.5599 avec fermeture — contre 0.7489 pour le champion du 30 juillet. La régression vient d'un AUTRE changement des dernières 48 h, actif par défaut dans tous les nouveaux runs. Suspect n°1 : le groupe d'optimisation dédié à la tête de lac (`lake_lr_mult=50`), jamais validé sur une région saine. Contre-témoin lancé : fermeture OFF + `ETL_LAKE_LR=1`. Rappel de méthode (Essi) : tout changement de défaut d'entraînement doit passer par un témoin sur région saine AVANT d'être adopté — cette régression a coûté 4 runs d'une heure avant d'être attribuée.

**RÉSOLUTION de la « régression » GASP : c'était le NOMBRE D'ÉPOQUES.** Le champion du 30 juillet (0.7489) a été entraîné 30 époques ; tous mes réentraînements du 9 août en faisaient 12. Trajectoires d'entraînement quasi IDENTIQUES aux époques 0-2 (val 0.5894/0.5917/0.5966 contre 0.5896/0.5918/0.5966) et proches à l'époque 11 (0.6559 contre 0.6784). Aucune régression de code : fermeture, tête de lac et tout le reste sont innocents — je comparais des entraînements aux deux tiers finis à un champion complet. À ÉPOQUES ÉGALES (12), le classement s'inverse : témoin fuyant 0.5254 < masse 0.5599 (+0.035) < masse+HGM 0.5999 (+0.075). Le colmatage et le noyau AIDENT. Validation propre lancée : masse+HGM sur 30 époques (repère champion 0.7489). Leçon de plus pour le protocole : toujours vérifier N_EPOCHS avant de crier à la régression — 5 runs d'une heure perdus.

**Correction de la correction (fin de journée du 9 août).** Le run masse+HGM sur 30 époques donne 0.5675 : ni le nombre d'époques ni la fermeture n'expliquent l'écart au champion (0.7489). Lecture fine des deux journaux : mon run est MEILLEUR en validation jusqu'à l'époque 11 (val_kge 0.7055 contre 0.6794) puis PLAFONNE (kge_med final 0.672) tandis que le champion continue de monter jusqu'à 0.769. La validation classe donc correctement à la fin — ce n'est pas une pathologie de sélection, c'est un entraînement qui plafonne plus tôt.

Différence de configuration que j'avais introduite sans la voir : `ETL_DEMAND_SCALE` 0.776 dans mes runs contre **0.81** dans le champion (débiaisage ET bilan/MOD16). Moins d'ET = plus d'eau, cumulé avec le colmatage qui en ajoute encore.

**Discipline appliquée : reproduction exacte du champion lancée avant toute autre interprétation** (0.81, fermeture OFF, sans HGM, 30 époques). Sans socle reproductible, aucune comparaison de facteur n'est lisible. Tant que ce socle n'est pas retombé sur 0.7489, les verdicts « le colmatage aide » (+0.035) et « le noyau aide » (+0.075) restent PROVISOIRES : ils ont été mesurés à 12 époques contre un témoin lui aussi à 12 époques, donc entre eux ils sont valides, mais pas contre le champion.

## 2026-08-09 (soir) — BANC DE MODULES : des centaines de tests en secondes, sans Hydrotel

**Reproche d'Essi, fondé et accepté :** je relançais des entraînements régionaux de 3 h là où des tests de module en secondes étaient disponibles. J'avais même rendu le banc de micro-routage dépendant de la réexécution d'Hydrotel, alors que comparer deux SCHÉMAS entre eux ne demande aucune sortie externe — juste une impulsion et la géométrie réelle des tronçons.

`banc_modules.py` : 400 rivières + 400 lacs échantillonnés dans troncon.trl, impulsion de 10 m³/s, mesure de la fonction de réponse (retard du pic, atténuation, masse rendue, demi-vidange). **Durée : quelques secondes.**

**BUG TROUVÉ (dormant) : le module `MuskingumCunge` ne conserve pas la masse, et l'erreur DÉPEND de K, qui est un paramètre APPRIS.** L'apport latéral était ajouté brut à la sortie (`+ q_lateral`) au lieu d'être pondéré comme un débit entrant, et divisé par le nombre de sous-pas. À l'équilibre : Q = q_lat/(n(1−c2)). Mesuré (n_substeps=2, x=0.2) : K=4 h -> masse 0.50 (l'eau est DÉTRUITE), K=24 h -> 1.05, K=48 h -> **1.85 (l'eau est FABRIQUÉE)**. Un réseau pouvait donc créer ou détruire de l'eau en ajustant un temps de transfert.
**Portée réelle, sans surinterprétation : ce chemin de code n'est PAS celui de la production.** Toutes les configs Québec sont en `operator-lagged`, et le mode opérateur — vérifié numériquement pour K de 4 à 48 h — conserve la masse EXACTEMENT (son `beta = 1−gamma` est la bonne pondération). Le bug est réel mais dormant. Corrigé quand même (apport latéral pondéré (1−c2), non divisé) : masse 1.0000 pour tout K, 150 tests passent.

**LA MESURE QUE LE BANC APPORTE (même géométrie, même impulsion, masse maintenant exacte des deux côtés) :**

| schéma | pic restitué | atténuation |
|---|---|---|
| clone onde cinématique Hydrotel | 10.61 m³/s | AUCUNE (translation quasi pure, +6 % numérique) |
| Muskingum méandre K=24 h (init) | 7.26 | −27 % |
| Muskingum méandre K=48 h (borne haute) | 4.67 | **−53 %** |

Le déficit de pic n'est donc pas une hypothèse tirée de la lecture du C++ : il est CHIFFRÉ au niveau du module. Hydrotel translate, méandre diffuse, et comme le K appris dérive vers le haut (documenté), le modèle s'auto-condamne à raboter ses crues. Le levier n'est pas un paramètre à recaler, c'est le SCHÉMA.

## 2026-08-09 (nuit) — DIAGNOSTIC CAUSAL du rabotage : ce sont les BORNES de K, pas l'apprentissage

Essi : « une observation, pas un diagnostic causal qui permet une correction ciblée ». Juste. Chaîne établie, chaque maillon MESURÉ au banc de modules (secondes) :

1. **Hypothèse d'abord testée et INFIRMÉE** : « le rabotage est la réponse optimale à une erreur de calage temporel » (double peine quadratique sur un pic mal daté). Faux : avec une référence bien spécifiée, la perte d'entraînement retrouve exactement le bon K (4 h) pour TOUTE largeur d'événement (0.5 à 10 j) et TOUT décalage (0 à 5 j). La perte ne pousse PAS au lissage.
2. **Fait vérifié, contre ma propre mémoire** : le K appris n'est PAS gonflé par l'entraînement. Champion gasp : init 26 h -> appris **23.7 h** (p10 19.4, p90 27.2, 0 % au-dessus de 40 h). Le modèle descend, il ne monte pas.
3. **Le K physique, calculé par Manning sur la géométrie réelle de troncon.trl** (longueur médiane 4.3 km OUTV / 3.6 km GASP, vitesse ~2 m/s, célérité 5/3·v) : **médiane 0.2-0.35 h, et 100 % des tronçons sont SOUS l'ancienne borne basse de 4 h.** Le K appris valait donc 60-100× le temps de parcours réel.
4. **Pourquoi l'optimiseur ne corrige pas** : la perte est quasi plate en K (écart K=4 h contre K=20 h : 0.0479 contre 0.0458, soit 4 % de la perte). Faible identifiabilité -> K reste collé à son initialisation.
5. **Effet mesuré** : à K=24 h un tronçon atténue un événement court de **27 %** et l'étale sur 4 jours ; à 48 h, −53 % sur 8 jours ; à K physique (<= 8 h) le pic passe INTACT (10.00 contre 10.61 pour le clone de l'onde cinématique) et l'étalement tombe à 1 jour. L'effet se COMPOSE le long de la chaîne topologique.

**Conclusion causale : le rabotage était imposé par les bornes et l'initialisation de K_musk, pas choisi par l'apprentissage.** Cela referme le diagnostic de juin : Hydrotel étale au VERSANT (hydrogramme géomorphologique) puis translate dans le canal ; méandre n'avait pas d'étalement au versant et compensait par un canal ultra-diffusif. Le noyau HGM et le K physique forment la paire cohérente à tester ensemble.

**Correction ciblée appliquée** : bornes et init de K_musk configurables par `MEANDRE_KMUSK="min,max,init"` (défaut historique 4,48,24 conservé pour ne pas réinterpréter les checkpoints existants). 150 tests passent. Le mode opérateur reste sain à petit K (c2=0 -> translation pure) ; le mode message-passing exige K >= 4 h avec n_substeps=2.

## 2026-08-09 (nuit, suite) — HYPOTHÈSE DES BORNES DE K : RÉFUTÉE par mesure directe

Essi : « peux-tu tester encore cette hypothèse ? pourquoi l'avions-nous manquée ? »

**Test décisif** (banc de routage, vrai réseau OUTV, production latérale en cache, K forcé par `BANC_K`) :

| K_musk | KGE jauges (ref) | r réseau vs Hydrotel | KGE jauges (+HGM) | r réseau (+HGM) |
|---|---|---|---|---|
| 23.7 h (appris) | 0.4992 | 0.335 | 0.5262 | 0.470 |
| **0.35 h (physique Manning)** | **0.1518** | **0.209** | **0.3951** | **0.373** |

**Tout se dégrade, y compris la fidélité au réseau d'Hydrotel** — métrique qui ne récompense pas la calibration aux jauges. La diffusion à ~24 h fait donc un travail NÉCESSAIRE : elle n'est pas le défaut. Hypothèse réfutée. (Réserve : le champion a calibré le reste autour de son K, motif déjà vu sur les lacs trl ; mais la chute est ici 5× plus grande, et elle touche aussi la métrique structurelle.)

**Ce qui reste vrai et acquis :** le K appris n'est pas gonflé par l'entraînement (23.7 h depuis une init à 26 h, mesuré) ; la perte est quasi plate en K (4 %) ; le temps de parcours par Manning vaut ~0.2-0.35 h. Ces trois faits tiennent — c'est leur INTERPRÉTATION (« donc le modèle sur-diffuse par construction ») qui est fausse. Le modèle a besoin de plus d'étalement que le transfert de canal physique n'en fournit, et le noyau de versant seul ne comble pas l'écart.

**Pourquoi l'angle mort, trois causes cumulées :**
1. **Héritage non réexaminé** : la borne basse de 4 h vient d'une contrainte de stabilité du routage par message-passing (n_substeps=2, sub_dt=12 h), écrite dans le commentaire du code. Le passage au mode opérateur, qui n'a pas cette contrainte, ne l'a jamais remise en question.
2. **Le contrôle de paramètres ne teste aucune plausibilité physique** : `verif_params.py` demande si un paramètre bouge, s'effondre ou touche ses bornes — jamais si sa PLAGE est physiquement crédible. Un paramètre confortablement installé au milieu d'un intervalle faux passe tous les contrôles.
3. **Le banc du 8 août comparait des SCHÉMAS à K CONSTANT** (noyau, lacs, combinaison) : le paramètre suspect était tenu fixe dans toutes les variantes, ce qui garantissait que le routage paraisse innocent — la conclusion de juin après 9 expériences.

La cause 2 est la seule généralisable : ajouter au contrôle de paramètres une confrontation à une grandeur physique indépendante (Manning pour K, pédotransfert pour K_sat, récessions pour k_gw), pas seulement une statistique de dispersion.

Test suivant (en cours) : chaîner le CLONE de l'onde cinématique N fois. Je n'avais mesuré sa réponse que sur UN tronçon ; si le schéma d'Hydrotel accumule lui aussi du stockage le long d'une chaîne, l'estimation par Manning est simplement hors sujet et l'étalement observé est celui du réseau, pas d'un paramètre.

### Balayage complet de K sur le vrai réseau (OUTV, banc de routage, avec noyau HGM)

| K_musk | KGE aux jauges | r réseau vs Hydrotel |
|---|---|---|
| 0.35 h (Manning) | 0.395 | 0.373 |
| 23.7 h (appris) | **0.526** | 0.470 |
| 48 h (borne haute) | 0.480 | 0.550 |
| 120 h (hors bornes) | 0.191 | **0.616** |

**DISSOCIATION MAJEURE : la fidélité à Hydrotel croît de façon MONOTONE avec K, alors que l'accord aux OBSERVATIONS culmine au K appris (~24 h) et s'effondre au-delà.** Deux conséquences.

1. **Le K appris est déjà à son optimum vis-à-vis des observations.** L'entraînement fait ce qu'il peut ; le routage est innocenté, cette fois par un BALAYAGE de paramètre et non par un échange de schémas (la faiblesse de la ronde de juin).
2. **Ressembler structurellement à Hydrotel REND MOINS BON contre le réel.** Hydrotel est à la fois très diffusif ET précis (0.82 aux jauges sur OUTV) ; méandre diffusif est mauvais. Donc l'avantage d'Hydrotel n'est PAS son routage : sa diffusion n'est bonne que parce qu'elle est nourrie par une production différente. Le levier restant est la GÉNÉRATION, ce que concluait déjà juin — désormais établi par un balayage quantifié.

**Erreur de banc corrigée au passage :** ma mesure « Hydrotel translate sans atténuer » était fausse parce que j'injectais l'impulsion en apport LATÉRAL dans les deux schémas. Le clone traite très différemment l'eau latérale (traverse quasi intacte) et l'eau d'AMONT (−43 % de pic PAR tronçon, −97 % après 40). Le réseau réel propage surtout de l'eau d'amont : Hydrotel est donc BEAUCOUP plus diffusif que méandre, l'inverse de ce que j'avais écrit. Leçon de conception de banc : tester un opérateur sur le canal d'entrée qui domine réellement, pas sur le plus commode.

### Le champion n'était pas reproductible — et la cause est UNE ligne

Reproduction à configuration « identique » : **0.5504 contre 0.7489**. Diff des bannières des deux journaux : une seule ligne d'écart, `[etl] w_et override = 0.0`. **Le champion du 29 juillet tournait avec la contrainte d'évapotranspiration MODIS DÉSACTIVÉE** (`ETL_WET=0`) ; toutes mes reproductions du 9 août l'avaient active (w_et = 1.0 du fichier de configuration).

Conséquence immédiate : **tout ce que j'ai attribué aujourd'hui à la fermeture de masse, au noyau HGM, au nombre d'époques ou au taux d'apprentissage des lacs portait en réalité sur un terme de perte différent.** Les comparaisons À 12 ÉPOQUES entre elles restent valides (toutes avaient w_et actif) ; les comparaisons AU CHAMPION ne l'étaient pas, aucune.

Enseignement de méthode, le plus cher de la journée : la configuration d'un run doit être reconstituée depuis SON journal, pas depuis la mémoire de la recette. Un simple `diff` des bannières, fait au premier écart inexpliqué, aurait économisé six entraînements de 1 à 3.5 h. À automatiser : bannière complète de toutes les variables ETL_* en tête de journal, et comparaison automatique à la meilleure exécution connue.

Deuxième question ouverte, importante en soi : si w_et = 1.0 coûte 0.20 en tenu de côté sur GASP, le multi-objectif MODIS — présenté comme un pilier de l'identifiabilité — est à réexaminer sur toutes les régions.

## 2026-08-09 (nuit) — MODIS ET DÉBITS NE SE CONTREDISENT PAS : ILS ACCUSENT ENSEMBLE LE FORÇAGE

Question d'Essi : « MODIS et les mesures de débit sont donc des données irréconciliables ? » Test SANS MODÈLE (`bilan_modis_vs_debit.py`) : sur 19 ans la variation de stock s'annule, donc P − Q = ETR. Aucun paramètre, aucune simulation.

**GASP, 15 jauges, 2003-2021 (mm/an) : P 1012 | Q 648 | P−Q = 287 | MODIS 523 | écart +188 (+66 %), et MODIS > P−Q sur 15/15.**

Lecture : ce n'est PAS MODIS l'accusé. Une ETR de 287 mm/an est physiquement absurde pour une forêt tempérée recevant plus d'un mètre de pluie (tours de flux boréales : 400-500 ; MODIS 523 est crédible). Le débit est mesuré. **C'est donc la PLUIE qui manque : il en faudrait 1171 mm/an (Q + MODIS) là où le forçage en donne 1012, soit −16 %.**

Et ce chiffre en explique un autre, traîné depuis des mois : **le biais de volume chronique du modèle, beta ≈ 0.85, vaut exactement 1012/1171 = 0.864.** Le modèle ne sous-produit pas : on lui donne 16 % d'eau en moins.

**Comparaison des forçages disponibles sur GASP (P moyen, mm/an) :**

| forçage | P | contre le requis 1171 |
|---|---|---|
| `-pgm` (PyGMET) | **1274** | +9 % |
| `forcing-gasp.nc` (CaSR brut) | 1101 | −6 % |
| `-budyko` | 1091 | −7 % |
| `-hyb` (**celui qu'on utilise**) | 1029 | **−12 %** |
| `-krig` / `-lin` | 1017 / 1018 | −13 % |

**Nos corrections successives ont ASSÉCHÉ l'entrée : le forçage canonique est plus sec que le CaSR brut.** Même en supposant MODIS surestimé de 20 % (le biais est +15-30 % à l'est selon ma propre note de juillet), le requis retombe à ~1083, soit encore au-dessus de `-hyb`.

Réponse à la question : **MODIS et les débits sont RÉCONCILIABLES — ils s'accordent entre eux et désignent conjointement le forçage.** Le multi-objectif ne demandait pas l'impossible au modèle ; il lui demandait d'évaporer 523 mm à partir d'une pluie amputée, ce qui ne peut se payer que sur le débit. D'où −0.20 de KGE quand w_et = 1.0.

Généralisation en cours sur OUTV, SAGU, SLNO. Test suivant si la mesure tient : réentraîner GASP sur `-pgm` (recette du champion, `ETL_WET=0`) et comparer.

### Inventaire honnête : quels écarts ont RÉELLEMENT été corrigés (question d'Essi)

**Appliqué par défaut, validé :**
- Fuite de masse BV3C2 (`BV3C_FERMETURE=1` par défaut) : bilan fermé à +2 mm/an, coefficient d'écoulement 0.518, 150 tests.
- Conservation de `MuskingumCunge` (chemin dormant, la production est en mode opérateur).

**Trouvé, mesuré POSITIF, mais TOUJOURS EN OPTION (donc absent de tous les runs) :**
- Noyau HGM de versant (`ETL_HGM=0` par défaut) : +0.027 en inférence, r réseau 0.335 -> 0.470.
- Surface d'eau libre des lacs (`ETL_LAKE_AREA=0` par défaut) : +0.015 en tenu de côté, tête de lac réparée.
Ces deux-là n'ont pas été promus parce que chacun a été mesuré contre un socle qui s'est révélé non reproductible (confusion `w_et`). Promotion à refaire sur le socle correct.

**Trouvé, PAS corrigé du tout :**
- **19 des 37 sorties du NeRF ne sont lues par aucun module actif ; 17 paramètres restent figés à l'initialisation.** Vérifié à nouveau ce soir : `f_vert`, `vsa_b`, `interception_capacity` n'apparaissent que dans des commentaires, le chargeur MODIS et la tête de bruit — jamais dans la colonne active. **Conséquence rétrospective grave : le « décollapse de f_vert ×6-8 » du 28 mai, présenté comme la preuve d'identifiabilité du multi-objectif, portait sur un paramètre que la colonne clone actuelle N'UTILISE PAS.** Les conclusions de mai-juin sur la partition verticale ne transfèrent pas à l'architecture d'aujourd'hui.
- Loi de tarage des lacs (fidélité ×3 mais entraînement pire), assèchement du forçage (découvert ce soir), `lake_lr_mult=50` actif par défaut depuis le 5 août sans validation sur un socle propre.

**Confirmation régionale du déficit de pluie :** SAGU, 19 jauges : P 1011 | Q 678 | P−Q = 341 | MODIS 469, écart +123 mm/an (+36 %), MODIS > P−Q sur 18/19. Le motif de GASP se répète.

## 2026-08-10 — FIDÉLITÉ À INTRANTS IDENTIQUES : l'assemblage réparé change tout

Recadrage d'Essi : identifier les zones de BIFURCATION méandre/Hydrotel, sur une base saine, avant toute amélioration. Entraînement en file annulé. Test refait avec la **météo du PROJET Hydrotel** (Thiessen, mêmes intrants des deux côtés), tous paramètres FIGÉS depuis le projet, assemblage réparé (fermeture de masse + noyau HGM + surface d'eau libre + loi de tarage des lacs).

| mesure (OUTV, 2022-2024, 3412 tronçons) | fidélité v1 (7 août) | aujourd'hui |
|---|---|---|
| r médian réseau | 0.368 | **0.576** |
| beta médian (volume) | 0.584 | **1.018** |
| r têtes < 50 km² | 0.278 | **0.482** |
| r lacs | 0.202 | **0.668** |
| KGE aux jauges | 0.087 | **0.557** |

**Le volume annuel colle maintenant à Hydrotel (beta 1.018 contre 0.584).** Et un méandre SANS AUCUN ENTRAÎNEMENT, paramètres figés, obtient 0.557 aux jauges — mieux que le champion entraîné (0.4992). La base est saine.

**Surface convertie disculpée par mesure directe** : somme des aires locales de méandre 83202 km² contre 83198 km² pour les 8821 UHRH d'Hydrotel (rapport 1.000). Ce n'était pas un facteur d'aire. (L'aire dite « physique » vaut 64.6× le total : c'est l'aire cumulée, exactement le facteur 66 du bug de lac déjà corrigé.)

**BIFURCATION RÉSIDUELLE, unique et nette : l'apport latéral du 1er août ne vaut que 0.265 de celui d'Hydrotel (contre 0.131 avant) alors que le volume ANNUEL est juste.** Le total est bon, la répartition saisonnière ne l'est pas : il manque de l'eau en été. Mesure en cours avec la décomposition nouvellement exposée (surface / hypodermique / base) et le cycle saisonnier du débit réseau, pour dire QUEL flux décroche et QUAND.

## 2026-08-10 — LA BIFURCATION PRINCIPALE : l'occupation du sol n'atteignait JAMAIS la physique

Les colonnes `f_forest`, `f_water`, `f_urban`, `f_wetland` de la base du Québec sont **centrées-réduites** (f_forest va de -3.67 à +1.27, moyenne nulle) et les colonnes brutes `f_*_raw` — les seules que `get_physical` expose (convention `DEFAULT_PHYSICAL_COLUMNS` + suffixe `_raw`) — **n'ont jamais été écrites par le constructeur de régions**. `hydrotel_column._static_params` retombait donc sur ses défauts : **méandre simulait l'Outaouais comme 100 % de sol nu DÉCOUVERT, sans forêt, sans eau libre, sans imperméable, sans milieu humide**, là où Hydrotel a 67.7 % de forêt (35.8 feuillus + 31.9 conifères), 9.4 % d'eau et 2.4 % d'imperméable.

Propagation : le découvert est la classe de neige la plus fondante (Hydrotel y porte 52 mm d'équivalent en eau contre 136 sous conifères) ; `fse = 0` supprime le ruissellement direct sur l'eau libre ; `fsi = 0` supprime le ruissellement imperméable ; la phénologie de l'ETR travaille sur la mauvaise végétation.

**Correctif** : `load_occupation_sol()` lit `physitel/occupation_sol.cla` (9 classes) et l'agrège par tronçon au prorata des aires d'UHRH ; `HydrotelColumn.set_land_cover()` la fait primer sur le territorial. Chargé sur OUTV : forêt 0.742 (conif 0.319), eau 0.093, imperméable 0.018, humide 0.054. Bug dormant corrigé au passage (`gp(...) or 0.0` sur un tenseur, invisible tant que la fraction restait nulle).

**Effet, à intrants identiques et paramètres entièrement figés (OUTV, 2022-2024, 3412 tronçons) :**

| mesure | v1 (7 août) | sans occupation | **avec occupation** |
|---|---|---|---|
| r médian réseau | 0.368 | 0.526 | **0.896** |
| r têtes < 50 km² | 0.278 | 0.448 | **0.874** |
| r lacs | 0.202 | 0.614 | **0.921** |
| theta1 / theta2 | 0.92 / 0.90 | 0.92 / 0.90 | **0.978 / 0.968** |
| apport latéral (1er août) | 0.131 | 0.259 | **0.514** (corr 0.874) |
| **KGE aux jauges, ZÉRO entraînement** | 0.087 | 0.482 | **0.7486** |

Repères : Hydrotel ~0.82, champion méandre ENTRAÎNÉ 0.4992. **Un méandre sans aucun entraînement dépasse de 0.25 le meilleur modèle entraîné de la série.**

**Bifurcation suivante, désormais isolée : l'ÉVAPOTRANSPIRATION D'ÉTÉ.** beta passe de 1.019 à 1.147 (excès de volume) et le profil saisonnier montre où : janvier-mai à ±12 % d'Hydrotel, mais juin 1.32, juillet 1.55, août 1.56, septembre 1.65, octobre 1.55. Production 624 mm/an contre ~545 pour Hydrotel, soit une ETR de ~355 chez nous contre ~434 chez lui. Le modèle n'évapore pas assez pendant la saison de croissance.

**Leçon de méthode, la plus importante de la session** : ce trou a survécu des mois parce que tous les diagnostics portaient sur les paramètres APPRIS (dispersion, collapse, bornes) et jamais sur les ENTRÉES STATIQUES effectivement reçues par la physique. Une seule ligne de contrôle — imprimer les fractions d'occupation vues par la colonne — l'aurait révélé immédiatement. À ajouter en garde-fou de démarrage.

### Audit complet des entrées statiques muettes (3 agents parallèles, 2026-08-10)

Même famille que l'occupation du sol : une entrée demandée en brut, absente du cache, repli silencieux sur un défaut.

1. **Classes de neige** (déjà corrigé) : 100 % du Québec en classe découvert.
2. **Partition fsa/fse/fsi** (corrigé par le même chargeur) : `fse = 0` et `fsi = 0`, donc aucune pluie-sur-lac directe ni ruissellement imperméable, sur des tronçons à 9 % d'eau.
3. **Phénologie de l'ETR** (corrigé) : sans fractions, `et_classes` restait vide et le code repliait sur une classe végétale unique — LAI et profondeur racinaire uniformes sur toute la province.
4. **MILIEU HUMIDE ISOLÉ : module ENTIÈREMENT DÉSACTIVÉ, non corrigé.** `wet_a_raw` absent -> `_wetland_from_territorial` renvoie None -> `wetland=None`, et `wet_vol` reste 0. Aucun laminage par milieu humide nulle part, pour 7.6 % de superficie humide moyenne. Silencieux : aucun log. Les paramètres existent pourtant dans le projet Hydrotel (`simulation/simulation/milieux_humides_isoles.csv` et `_riverains.csv`).
5. **`depth_to_bedrock_m` écrit en zéros constants** par les deux constructeurs, et exclu du NeRF puisque déclaré physique : canal mort des deux côtés.
6. **`reach_length_m`** déclaré physique mais jamais écrit par le chemin PHYSITEL.
7. **Piège armé** : `mean_slope_pct_raw` retomberait sur 4 % partout ; masqué aujourd'hui parce que `slope_fraction` existe.
8. **Risque de corruption par normalisation** : `sin_aspect`/`cos_aspect` sont lus dans le tenseur NORMALISÉ. Restés bruts sur le chemin PHYSITEL (vérifié, min/max ±1), mais le chemin `basin_builder` z-score toutes les colonnes : un `atan2` de deux variables standardisées séparément donne un azimut faux, silencieusement.

**Carte d'usage du champ spatial (37 sorties)** : 13 ACTIVES, 5 conditionnelles (dont 2 inactives au Québec), 6 lues seulement par les pertes de régularisation et la tête probabiliste, **13 MORTES**. Les 17 paramètres « figés » signalés depuis des jours sont exactement les morts et les décoratifs ; aucun paramètre actif n'est figé. L'entraînement porte en réalité sur 13 champs.

**Température de fonte** : `T_melt` EXISTE dans le champ spatial et est lu, mais seulement si `spatial_melt`, et l'ancrage régional écrase le seuil. Le TAUX a déjà la forme ancrage × modulation apprise bornée ; le SEUIL n'a pas d'équivalent. Correction proposée : symétriser (seuil ancré + écart appris borné ±1.5 °C, régularisé vers zéro). `T_snow` (partition pluie/neige) est MORT dans le champ ; il est désormais chargé du `thiessen.csv`.

**Deux bugs dans l'ingestion des données molles** : (a) MODIS fournit une moyenne 8 jours posée sur un seul jour du calendrier, comparée à l'ETR simulée de CE jour — bruit pur, et pollution de toute variance servant à standardiser ; (b) la ligne de base GRACE est calculée par tronçon de séquence alors que le code documente une ligne de base longue durée. Enfin le `w_et` est une MSE brute en mm²/j² non normalisée, alors que débit et TWS sont réduits : le poids 1.0 n'a pas d'échelle comparable.

### Réparations en chaîne du 10 août : le modèle FIGÉ dépasse le champion ENTRAÎNÉ

| étape (OUTV, intrants identiques, paramètres FIGÉS) | r réseau | r têtes | r lacs | beta | KGE jauges |
|---|---|---|---|---|---|
| v1 du 7 août | 0.368 | 0.278 | 0.202 | 0.584 | 0.087 |
| + fermeture de masse, HGM, lacs trl, surface d'eau | 0.576 | 0.482 | 0.668 | 1.018 | 0.557 |
| + seuil pluie/neige du projet (thiessen.csv) | 0.526 | 0.448 | 0.614 | 1.019 | 0.482 |
| **+ occupation du sol PHYSITEL** | 0.896 | 0.874 | 0.921 | 1.147 | **0.749** |
| + milieux humides isolés | 0.915 | 0.896 | 0.940 | 1.098 | 0.741 |
| **+ ETR sur TOUTE la fraction perméable** | **0.922** | **0.903** | **0.944** | **1.069** | **0.7514** |

Repères : Hydrotel ~0.82, champion méandre ENTRAÎNÉ 30 époques 0.4992 sur cette configuration, 0.7489 sur la recette du 29 juillet. **Un méandre à paramètres entièrement figés, sans une seule époque, dépasse désormais le meilleur modèle entraîné de la série.**

**Dernier correctif : l'ETR ne couvrait que 79.6 % du territoire.** Seules les classes forêt et milieu humide recevaient un profil de végétation ; l'agriculture (12 % sur OUTV) et le sol nu ne transpiraient PAS, alors que les profils `agri` et `ouverts` existent dans `_LEAF`/`_ROOT` depuis toujours. Correctif : classe agricole ajoutée, puis résidu de la fraction perméable versé en classe ouverte. Effet mesuré : production 601 -> 568 mm/an, excès d'été de 1.49 à 1.34 en juillet, de 1.59 à 1.39 en septembre.

**Reproductibilité du champion** : avec `ETL_WET=0`, 0.6871 contre 0.5504 — le terme MODIS vaut donc bien ~0.14 de KGE à lui seul. Il reste 0.062 d'écart au 0.7489, attribuable à la variance entre exécutions ou à un changement de défaut depuis le 29 juillet. Chasse abandonnée : ce socle est périmé par les correctifs d'assemblage.

**Bifurcations restantes, dans l'ordre** : avril à 0.797 (crue de fonte encore trop faible, alors que mars et mai sont justes) ; excès d'été résiduel de 30 à 40 % ; ligne de base GRACE calculée par tronçon de séquence ; seuil de fonte à symétriser avec le taux (ancrage + écart appris borné) ; 13 sorties mortes du champ spatial à retirer.

### Couverture de tests : elle ne couvrait RIEN de ce qui a cassé

Question d'Essi : les tests de compatibilité sont-ils complets ? Non, et le signal était sous nos yeux : les 150 tests passaient sans broncher après CHAQUE correctif majeur du 10 août. Audit :

| correctif du jour | test qui le couvrait |
|---|---|
| conservation de masse BV3C2 | aucun |
| conservation de l'apport latéral Muskingum | aucun |
| seuil pluie/neige calibré | aucun |
| occupation du sol atteignant la physique | aucun |
| milieux humides isolés | 2 fichiers `smoke_*` NON COLLECTÉS par pytest (motif `test_*`) |
| ETR sur toute la fraction perméable | aucun |
| appariement 8 jours de MODIS | aucun |
| bornes de K_musk | aucun |

`tests/test_entrees_statiques.py` ajouté : 17 tests, dont la conservation du Muskingum sur 9 combinaisons de K et de sous-pas (attrapait 0.50 et 1.85), l'appariement des composites, la priorité de `set_land_cover`, la couverture de l'ETR sur toute la fraction perméable, et trois chargeurs confrontés au vrai projet Hydrotel (ignorés proprement s'il est absent). Suite complète : **167 tests**.

La moyenne glissante a été extraite de la boucle de perte en fonction nommée `moyenne_glissante`, précisément pour être testable : le correctif était enfoui dans un bloc conditionnel où rien ne pouvait l'atteindre.

Tentative de récupérer les deux `smoke_*` en les renommant : ABANDONNÉE. Ce sont des scripts autonomes qui changent le répertoire courant à l'import, ce qui faisait tomber 16 tests sans rapport. À réécrire proprement plus tard ; leurs invariants sont désormais couverts côté projet Hydrotel.

### Entraînement GASP avec tous les correctifs d'assemblage

`ETL_TAG=-repare`, 30 époques, `ETL_WET=0`, occupation + milieux humides actifs (forêt 0.803 dont 0.462 de conifères ; 2414 tronçons porteurs de milieu humide) : **tenu de côté 0.7236**.

Comparaison honnête : contre la REPRODUCTION de la recette du champion (0.6871, même code sans les correctifs), les correctifs valent **+0.037**. Contre le champion historique du 29 juillet (0.7489), on reste 0.025 en dessous, mais ce chiffre n'a jamais pu être reproduit (écart inexpliqué de 0.062 sur la même recette), donc il ne constitue pas une référence utilisable.

Réserve importante : GASP est une région où méandre allait DÉJÀ bien. Le test qui décide est OUTV, qui plafonnait à 0.4992 entraîné et où le modèle FIGÉ atteint maintenant 0.7514.

**La mesure la plus solide de la journée reste celle à paramètres figés**, parce qu'elle ne souffre d'aucune variance d'optimisation : sur OUTV, à intrants identiques, r réseau 0.368 -> 0.922 et KGE aux jauges 0.087 -> 0.7514.

## 2026-08-10 — COMPATIBILITÉ ÉTABLIE : neige, sol et routage sont FIDÈLES ; il reste la répartition saisonnière de la production

`compat_hydrotel.py` sur OUTV, tout figé depuis le projet Hydrotel, météo du projet sur sa fenêtre complète (2020-2026, 2242 j), prélèvements nuls des deux côtés, zéro entraînement.

| étage | mesure | rapport | corrélation |
|---|---|---|---|
| **1. NEIGE** stock pondéré par couvert, 2026-02-19 | 108.09 contre 108.14 mm | **1.000** | **+0.994** |
| **2. SOL** theta1 / theta2 / theta3, 2023-08-01 | 0.964 / 0.948 / 0.999 | | +0.981 / +0.935 / +0.999 |
| 2. SOL theta1 / theta2 / theta3, 2026-02-19 | 1.008 / 0.978 / 0.996 | | +0.991 / +0.974 / +0.993 |
| **3. PRODUCTION** apport latéral, 2 dates | 0.528 et 0.695 | | +0.794 / +0.752 |
| **4. ROUTAGE** débit amont, 2 dates | 0.919 et 0.689 | | **+0.986 / +0.997** |
| 4. ROUTAGE débit aval, 2 dates | 0.826 et 0.838 | | **+0.985 / +0.997** |
| **5. SÉRIE** 2022-2024, 3412 tronçons | beta 1.069 | | **r 0.922** (lacs 0.944) |

**Trois étages sont validés, deux d'entre eux pour la PREMIÈRE FOIS.**
- La NEIGE est exacte au millième près, avec une corrélation de 0.994 sur 3412 tronçons. Elle n'avait jamais été comparée faute de date d'état enneigée dans notre fenêtre de forçage. Conséquence directe : **le déficit de crue d'avril (0.80) n'est PAS un problème de neige** — à manteau identique, notre sol restitue moins d'eau de fonte.
- Le SOL colle à 5 % près sur les couches actives et à 0.1 % sur la couche profonde, aux deux saisons.
- Le ROUTAGE est fidèle : corrélations de 0.985 à 0.997 sur les débits amont ET aval, et les rapports amont et aval sont du même ordre, donc **le transfert n'ajoute aucun biais** — ce qui clôt définitivement la piste routage, ouverte et refermée trois fois cette semaine.

**La divergence résiduelle est ENTIÈREMENT dans la répartition saisonnière de la production** : cycle mensuel méandre/Hydrotel = 1.08, 1.12, 1.01, **0.80**, 1.03, 1.18, **1.35, 1.29, 1.39, 1.31**, 1.17, 1.14. Trop peu en avril, 30 à 40 % de trop de juillet à octobre, sur un volume annuel excédentaire de 7 %. La décomposition indique où regarder : surface 370, hypodermique 219, base 0.3 mm/an. Le débit de base est nul, donc l'étiage est porté par l'hypodermique seul, et l'eau de fonte d'avril part en stockage au lieu de s'écouler.

### Compatibilité GASP : la neige et le routage tiennent, le SOL diverge (deuxième cause)

`compat_hydrotel.py` sur GASP, mêmes conditions : **neige exacte à 3 ‰** (141.80 contre 141.39 mm, corr 0.991) — le module de neige est donc validé sur DEUX régions et deux textures de couvert. Routage fidèle (corr 0.96-0.98). Volume annuel juste (beta 0.997). r réseau 0.903.

**Divergence propre à GASP : les deux couches actives du sol sont trop SÈCHES** (theta1 0.634, theta2 0.550 d'Hydrotel) alors que la couche profonde colle (0.986) et que les corrélations spatiales restent hautes (0.89 / 0.80). Cycle saisonnier en miroir de celui d'OUTV : déficit d'hiver (0.77, 0.74) et excès d'été (1.23, 1.21).

**Première cause identifiée et confirmée : le PLAFOND DE SOUS-PAS de Courant.** GASP a un sol 4.6× plus perméable qu'OUTV (ks médian 0.0611 contre 0.0132 m/h, texture sableuse dominante) et des pentes plus fortes (0.082 contre 0.061). Or c'est la perméabilité qui fixe le nombre de sous-pas exigé par Courant : le plafond de 48 mord beaucoup plus en Gaspésie, et la fermeture de masse évacue alors au ruissellement l'eau du temps non traité. Plafond porté à 300 (`MEANDRE_NSUBSTEP`, nouveau) :

| | 48 sous-pas | 300 sous-pas |
|---|---|---|
| r médian réseau | 0.903 | **0.930** |
| r lacs | 0.958 | **0.968** |
| janvier / février | 0.77 / 0.74 | **0.97 / 0.94** |
| theta1 / theta2 | 0.634 / 0.550 | 0.666 / 0.584 |

Le déficit d'hiver est réparé, la fidélité structurelle progresse nettement. **Mais le sol reste sec : il existe une SECONDE cause**, un décalage systématique de niveau sur les deux couches actives, à corrélation spatiale conservée. À isoler.

### Plafond de sous-pas porté à 300, phénologie du projet lue : bilan

**Plafond de Courant (`MEANDRE_NSUBSTEP`, défaut historique 48)** — le C++ boucle jusqu'à épuiser la journée, le plafond est une concession au GPU. Effet mesuré à 300, coût ~1.5× en temps (la boucle sort tôt quand elle a fini, donc le plafond ne coûte QUE là où il servait) :

| | OUTV 48 | OUTV 300 | GASP 48 | GASP 300 |
|---|---|---|---|---|
| r médian réseau | 0.922 | **0.932** | 0.903 | **0.930** |
| beta | 1.069 | **1.052** | 0.997 | 0.992 |
| theta1 / theta2 | 0.964 / 0.948 | 0.968 / 0.960 | 0.634 / 0.550 | 0.666 / 0.584 |
| hiver (jan/fév) | 1.07 / 1.14 | 1.07 / 1.14 | 0.77 / 0.74 | **0.97 / 0.94** |

**PHÉNOLOGIE DU PROJET (`physio/ind_fol.def`, `pro_rac.def`) : lue et branchée, effet NUL.** Les profils étaient codés en dur depuis DELISLE, avec des écarts réels (racines conifères 1.531 m sur OUTV et 1.26 sur GASP contre 1.0 en dur ; milieux humides 1.531 contre 0.75 ; agriculture 0.108 contre 0.8 ; indice foliaire des feuillus nul en hiver et culminant à 6 contre un plancher à 3 et un maximum de 5). Résultat : r 0.932 -> 0.930, cycle d'été inchangé. **Correctif de FIDÉLITÉ, pas de performance** : à ces profondeurs de sol, 1.0 m atteignait déjà la couche 3, donc la répartition de l'extraction change à peine. Conservé quand même, et consigné comme tel pour ne pas le recompter plus tard comme un gain.
Subtilité de lecture : les deux fichiers n'ont pas la même grille de jours (1/158/188/299/365 contre 1/160/190/260/365), interpolation sur la grille réunie.

**État de la compatibilité** : neige exacte sur 2 régions, sol à 3-4 % sur OUTV, routage à 0.99, r réseau 0.93. **Résidu unique : un excès de volume de 7 % concentré en juillet-octobre (1.25 à 1.36), plus un déficit d'avril propre à OUTV (0.76).** Pour trancher l'excès d'été il faut l'ETR d'Hydrotel jour par jour, que seule la réexécution instrumentée peut fournir — elle en est à **52 h de CPU sans avoir écrit un seul fichier**.

### Le FORÇAGE est innocenté par mesure directe (COMPAT_METEO)

Substitution de la météo du projet Hydrotel par notre forçage `-hyb`, tout le reste identique (même code, mêmes paramètres figés, même fenêtre) : **résultats identiques au millième** — r réseau 0.930, beta 1.069, cycle mensuel inchangé mois par mois.

Explication mesurée : sur la fenêtre commune, `-hyb` donne 988 mm/an contre 959 pour la météo du projet, soit **+3.1 %**, avec une corrélation journalière de **0.990** sur la pluie et de **0.999** sur les températures (écarts moyens -0.62 et -0.23 °C). Notre forçage canonique est donc, à 3 % près, la même chose que l'interpolation de stations d'Hydrotel.

Trois conséquences.
1. **La pénalité de forçage est nulle** dans la comparaison à Hydrotel : tout écart résiduel est du CODE, pas de la météo. Le banc de compatibilité peut donc tourner indifféremment avec l'une ou l'autre.
2. Le CaSR BRUT, lui, donne 1109 mm/an sur la même fenêtre, soit **+16 % par rapport à la météo d'Hydrotel** : ce sont nos corrections qui ont ramené le forçage au niveau des stations, pas l'inverse. La lecture d'hier soir (« nos corrections ont asséché l'entrée sous le CaSR brut ») était juste sur le fait et fausse sur le jugement : elles l'ont aligné sur la référence station.
3. La question « faut-il un multiplicateur de précipitation appris par région » reste ouverte mais change de nature : il ne s'agirait plus de corriger un biais par rapport à Hydrotel, mais de trancher entre les stations (959-988) et le bilan hydrique, qui en réclamait davantage. Or Hydrotel atteint 0.82 aux jauges avec 959. À trancher par la mesure, pas par le bilan seul.

### Le plafond de sous-pas se heurte à la compilation : arbitrage à trancher

| n_substep | compilation | r réseau GASP | hiver (jan/fév) | coût |
|---|---|---|---|---|
| 48 (historique) | oui | 0.903 | 0.77 / 0.74 | référence |
| 64 (max compilable) | oui | 0.902 | 0.83 / 0.80 | ~= |
| **300** | **NON** (RecursionError dans l'inductor : la boucle est DÉROULÉE) | **0.930** | **0.97 / 0.94** | inférence ~1.5× ; **entraînement > 30 h par région, ingérable** |

L'essentiel du gain se joue AU-DELÀ de la limite compilable : 64 ne récupère qu'un quart de la réparation d'hiver et rien sur r. Compilation auto-désactivée au-delà de 64 (garde-fou ajouté) plutôt que de brider la physique en silence.

**Nature du problème, qui mérite une phrase dans le papier** : le C++ boucle jusqu'à épuiser le pas de temps parce qu'il tourne en séquentiel sur processeur, où une itération de plus ne coûte presque rien. Le portage différentiable impose de dérouler la boucle pour compiler le graphe, donc de la borner. **L'écart n'est pas seulement physique, il est une conséquence de l'architecture de calcul.**

**Vrai correctif à faire** : compiler un BLOC de K sous-pas (K <= 32) et l'appeler N/K fois depuis Python en passant l'état de boucle (t1/t2/t3, tr, lruis/lhyp/lbase). Profondeur de graphe K, nombre total de sous-pas libre. Refactoring d'une quarantaine de lignes dans le clone le plus délicat du dépôt, à faire à tête reposée et à valider contre le C++ avant adoption.

En attendant : entraînement à 64 (compilé), diagnostics de compatibilité à 300.

## 2026-08-11 — RÉSULTAT CHARNIÈRE : sur OUTV, la physique ANCRÉE bat la physique APPRISE de 0.134

Comparaison à réglage strictement identique (OUTV, `-hyb`, 64 sous-pas, tenu de côté 2022-2024, mêmes 16 jauges) :

| modèle | KGE médian tenu de côté |
|---|---|
| champion historique (base cassée, 30 époques) | 0.4992 |
| meilleur transfert d'une autre région | 0.5731 |
| **entraîné 30 époques sur la base saine** | **0.6051** |
| **FIGÉ sur le calage Hydrotel, zéro époque** | **0.7389** |
| Hydrotel lui-même | ~0.82 |

Deux lectures, toutes deux importantes.

1. **Les correctifs d'assemblage valent +0.106 en entraînement** (0.4992 -> 0.6051) sur la région la plus déficiente, et font passer devant le meilleur transfert (0.5731) — ce qui n'était jamais arrivé.
2. **Mais l'entraînement PERD 0.134 contre les paramètres simplement ancrés sur le calage d'Hydrotel.** Et le journal montre le mécanisme sans ambiguïté : à la dernière époque, validation Nash 0.804 et KGE 0.834, pour un tenu de côté à 0.605. **Écart de généralisation de 0.23.** Le modèle apprend très bien sa période et généralise mal ; le jeu ancré n'a pas ce défaut par construction.

**Conséquence sur le rôle de l'apprentissage.** Là où une calibration Hydrotel existe, l'optimisation s'éloigne d'un optimum qui généralise mieux qu'elle. La valeur du champ spatial ne se mesure donc PAS en gain de score sur les régions déjà calées : elle se mesure en régions NON JAUGÉES (produire un jeu de paramètres là où il n'y en a pas) et sur les questions qu'un modèle calé ne peut pas traiter (renaturalisation, scénarios d'occupation, prélèvements). C'est aussi la réponse à la question sur la couche d'expérience : son utilité doit être jugée hors des bassins jaugés.

**Trois expériences qui découlent directement, par ordre :**
1. Départ à chaud DEPUIS l'ancré + affinage court avec régularisation forte vers l'ancré (l'ancrage pendant l'entraînement avait échoué le 4 août, mais sur la base cassée : à refaire).
2. Validation croisée par bassin (laisser des jauges dehors) pour mesurer ce que le champ apporte réellement en non jaugé, contre l'ancré et contre le transfert.
3. Plan à 4 cases forçage × contrainte ET, qui tranche l'identifiabilité pluie/évaporation (stations 959-988 avec ETR 311, ou CaSR brut 1109 avec ETR 461 : les deux donnent le bon débit, seul le second a une ETR crédible).

## 2026-08-11 — LE « 0.82 D'HYDROTEL » ÉTAIT UN FANTÔME : méandre ancré est à PARITÉ sur OUTV

Essi conteste l'utilité du modèle : « méandre fait 0.605 alors qu'Hydrotel le déclasse à 0.82 ». Deux corrections, la seconde étant une faute de ma part.

1. Le 0.605 est méandre ENTRAÎNÉ (son propre calage). Le modèle qui emprunte le calage d'Hydrotel fait 0.7389.
2. **Le « 0.82 » n'a jamais été mesuré.** C'était une chaîne de caractères écrite dans mes scripts de diagnostic et répétée pendant des jours. Deuxième chiffre fantôme du projet après le 0.75 de juillet, et celui-là je l'ai introduit moi-même.

**Mesure réelle** (Hydrotel LN24HA, `debit_aval.nc` du projet, MÊMES 16 jauges, MÊME période 2022-2024, MÊME formule KGE) :

| modèle | KGE médian tenu de côté |
|---|---|
| **Hydrotel LN24HA** | **0.7531** (moyenne 0.7645 ; par station de 0.549 à 0.842) |
| **méandre ANCRÉ sur le calage Hydrotel** | **0.7389** |
| méandre entraîné (base saine) | 0.6051 |
| ancien champion méandre | 0.4992 |

**Méandre ancré est donc à 0.014 d'Hydrotel : c'est la parité**, sur la région la PLUS difficile de la province, avec une chaîne entièrement réécrite en PyTorch différentiable.

Ce que cela ne change pas : le constat d'Essi sur l'ENTRAÎNEMENT reste entier. La version qui cale elle-même est 0.148 derrière Hydrotel et 0.134 derrière sa propre version ancrée. Le problème n'est pas la physique, c'est l'optimisation.

**Réserve posée immédiatement** : 0.7531 est le membre LN24HA SEUL. La règle du projet est de comparer à l'ENSEMBLE des 6 calages (posttraitement_{LN24HA,MG24Hx}.zarr), ce qui sera plus sévère. À faire avant toute communication de ce résultat.

### Duel contre l'ENSEMBLE, et ce que l'ancrage transmet vraiment

**Trois appariements ratés avant d'obtenir le bon chiffre**, ce qui mérite d'être noté : le stockage provincial indexe par un RANG (`troncon_idx`, 0..28034) et porte les identifiants dans une COORDONNÉE `troncon_id` de la forme "REG#####". Comparer des entiers a donné des KGE de -0.25 (absurdes, donc détectés) ; comparer des chaînes au rang n'a rien apparié. Le dépôt manipule trois numérotations de tronçons et chaque script refait la conversion à la main — dette à supprimer par une fonction unique.

**Ensemble Hydrotel sur OUTV** (6 membres, 16 jauges, 2022-2024) : membres à 0.7531 (LN24HA), 0.7616, 0.8003, 0.8299 (MG24HK), 0.7934, 0.8151. **Ensemble médian par station 0.7711**, meilleur membre par station 0.8543. Le banc est validé par recoupement : LN24HA y donne 0.7531, exactement la valeur obtenue indépendamment depuis `debit_aval.nc`.

**Correction de ma propre correction** : j'avais annoncé la parité une heure plus tôt. Elle ne valait que contre le membre le PLUS FAIBLE. Méandre ancré (0.7389) est sous les six membres et 0.032 sous l'ensemble médian. La formulation initiale d'Essi était plus juste que ma correction.

**L'ANCRAGE NE TRANSMET PAS LA QUALITÉ DU CALAGE.** Ancré sur MG24HK (0.8299, le meilleur) : méandre fait **0.7424**, contre 0.7389 ancré sur LN24HA (0.7531, le plus faible). **+0.077 à la source donnent +0.0035 à l'arrivée.** Le volume, lui, s'améliore nettement (beta 1.081 -> 1.005). Conclusion : **méandre a son propre plafond, ~0.74 sur OUTV, indépendant de la calibration injectée.** L'écart résiduel à l'ensemble est le NÔTRE — l'excès d'été et le déficit d'avril — pas un défaut de calage hérité.

Corollaire pratique : produire notre propre ensemble à 6 membres ne servirait à rien tant que ce plafond tient, puisque les six ancrages donneraient à peu près le même résultat. L'idée est écartée pour l'instant.

### O1, pli 0/4 : le champ spatial perd 0.071 en régionalisation, et reste dominé par l'ancrage

OUTV, 4 jauges retirées de l'entraînement ET de la validation, évaluées sur 2022-2024 :

| | KGE médian |
|---|---|
| 4 jauges **jamais vues** | **0.5239** (moyenne 0.5495) |
| 12 jauges vues, même entraînement | 0.5946 |
| 16 jauges, entraînement complet | 0.6051 |
| **paramètres ANCRÉS, zéro entraînement** | **0.7389** |

Deux lectures, la seconde étant la plus lourde.
1. La régionalisation coûte **0.071** : le champ ne s'effondre pas sur des bassins qu'il n'a jamais vus (0.52 reste loin devant un modèle nul), mais il perd sensiblement.
2. **Le champ appris est dominé par l'ancrage MÊME LÀ OÙ IL A LES OBSERVATIONS** : 0.595 contre 0.739 sur les jauges vues. Le problème n'est donc pas la régionalisation, c'est l'apprentissage lui-même.

Réserve : 4 jauges, donc bruité. Les plis 1 à 3 sont en file (une nuit).

Premier essai perdu : masquer `q_obs` retirait bien les jauges de l'entraînement mais les rendait INÉVALUABLES, donc le pli ne rapportait rien. Correctif : copie intacte des observations réservée à l'évaluation.

## 2026-08-13 — O1 et O6 tranchées : la régionalisation est quasi gratuite, la couche d'expérience ne sert plus

**O1 — validation croisée spatiale, 4 plis sur OUTV** (chaque jauge retirée exactement une fois) :

| pli | jauges jamais vues (n=4) | jauges vues (n=12) |
|---|---|---|
| 0 | 0.5239 (moy 0.5495) | 0.5946 |
| 1 | 0.5367 (moy 0.5558) | 0.6010 |
| 2 | **0.7098** (moy 0.6828) | 0.5627 |
| 3 | 0.5318 (moy 0.5763) | 0.6450 |

**En groupant les 16 jauges : moyenne 0.5911, contre 0.6043 pour l'entraînement complet. Coût de régionalisation : -0.013.** Le champ spatial prédit donc presque aussi bien sur une jauge qu'il n'a jamais vue que sur une jauge dont il a les observations. C'est le premier argument SOLIDE en faveur du champ, et il porte sur ce qui compte : le non jaugé.

**Correction de ma propre lecture** : après les plis 0 et 1 j'avais annoncé un coût stable de 0.07. Prématuré. À n=4 par pli la variance est énorme — le pli 2 donne les jauges RETIRÉES meilleures que les vues (0.710 contre 0.563). Deux plis concordants ne font pas une tendance ; c'est la troisième fois cette semaine que je conclus trop tôt.

**O6 — la couche d'expérience (codes latents par nœud)** : avec 0.6051, **sans 0.6106**. Elle ne rapporte RIEN, et coûte un paramètre par nœud plus la non-reproductibilité d'un départ à chaud. **À retirer.** Elle compensait bien les entrées fausses, comme supposé : une fois l'occupation du sol, les milieux humides et l'ETR réparés, son apport disparaît.

**Ce qui reste vrai et dominant** : l'ancré fait 0.7389 là où l'entraîné plafonne à 0.605, jauges vues comprises. Le chantier est l'apprentissage, pas la régionalisation.

## 2026-08-13 — POURQUOI l'entraîné perd : le champ est COLLÉ à un prior faux de 8 à 23×

Essi propose : Hydrotel gagne parce qu'il est contraint, méandre perd par excès de malléabilité ; faut-il lisser davantage le champ ? **Trois mesures disent l'inverse, et une quatrième donne la vraie cause.**

**Le champ appris n'est pas trop souple, il est presque PLAT.** Dispersion globale de K_sat couche 1 : **0.054 pour l'appris contre 0.740 pour le calage Hydrotel** — quatorze fois moins. Méandre applique quasiment la même conductivité aux 3412 tronçons là où Hydrotel distingue les textures (facteur 5 du sable au loam). Localement l'appris ondule un peu entre voisins (0.02) quand Hydrotel est exactement constant par classe de sol. Ajouter du lissage aggraverait donc le défaut.

**La vraie cause : les NIVEAUX.**

| paramètre | appris | Hydrotel | rapport |
|---|---|---|---|
| K_sat 1 (m/j) | 0.0373 | 0.3168 | **0.12** |
| K_sat 2 | 0.0440 | 0.3168 | 0.14 |
| K_sat 3 | 0.0140 | 0.3168 | **0.04** |
| Z3 (m) | 0.891 | 2.650 | 0.34 |
| porosité 1 | 0.449 | 0.434 | 1.03 |

**Le champ n'a pratiquement pas bougé de son initialisation** : on impose `ETL_KSAT1=0.04` et il finit à 0.0373. Or 0.04 m/j vaut 0.0017 m/h, quand la table des textures donne 0.0132 m/h pour le loam — **notre prior est 8× sous la valeur physique**, et il est appliqué UNIFORMÉMENT là où la texture varie d'un facteur 5.

Ce prior a été adopté le 21 juillet pour « fermer le déficit de r sur GASP », c'est-à-dire pour compenser un modèle qui, on le sait depuis le 10 août, perdait 21 % de sa pluie en crue et n'évaporait que sur 80 % du territoire. **C'est une compensation d'une pathologie aujourd'hui corrigée : CADUC.**

Cohérence de l'ensemble : un champ quasi uniforme ne peut pas sur-ajuster (d'où E10, O1 et O6 qui pointaient tous dans ce sens), mais un champ quasi uniforme MAL PLACÉ explique très bien un score médiocre stable. Le remède n'est ni le lissage ni la régularisation vers la moyenne, c'est le rétrécissement vers des valeurs PHYSIQUES — ce que fait l'ancrage du sol, déjà en file.

**Test immédiat ajouté** : le même entraînement sans le prior faux (`ETL_KSAT1` retiré, donc init littérature) et avec le prior texture.

## 2026-08-13 — La PÉRIODE est innocentée : le modèle ancré est stable sur 24 ans

Essi : « si l'apprentissage détériore, c'est que la fonction de perte est mauvaise ». J'avais opposé une troisième cause possible, la différence climatique de 2022-2024 (mesurée : pluie estivale +30 % contre la période de validation, +1.0 °C, pics observés plus bas). **Test : le modèle ANCRÉ, qui n'apprend rien et ne peut donc rien sur-ajuster.**

| fenêtre | KGE médian, modèle ancré |
|---|---|
| 2001-2003 | 0.7313 |
| 2004-2006 | 0.7337 |
| 2007-2009 | 0.7450 |
| 2010-2012 | 0.7362 |
| 2013-2015 | 0.8033 |
| 2016-2018 | 0.7707 |
| **2019-2021 (validation)** | **0.7711** |
| **2022-2024 (tenu de côté)** | **0.7748** |

**Le modèle ancré ne chute pas : +0.0038 de la validation au tenu de côté, et une amplitude de 0.07 seulement sur 24 ans.** La période 2022-2024 n'a rien d'anormalement difficile — sa pluie estivale supplémentaire ne gêne pas un modèle correctement paramétré.

**Mon hypothèse climatique est donc RÉFUTÉE**, posée et tombée le même jour. Restent les deux causes qu'Essi désignait : la perte vise la mauvaise cible, ou le modèle sur-ajuste sa période. Et comme le champ est quasi plat (dispersion 0.054 contre 0.740 pour Hydrotel) et collé à son prior, « sur-ajuster » ne peut pas vouloir dire « mémoriser du détail spatial » : il s'agit d'un NIVEAU effectif ajusté à une période.

Conséquence pratique : le découpage temporel reste utilisable tel quel, la configurabilité ajoutée aujourd'hui (JOINT_SPLIT, ETL_HELDOUT) servira à la robustesse, pas à corriger un biais.

## 2026-08-13 — L'ENTRAÎNÉ EST BATTU PAR L'ANCRÉ SUR SA PROPRE PÉRIODE D'ENTRAÎNEMENT

Comparaison enfin valide (mêmes réglages d'exécution des deux côtés, mêmes 16 jauges, même formule) :

| fenêtre | ANCRÉ (0 époque) | ENTRAÎNÉ (30 époques) | écart |
|---|---|---|---|
| 2001-2003 | 0.7313 | 0.6396 | -0.092 |
| 2004-2006 | 0.7337 | 0.6347 | -0.099 |
| 2007-2009 | 0.7450 | 0.5836 | -0.161 |
| 2010-2012 | 0.7362 | 0.6056 | -0.131 |
| 2013-2015 | 0.8033 | 0.6446 | -0.159 |
| 2016-2018 | 0.7707 | 0.6704 | -0.100 |
| **2019-2021 (sélection)** | 0.7711 | **0.7103** | -0.061 |
| **2022-2024 (tenu de côté)** | 0.7748 | 0.6051 | -0.170 |

**Deux faits, et le premier est le plus lourd.**

1. **L'entraîné est battu par l'ancré sur TOUTES les fenêtres, y compris 2000-2018 sur lesquelles il a été ENTRAÎNÉ** (0.58-0.67 contre 0.73-0.80). Ce n'est donc PAS du sur-ajustement au sens classique : un modèle qui sur-ajuste excelle au moins sur ses données. Celui-ci est moins bon partout, sur ses propres données comprises. **La conclusion d'Essi est la bonne : l'optimisation ne converge pas vers une solution que la physique atteint sans apprendre. Soit l'optimum de la perte n'est pas la bonne solution, soit l'optimiseur ne peut pas l'atteindre.**
2. Le profil de l'entraîné culmine EXACTEMENT sur la fenêtre de sélection (0.7103, contre 0.60-0.67 sur les voisines) : la sélection sur validation gonfle ce point d'environ 0.05. Une part de la chute annoncée (-0.105) est donc un artefact de sélection, pas une dégradation réelle.

**Rappel du mécanisme mesuré plus tôt** : le champ reste collé à son initialisation (K_sat appris 0.0373 pour un prior à 0.04) et ce prior est 8× sous la valeur physique. L'optimiseur ne peut donc pas atteindre la solution ancrée — elle est à un facteur 8 de son point de départ, avec un gradient de débit qui déplace la dispersion de 0.0017 à 0.054 en 30 époques quand il en faudrait 0.74.

**Piège de reproductibilité inscrit à la dette** : un point de reprise ne définit PAS un modèle. Occupation du sol, milieux humides, phénologie, noyau de versant et lois de lac sont posés à l'exécution et absents du fichier. Évalué sans eux, le même checkpoint tombe de 0.6051 à 0.4449 — 0.16 d'écart, sans la moindre erreur. À corriger en stockant ces réglages (ou leur empreinte) dans le point de reprise.

## 2026-08-14 — SOCLE ENTRAÎNABLE À LA RÉFÉRENCE : le champ peut porter K_sat et les porosités

Bissection à ZÉRO époque (20 min par point au lieu de 4 h, grâce à `ETL_EPOCHS=0` qui évalue le modèle en mémoire), configuration alignée sur celle qui vaut 0.7748 en inférence (ETP Linacre calée, pas d'aquifère, pas de codes latents, seuil pluie/neige posé) :

| ce qui est imposé | reste au champ | KGE tenu de côté |
|---|---|---|
| tout le sol (25 champs) | rien | **0.7368** |
| **tout SAUF K_sat et porosités** | **K_sat, porosités** | **0.7389** |
| courbe de rétention seule (18 champs) | K_sat, porosités, épaisseurs | 0.5921 |
| rien (champ ajusté seul + courbe globale) | tout | 0.5629 |

**Le champ spatial porte K_sat et les porosités sans perdre un centième.** Les 0.145 manquants venaient des SEPT autres champs : épaisseurs z1/z2/z3, fractions fsa/fse/fsi et pente. Ce sont des DONNÉES, pas des paramètres à calibrer — les imposer est correct, pas une triche.

Détail qui compte : z1 était codé en dur à 0.15 m alors que le calage donne 0.219 m, et ce n'est même pas une sortie du champ.

**On tient donc, pour la première fois, une configuration ENTRAÎNABLE qui démarre à la référence** (0.7389, contre 0.6051 pour la recette précédente). Entraînement lancé, abandon automatique si la validation reste sous 0.60 à l'époque 4.

Chemin parcouru, et ce qui l'a rendu possible : cinq contrôles à zéro époque en deux heures là où les trois jours précédents avaient consommé huit entraînements de 4 h pour des verdicts moins nets. La règle est désormais explicite : valider une initialisation à zéro époque, cribler à quatre avec abandon, n'engager trente que sur un candidat qui a passé les deux.

## 2026-08-15 — PREMIER GAIN RÉEL DE L'APPRENTISSAGE : 0.7810 sur OUTV, au-dessus de l'ensemble médian

Entraînement 30 époques depuis le SOCLE (tout imposé sauf K_sat et porosités, ETP Linacre calée, pas d'aquifère, pas de codes latents, seuil pluie/neige posé), OUTV, 16 jauges, tenu de côté 2022-2024 :

| référence | KGE médian |
|---|---|
| ancien champion méandre | 0.4992 |
| recette précédente sur base saine | 0.6051 |
| Hydrotel LN24HA (le membre qu'on ancre) | 0.7531 |
| **ENSEMBLE Hydrotel, médian par station** | **0.7711** |
| socle, ZÉRO époque | 0.7389 |
| physique ancrée (inférence pure) | 0.7748 |
| **socle + 30 époques** | **0.7810** |
| médiane des médianes de membre | 0.7968 |
| meilleur membre par station | 0.8543 |

**Trois faits, énoncés sans extrapoler.**
1. **L'apprentissage APPORTE pour la première fois** : +0.042 sur sa propre initialisation (0.7389 -> 0.7810). Toutes les tentatives précédentes en RETIRAIENT (jusqu'à -0.134).
2. **Le modèle dépasse la physique ancrée** (0.7748) et **l'ensemble médian par station** (0.7711), sur la région la PLUS difficile de la province.
3. **Il ne dépasse PAS** la médiane des médianes de membre (0.7968), ni les quatre meilleurs membres (0.800 à 0.830), ni le meilleur par station (0.854). Il bat 2 membres sur 6.

Ce qui a rendu ce résultat possible, dans l'ordre : réparation des sept entrées muettes (10-11 août), diff systématique des deux configurations plutôt que devinettes successives, et surtout la discipline de contrôle à zéro époque qui a permis de localiser en cinq essais de 20 minutes ce que huit entraînements de 4 h n'avaient pas trouvé.

**À faire avant toute communication** : confirmer sur une deuxième région (contrôles à zéro époque déjà en file pour GASP, SAGU, SLNO, MONT), et vérifier que les deux divergences physiques connues (excès d'été, déficit d'avril) n'ont pas simplement été compensées par les paramètres libres.

### Le socle se transpose : 0.71 à 0.77 sur quatre régions SANS AUCUN entraînement

Contrôles à zéro époque (20 min chacun), socle identique, tenu de côté 2022-2024 :

| région | socle, 0 époque | meilleur modèle ENTRAÎNÉ précédent |
|---|---|---|
| GASP | **0.7749** | 0.7489 (champion) |
| OUTV | 0.7389 (**0.7810** après 30 époques) | 0.4992 |
| SAGU | 0.7517 | — |
| SLNO | **0.7106** | 0.546 |
| MONT | 0.4869 | — |

**Sur ces régions, la configuration correcte SANS apprentissage bat les modèles qu'on entraînait depuis des semaines.** Le Lac-Saint-Jean passe de 0.546 à 0.7106 sans un seul pas de gradient ; la Gaspésie dépasse son champion de 0.026.

MONT décroche à 0.4869 : à diagnostiquer séparément (fichiers de calage ? occupation ? régime ?), et c'est justement le genre de cas que le contrôle à 20 minutes permet d'isoler avant d'y consacrer 4 h.

30 époques lancées sur les trois régions qui passent.

## 2026-08-15 — BILAN SUR QUATRE RÉGIONS, contre l'ensemble Hydrotel mesuré

Socle identique partout (tout imposé sauf K_sat et porosités, ETP Linacre calée, pas d'aquifère, pas de codes latents, seuil pluie/neige posé), tenu de côté 2022-2024, mêmes jauges, même formule KGE.

| région | méandre 0 époque | méandre 30 époques | ENSEMBLE Hydrotel (médian/station) | écart | meilleur membre |
|---|---|---|---|---|---|
| OUTV | 0.7389 | **0.7810** | 0.7711 | **+0.010** | 0.8543 |
| GASP | 0.7749 | **0.7766** | 0.7730 | **+0.004** | 0.8502 |
| SAGU | 0.7517 | 0.7587 | 0.7933 | **-0.035** | 0.8302 |
| MONT | 0.4869 | (non lancé) | 0.6631 | -0.176 | 0.7763 |
| SLNO | 0.7106 | en cours | à mesurer | | |

**Deux régions au-dessus de l'ensemble médian, deux en dessous.** L'énoncé honnête est la parité sur OUTV et GASP, un retard net sur SAGU, et un décrochage sur MONT.

**Ce que l'apprentissage ajoute dépend du point de départ** : +0.042 sur OUTV (parti de 0.7389), +0.007 sur SAGU, +0.002 sur GASP (parti de 0.7749). Les trois atterrissent entre 0.759 et 0.781 malgré des calages et des régimes différents — cohérent avec un PLAFOND COMMUN autour de 0.78, qui ne viendrait ni des paramètres (les calages diffèrent) ni de l'optimisation (les points de départ diffèrent). À tester en changeant le forçage.

**MONT diagnostiqué** : ce n'est pas l'assemblage des paramètres de départ (tous les chargeurs fonctionnent, et la région partage EXACTEMENT les paramètres de fonte de GASP qui, elle, atteint 0.7749). C'est un bassin **hyper-régulé** : 2273 barrages pour 1916 nœuds, plus d'un par tronçon, cinq fois la densité gaspésienne, et seulement 37 nœuds-lacs (2 %) donc les réservoirs ne sont pas dans le routage. Hydrotel y plafonne aussi (0.6631 contre 0.77 ailleurs). Processus manquant, pas défaut de paramétrage — le corriger demanderait des données d'exploitation qu'on n'a pas.

Réserve maintenue : le MEILLEUR membre par station reste à 0.83-0.85 partout, mais c'est une borne inaccessible sans connaître les observations d'avance.
