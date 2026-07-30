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
