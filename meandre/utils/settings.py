"""Réglages du paquet : chaque réglage est déclaré ici, une seule fois.

Le paquet lisait sa physique et son calcul dans l'environnement, en dizaines d'endroits
dispersés. Chaque réglage est maintenant déclaré dans `SCHEMA` avec une clé en anglais,
qui est aussi sa clé dans le fichier de configuration (`column.n_substeps` est la clé
`n_substeps` de la section `[column]`), son ancien nom de variable d'environnement, gardé
comme alias pour que les scripts existants se comportent comme avant, son défaut et son
rôle.

Ordre de priorité, le même que celui de la section [recette] :
    1. la variable d'environnement de l'alias, si elle est posée ;
    2. la valeur du fichier de configuration passée à `configure` ;
    3. le défaut déclaré ici, identique au défaut historique du code.

Un lanceur qui n'appelle pas `configure`, comme les tests ou le banc SLSO, obtient les
étapes 1 et 3 : son comportement ne change pas.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    env: str
    kind: type
    default: object
    doc: str


SCHEMA: dict[str, Setting] = {
    # -- Colonne verticale --------------------------------------------------------------
    "column.n_substeps": Setting("MEANDRE_NSUBSTEP", int, None, "Sous-pas du sol par jour ; absent, l'argument de construction de la colonne s'applique."),
    "column.z1": Setting("MEANDRE_Z1", float, 0.15, "Épaisseur de la couche 1 du sol, en mètres, uniforme ; 0,15 jusqu'ici en dur dans la colonne. Épreuve du calendrier de la réponse (2026-10-29) : une couche de surface mince et poreuse joue le rôle de l'horizon organique."),
    "spatial.z1_head": Setting("MEANDRE_Z1_CHAMP", bool, False, "Épaisseur de la couche 1 apprise par tronçon : tête linéaire du champ (fc_z1), nulle au départ en poids et dont le biais rend column.z1, bornée par Z1 de [field.bounds] (2026-10-29) ; absente, column.z1 uniforme."),
    "column.penman_snow": Setting("MEANDRE_PENMAN_NEIGE", bool, False, "Couvert nival dans l'ETP de Penman (R265)."),
    "column.transpiration_soil_temp_ref": Setting("MEANDRE_TRANSP_TSOL_REF", float, None, "Échelle, en degrés, du frein de la transpiration par le sol froid (R264) ; absent, pas de frein."),
    "column.viscosity_tref": Setting("MEANDRE_VISCOSITE_TREF", float, None, "Température de référence, en degrés, de la correction de conductivité par la viscosité de l'eau (Vogel) ; absent, pas de correction."),
    "column.semi_implicit_soil": Setting("MEANDRE_SOL_SEMI_IMPLICITE", bool, False, "Schéma linéairement implicite du sol : gradient borné, graines reproductibles."),
    "column.frozen_gate_amplitude": Setting("MEANDRE_GEL_AMPLITUDE", float, 1.0, "Amplitude de la porte de gel : capacité d'infiltration ks1 (1 − a · fraction gelée), et même facteur sur le drainage de la couche 1 ; 1 reproduit la porte d'origine, qui ferme le sol dès que le gel atteint l'épaisseur de la couche 1 (2026-10-23)."),
    "column.root_zone_stress": Setting("MEANDRE_STRESS_RACINAIRE", str, "layer1", "Stress hydrique de la transpiration : « layer1 » (fidèle, eau de la zone racinaire jugée sur la capacité au champ et le point de flétrissement de la couche 1) ou « layered » (eau utile de chaque couche, pondérée par sa part racinaire) ; une couche 1 organique jugeait les couches minérales presque sèches (R445, 2026-10-29)."),
    "land_cover.impervious_table": Setting("MEANDRE_TABLE_IMPERMEABLE", str, None, "Table parquet (territory, node_id, impervious) des surfaces réellement imperméables par tronçon, par exemple le bâti ESA WorldCover 2021, qui remplace la classe d'usage urbain de PHYSITEL (R455, 2026-10-30) ; absente, PHYSITEL."),
    "column.retention": Setting("MEANDRE_RETENTION", str, "campbell", "Loi de rétention et de conductivité du sol : campbell (clone fidèle) ou van_genuchten (Mualem-van Genuchten, n = sortie vg_n du champ, theta_r et alpha tels que la courbe passe par la capacité au champ et le point de flétrissement de chaque couche) (2026-11-03)."),
    "column.saturated_area_exponent": Setting("MEANDRE_VSA_B", float, None, "Exposant b du ruissellement sur aire saturée, part (theta1/thetas1)^b (R386) ; absent, pas de ruissellement de saturation partiel."),
    "column.dynamic_compile": Setting("MEANDRE_COMPILE_DYNAMIQUE", bool, False, "Compilation du sol en formes dynamiques ; faux garde le mode fixe mesuré à x17,6."),
    # -- Routage et calcul ----------------------------------------------------------------
    "routing.muskingum_k_hours": Setting("MEANDRE_KMUSK", tuple, (4.0, 48.0, 24.0), "Temps de transfert de Muskingum, en heures : borne basse, borne haute, valeur initiale."),
    "routing.substeps": Setting("ETL_ROUTAGE_SOUSPAS", int, 2, "Sous-pas du routage par jour."),
    "routing.lake_evaporation": Setting("MEANDRE_EVAP_LACS", bool, True, "Évaporation des lacs routés par temps sec."),
    "compute.operator_block": Setting("MEANDRE_OPERATOR_BLOCK", int, None, "Taille maximale d'un bloc de l'opérateur de routage ; absent, un bloc par étage. Résultat identique, mémoire bornée."),
    "compute.gradient_segments": Setting("MEANDRE_SEGMENT_GRADIENT", int, 0, "Longueur, en jours, des segments recalculés à la rétropropagation ; 0, pas de recalcul. Gradient identique, mémoire bornée."),
    # -- Tête de quantiles ----------------------------------------------------------------
    "quantile.enabled": Setting("ETL_QUANTILE", bool, False, "Tête de quantiles (phase probabiliste, épine dorsale gelée)."),
    "quantile.log_step": Setting("MEANDRE_QUANTILE_LOG_STEP", float, 0.05, "Largeur de chaque niveau de quantile, dans le logarithme du débit."),
    "quantile.log_bound": Setting("MEANDRE_QUANTILE_LOG_BOUND", float, 4.0, "Un quantile extrême reste entre e^-b et e^b fois la médiane."),
    # -- Diagnostics de sortie --------------------------------------------------------------
    "diagnostics.cpu": Setting("MEANDRE_DIAG_CPU", bool, False, "Diagnostics journaliers gardés sur le processeur plutôt que sur la carte graphique."),
    "diagnostics.half_precision": Setting("MEANDRE_DIAG_F16", bool, False, "Diagnostics non dérivés gardés en demi-précision sur le processeur."),
    "diagnostics.kept": Setting("MEANDRE_DIAG_KEEP", str, None, "Diagnostics journaliers collectés pendant la simulation, séparés par des virgules (noms internes de la colonne : etr, recharge, q_baseflow...) ; absent, tous. Les autres valent None, ce qui fait tenir une passe provinciale avec diagnostics en mémoire."),
    "diagnostics.retained": Setting("MEANDRE_DIAG_DERIVES", str, None, "Diagnostics gardés avec leur graphe, séparés par des virgules ; absent, tous ceux qu'un terme peut dériver."),
    # -- Données -----------------------------------------------------------------------------
    "data.withdrawals_table": Setting("MEANDRE_WITHDRAWALS_TABLE", str, "withdrawals", "Table des prélèvements lue dans la base de chaque territoire."),
    "data.territorial_extra": Setting("MEANDRE_TERRITORIAL_EXTRA", str, "", "Tables territoriales supplémentaires ajoutées aux attributs du champ, séparées par des virgules."),
    # -- Entraîneur ---------------------------------------------------------------------------
    "trainer.lr_mult_heads": Setting("MEANDRE_LR_MULT_TETES", float, None, "Multiplicateur commun du taux des têtes du champ et des lacs ; absent, 10 pour la sortie du champ et lake_lr_mult pour les lacs."),
    "trainer.lr_mult_soil": Setting("MEANDRE_LR_MULT_SOL", float, 50.0, "Multiplicateur du taux du profil de sol appris."),
    "trainer.lr_mult_routing": Setting("MEANDRE_LR_MULT_ROUTAGE", float, 100.0, "Multiplicateur du taux de la tête du temps de transfert."),
    "trainer.lr_mult_lakes": Setting("MEANDRE_LR_MULT_LACS", float, 50.0, "Multiplicateur du taux des scalaires des lacs hors réseau."),
    "trainer.save_every_epoch": Setting("MEANDRE_SAUVER_EPOQUES", bool, False, "Un point de reprise par époque, pour évaluer toute la trajectoire."),
    "trainer.abandon": Setting("MEANDRE_ABANDON", str, None, "Arrêt époque:seuil si la métrique de sélection reste sous le seuil après cette époque."),
    "trainer.step_per_block": Setting("MEANDRE_PAS_PAR_BLOC", bool, False, "Un pas d'optimisation par bloc ; faux, un pas par époque (R78, R288)."),
    "trainer.history_priming": Setting("MEANDRE_HISTORIQUE_AMORCE", bool, False, "Passe sans gradient en début d'époque pour que chaque bloc voie le KGE de toute la période."),
    # -- Sondes de diagnostic -------------------------------------------------------------
    "probe.terms": Setting("MEANDRE_SONDE_TERMES", str, "", "Champs dont la sonde mesure le gradient de chaque terme, séparés par des virgules."),
    "probe.terms_only": Setting("MEANDRE_SONDE_TERMES_SEULS", str, "", "Termes retenus par la sonde, séparés par des virgules ; vide, tous."),
    "probe.terms_output": Setting("MEANDRE_SONDE_TERMES_SORTIE", str, "sonde_termes.npz", "Fichier de sortie de la sonde par terme."),
    "probe.conflict": Setting("MEANDRE_SONDE_CONFLIT", str, "", "Fichier de sortie de la sonde de conflit des gradients ; vide, sonde éteinte."),
    "probe.conflict_blocks": Setting("MEANDRE_SONDE_CONFLIT_BLOCS", int, 4, "Nombre de blocs mesurés par la sonde de conflit avant l'arrêt."),
    "probe.segment": Setting("MEANDRE_SONDE_SEGMENT", bool, False, "Sonde du recalcul par segments : gradient et mémoire du deuxième bloc."),
    # -- Phénologie : la section [phenology] est lue par le pilote, qui transmet ses valeurs
    # au modèle ; ces réglages ne servent qu'en son absence (voir PILOT_SECTIONS). ------------
    "phenology.params": Setting("MEANDRE_PHENOLOGIE_PARAMS", str, None, "Paramètres gelés du mode photo, « emergence,seuil,largeur1,largeur2 »."),
    "debug.simulate_end": Setting("MEANDRE_FIN_SIMULATION", str, None, "Coupe la fenêtre simulée à cette date (AAAA-MM-JJ), depuis le début du forçage : contrôles d'identité et essais de fumée en quelques minutes au lieu de 25 ans ; jamais dans un modèle livré (2026-10-26)."),
    "spatial.priors": Setting("MEANDRE_PRIORS", bool, False, "Priors par tronçon sur le sol (porosités, capacité au champ, point de flétrissement, conductivités, épaisseurs) tirés de la texture et des dépôts du domaine ; le champ apprend un écart pénalisé ([training] w_deviation_prior). Les valeurs du modèle chargé sont conservées à l'adoption (2026-10-26)."),
    "priors.keys": Setting("MEANDRE_PRIOR_CLES", str, None, "Sorties placées sous prior, séparées par des virgules (par exemple « porosity_3,theta_fc_frac_3 ») ; absent, toutes celles du catalogue. Sous prior une sortie n'a plus de sigmoïde, donc plus de saturation : avec une pénalité nulle, c'est la façon de libérer une sortie collée à sa borne sans changer sa valeur de départ (2026-10-27)."),
    "stations.area_ratio": Setting("MEANDRE_RAPPORT_AIRES", bool, False, "Rapporte le débit observé de chaque station à l'aire du tronçon qui la porte (observé × aire du tronçon / aire de la station), à débit spécifique uniforme ; sans ce réglage, la station est comparée telle quelle au tronçon (2026-10-28, R429)."),
    "stations.exclude_reconstructed": Setting("MEANDRE_SANS_RECONSTITUES", bool, False, "Retire de la perte et de l'évaluation les débits que l'organisme a reconstitués (indicateur `reconstructed` de la base, remarque R), surtout sous la glace : interpolations entre jaugeages, pas des mesures (R385, décision d'Essi du 2026-10-28)."),
    "stations.attachment_table": Setting("MEANDRE_TABLE_RATTACHEMENT", str, None, "Table de rattachement station-tronçon écrite par `meandre/adapters/physitel/builders/audit_stations.py` (colonnes `territory`, `station`, `reach_id`, `status`), {DERIVES} accepté ; les stations `outside_territory` sont exclues, les `changed` prennent le tronçon `reach_id` (2026-10-28)."),
    "stations.excluded": Setting("MEANDRE_STATIONS_EXCLUES", str, None, "Stations retirées de la comparaison et de l'entraînement, séparées par des virgules, par exemple une station hors du réseau modélisé (062002, R429)."),
    "priors.k_sat_3_center": Setting("MEANDRE_PRIOR_KSAT3_CENTRE", str, None, "Centre du prior de la conductivité de la couche 3 : « substratum » prend la conductivité par tronçon de [soil.ceiling_from_drainage] (classe de drainage) au lieu de la pédotransfert de texture ; absent, la texture (2026-10-27)."),
    "priors.start_at_center": Setting("MEANDRE_PRIOR_DEPART_CENTRE", str, None, "Sorties sous prior qui DÉMARRENT au centre du prior à l'adoption, séparées par des virgules (par exemple « K_sat_3 ») ; les autres conservent la valeur du modèle chargé. Sert quand le centre est une grandeur physique que le modèle chargé n'a jamais vue (2026-10-27)."),
    "priors.level_keys": Setting("MEANDRE_PRIOR_CLES_NIVEAU", str, None, "Sorties sous prior jugées en mode niveau quel que soit priors.mode, séparées par des virgules : leur niveau est tenu par la carte, pas seulement leurs contrastes (2026-10-27)."),
    "priors.mode": Setting("MEANDRE_PRIOR_MODE", str, "anomaly", "Comparaison du champ à ses priors : level (niveau de la carte), anomaly (contrastes entre tronçons, niveau libre, comme MOD16), shape (motif seul, niveau et amplitude libres, comme GRACE). La dispersion mesurée contre l'IRDA est une dispersion centrée : elle correspond au mode anomaly (2026-10-26)."),
    "priors.sigma_k_sat": Setting("MEANDRE_PRIOR_SIGMA_KSAT", float, 1.609, "Écart-type, en logarithme naturel, du prior de conductivité : ln 5. Mesuré le 2026-10-26 contre la perméabilité des levés IRDA par tronçon : dispersion d'un facteur 2 au sud (Saint-Laurent sud-ouest, Montérégie, Outaouais), 2,8 à 5 au nord ; on retient le pire."),
    "priors.sigma_porosity": Setting("MEANDRE_PRIOR_SIGMA_POROSITE", float, 0.20, "Écart-type en logarithme du prior de porosité (pédotransfert de Saxton et Rawls) ; non mesuré contre une donnée indépendante."),
    "priors.sigma_theta_fc_frac": Setting("MEANDRE_PRIOR_SIGMA_CC", float, 0.20, "Écart-type en logarithme du prior de capacité au champ, en fraction de la porosité ; non mesuré."),
    "priors.sigma_theta_wp_frac": Setting("MEANDRE_PRIOR_SIGMA_PF", float, 0.30, "Écart-type en logarithme du prior de point de flétrissement, en fraction de la capacité au champ ; non mesuré."),
    "priors.sigma_z2": Setting("MEANDRE_PRIOR_SIGMA_Z2", float, 0.693, "Écart-type en logarithme du prior d'épaisseur de la couche 2 : ln 2 ; non mesuré."),
    "priors.sigma_z3": Setting("MEANDRE_PRIOR_SIGMA_Z3", float, 0.916, "Écart-type en logarithme du prior d'épaisseur de la couche 3 (épaisseur des dépôts SIGEOM) : ln 2,5 ; non mesuré."),
    "phenology.classes": Setting("MEANDRE_PHENOLOGIE_CLASSES", str, None, "Classes d'occupation modulées, séparées par des virgules ; absent, toutes."),
    # -- Pilote régional : réglages lus sous forme de texte, comme dans l'environnement --------
    "snow.wet_bulb_threshold": Setting("ETL_SEUIL_TWB", str, None, "Seuil de partage pluie-neige au thermomètre mouillé, en degrés (Stull 2011) ; absent, seuil sur la température de l'air."),
    "snow.seasonal_melt_amplitude": Setting("ETL_MELT_SAISON", str, None, "Amplitude de la modulation saisonnière du facteur de fonte, s(j) = 1 + a sin(2π(j−81)/365) ; absent ou 0, clone fidèle."),
    "groundwater.aquifer": Setting("ETL_AQUIFER", str, None, "Voie souterraine active (aquifère restituant ou nappe libre)."),
    "groundwater.free_recession": Setting("ETL_KREC_LIBRE", str, None, "Constante de récession apprise par le champ, moyenne ancrée au prior."),
    "groundwater.water_table": Setting("ETL_NAPPE_LIBRE", str, None, "Nappe libre dont la profondeur est l'état, à la place du réservoir restituant."),
    "groundwater.exponent": Setting("ETL_NAPPE_EXP", str, None, "Exposant de vidange de la nappe, ou « mesure » pour l'ancrer sur les récessions observées ; absent, 2."),
    "groundwater.specific_yield": Setting("ETL_NAPPE_SY", str, None, "Porosité de drainage de la nappe ; absent, 0,05."),
    "groundwater.conductance": Setting("ETL_NAPPE_KB", str, None, "Conductance de vidange de la nappe ; absent, 2e-3."),
    "groundwater.river_depth_m": Setting("ETL_NAPPE_ZRIV", str, None, "Profondeur du lit de la rivière sous la surface, en mètres ; absent, 8."),
    "groundwater.reference_head_m": Setting("ETL_NAPPE_HREF", str, None, "Charge de référence de la nappe, en mètres ; absent, 4."),
    "groundwater.extinction_depth_m": Setting("ETL_NAPPE_ZEXT", str, None, "Profondeur d'extinction de l'extraction par les racines, en mètres ; absent, 9."),
    "groundwater.extraction_fraction": Setting("ETL_NAPPE_EFRAC", str, None, "Part de l'évapotranspiration prise dans la zone saturée ; absent, 0,35."),
    "spatial.no_latent_codes": Setting("ETL_NO_LATENT", str, None, "Codes latents par tronçon coupés (non transférables)."),
    "spatial.global_normalization": Setting("JOINT_GLOBAL_NORM", str, None, "Attributs du territoire normalisés sur toute la province."),
    "evaporation.open_water": Setting("ETL_EVAP_EAU_LIBRE", str, None, "Évaporation de l'eau libre par temps sec (R285, R286)."),
    "evaporation.open_water_reservoir": Setting("MEANDRE_RESERVOIR_EAU_LIBRE", str, None, "Réservoir de la pluie tombée sur l'eau libre hors lacs routés, « T,beta,h_ref » (séjour en jours à la hauteur h_ref en m, exposant de vidange) ; l'évaporation reste prise sur le débit latéral (R285). Exige evaporation.open_water. Absent, la pluie sur l'eau libre rejoint la rivière le jour même (2026-11-03)."),
    "withdrawals.disabled": Setting("ETL_SANS_PRELEV", bool, False, "Simulation sans prélèvements ni rejets (débits naturalisés)."),
    "lakes.enabled": Setting("MEANDRE_LACS_HORS_RESEAU", bool, False, "Lacs hors réseau (HydroModel.set_distributed_lakes) : une part de la production locale traverse un réservoir de la surface d'eau libre hors lacs routés, vidé par q = f k h^beta ; le ruissellement des processus déclarés les contourne (lakes.spill_bypass). Exige evaporation.open_water (2026-11-04)."),
    "lakes.share_coefficient": Setting("MEANDRE_LACS_PART", float, 6.7, "Coefficient c de la part de la production locale qui traverse les lacs hors réseau, 1 − exp(−c × fraction d'eau libre hors lacs routés), sans dimension."),
    "lakes.residence_from_volume": Setting("MEANDRE_LACS_DIFFUS_VOLUME", bool, False, "Séjour des lacs hors réseau déduit par tronçon du volume des lacs et de l'eau qui les traverse, T = D f / (rd R), D = lakes.mobilizable_depth_mm ; faux, séjour uniforme lakes.residence_days (2026-11-05)."),
    "lakes.mobilizable_depth_mm": Setting("MEANDRE_LACS_HAUTEUR_MOBILISABLE", float, 171.0, "Hauteur d'eau mobilisable D des lacs hors réseau, en mm, lue quand lakes.residence_from_volume est vrai."),
    "lakes.residence_days": Setting("MEANDRE_LACS_SEJOUR", float, 30.0, "Temps de séjour uniforme des lacs hors réseau à la hauteur de référence, en jours, lu quand lakes.residence_from_volume est faux."),
    "lakes.outflow_exponent": Setting("MEANDRE_LACS_EXPOSANT", float, 1.0, "Exposant beta de la loi de vidange des lacs hors réseau, q = f k h^beta, sans dimension."),
    "lakes.reference_height_m": Setting("MEANDRE_LACS_HAUTEUR_REF", float, 0.5, "Hauteur au-dessus du seuil à laquelle le temps de séjour est défini, k = 1 / (T h_ref^(beta − 1)), en mètres."),
    "lakes.share_bounds": Setting("MEANDRE_LACS_PART_BORNES", tuple, (0.5, 50.0), "Bornes douces (sigmoïde en logarithme) du coefficient lakes.share_coefficient, min et max."),
    "lakes.mobilizable_depth_bounds_mm": Setting("MEANDRE_LACS_HAUTEUR_BORNES", tuple, (20.0, 2000.0), "Bornes douces de la hauteur mobilisable lakes.mobilizable_depth_mm, en mm."),
    "lakes.residence_bounds_days": Setting("MEANDRE_LACS_SEJOUR_BORNES", tuple, (2.0, 120.0), "Bornes douces du séjour uniforme lakes.residence_days, en jours."),
    "lakes.learn": Setting("MEANDRE_LACS_APPRIS", list, (), "Paramètres appris des lacs hors réseau, parmi share_coefficient et celui du séjour en vigueur (mobilizable_depth_mm ou residence_days) ; vide, aucun. Points de reprise : lakes_learn_c et lakes_learn_T."),
    "lakes.spill_bypass": Setting("MEANDRE_LACS_DIFFUS_CONTOURNES", bool, True, "Le ruissellement des processus déclarés (débordement commandé par la nappe) contourne les lacs hors réseau ; faux, il les traverse comme le reste de la production (2026-11-05)."),
    "lakes.runoff_ratio": Setting("MEANDRE_LACS_DIFFUS_COEF", float, 0.5, "Coefficient d'écoulement qui convertit la précipitation moyenne du tronçon en écoulement moyen R, pour le séjour déduit du volume des lacs."),
    "evaporation.mod16_snow_scale_mm": Setting("ETL_ET_NEIGE_MM", str, None, "Échelle, en mm, de la pondération du terme MOD16 hors neige, w = exp(−SWE/échelle) ; absent, poids 1."),
    "forcing.synthetic": Setting("MEANDRE_FORCAGE_FICTIF", str, None, "Banc fictif : fichier CSV d'une météo fabriquée (colonnes P, Tmin, Tmax, Rn, u2, ea, une ligne par jour depuis le début du forçage), appliquée telle quelle à tous les tronçons à la place du forçage ; à combiner avec debug.simulate_end. Jamais dans un modèle livré (2026-11-03)."),
    "bench.variants": Setting("MEANDRE_BANC_VARIANTES", str, None, "Banc fictif en un seul processus : fichier TOML des variantes (multiplicateurs du champ, réglages) ; le pilote simule chacune avec son témoin sans orages, puis s'arrête (meandre/probes/synthetic_bench.py)."),
    "bench.real_variants": Setting("MEANDRE_BANC_REEL", str, None, "Banc sur débits réels : fichier TOML des variantes (multiplicateurs, réglages, lacs hors réseau, paramètres des processus déclarés) ; le pilote charge le domaine une fois, simule chaque variante sur le forçage réel, écrit les débits aux stations de la période d'évaluation (q-<nom>.npz), puis s'arrête (2026-11-04)."),
    "bench.output": Setting("MEANDRE_BANC_SORTIE", str, ".", "Dossier des grandeurs par tronçon du banc fictif, un parquet par variante."),
    "groundwater.transmissivity_feedback_mm_per_day": Setting("MEANDRE_NAPPE_TRANSMISSIVITE", float, 0.0, "Retroaction de transmissivite de la nappe libre : ecoulement lateral q0 exp(-z / m) quand la surface libre affleure, en mm/j a z = 0 ; 0 l'eteint (2026-11-09)."),
    "groundwater.transmissivity_decay_m": Setting("MEANDRE_NAPPE_TRANSMISSIVITE_M", float, 0.5, "Profondeur m de decroissance de la retroaction de transmissivite, en metres."),
    "groundwater.connectivity_head_m": Setting("MEANDRE_NAPPE_CONNECTIVITE", float, 0.0, "Connectivite versant-riviere : l'ecoulement hypodermique n'atteint la riviere que sur la part min(1, h / h_c) du versant, h la charge de la nappe au-dessus du lit ; le reste recharge la nappe. h_c en metres ; 0 l'eteint (2026-11-09)."),
    "groundwater.exchange_mm_per_day": Setting("MEANDRE_NAPPE_ECHANGE", float, 0.0, "Echange souterrain de la nappe libre, en mm/j : positif, perte vers un aquifere profond ou un bassin voisin ; negatif, gain ; 0 l'eteint (2026-11-09)."),
    "period.train_start": Setting("MEANDRE_DEBUT_ENTRAINEMENT", str, "2001-01-01", "Début de la période d'entraînement ; les jours depuis le début du forçage (2000-01-01) servent de mise en régime, simulés sans perte. Jusqu'au 2026-11-06, l'entraînement commençait au 2000-01-01, sans mise en régime."),
    "period.train_end": Setting("MEANDRE_FIN_ENTRAINEMENT", str, "2018-12-31", "Fin de la période d'entraînement ; JOINT_SPLIT la surcharge."),
    "period.val_start": Setting("MEANDRE_DEBUT_VALIDATION", str, "2019-01-01", "Début de la validation, qui choisit le point de reprise."),
    "period.val_end": Setting("MEANDRE_FIN_VALIDATION", str, "2021-12-31", "Fin de la validation."),
    "period.eval_start": Setting("MEANDRE_DEBUT_EVALUATION", str, "2022-01-01", "Début de l'évaluation, jamais vue à l'entraînement ; ETL_HELDOUT la surcharge."),
    "period.eval_end": Setting("MEANDRE_FIN_EVALUATION", str, "2024-12-31", "Fin de l'évaluation."),
    "forcing.suffix": Setting("JOINT_FX_SUFFIX", str, None, "Forçage météo d'entrée : suffixe du fichier, -casr-brut pour CaSR brut."),
    "groundwater.krec": Setting("ETL_KREC", str, None, "Constante de recharge imposée, uniforme ; absent, celle du champ."),
    "groundwater.krec_frozen": Setting("ETL_KREC_GEL", str, None, "Constante de recharge imposée gelée de bout en bout, y compris après le chargement d'un point de reprise."),
    "groundwater.krec_prior": Setting("ETL_KREC_PRIOR", str, None, "Valeur de krec visée par le prior physique ; absent, défaut du prior."),
    # -- Constantes de la physique, sorties du code le 2026-11-05 : défaut = valeur jusque-là en dur --
    "column.campbell_b_bounds": Setting("MEANDRE_CAMPBELL_B_BORNES", tuple, (1.4, 6.0), "Bornes de l'exposant b de Campbell, appris par couche, sans dimension."),
    "column.campbell_psi_s_bounds_m": Setting("MEANDRE_CAMPBELL_PSIS_BORNES", tuple, (0.01, 1.0), "Bornes de la succion d'entrée d'air psi_s de Campbell, apprise par couche, en mètres."),
    "column.campbell_initial_texture": Setting("MEANDRE_CAMPBELL_TEXTURE_INIT", str, "sandy_loam", "Texture de SOIL_TEXTURES (bv3c2) dont b = 1/lambda et psi_s donnent la valeur de départ des trois couches."),
    "column.etr_alpha_init": Setting("MEANDRE_ETR_ALPHA_INIT", float, 4.5, "Valeur de départ du coefficient d'assèchement alpha de l'ETR (appris, global), sans dimension."),
    "column.etr_des": Setting("MEANDRE_ETR_DES", float, 0.6, "Paramètre des de l'ETR d'Hydrotel, sans dimension."),
    "column.etr_drying_coefficient": Setting("MEANDRE_ETR_COEF_ASSECH", float, 1.0, "Coefficient d'assèchement coef_assech de l'ETR d'Hydrotel, sans dimension."),
    "column.courant_variation": Setting("MEANDRE_CIN", float, 0.03, "Variation relative maximale de la teneur en eau par sous-pas du sol (condition de Courant, cin d'Hydrotel), sans dimension."),
    "column.slope_default_pct": Setting("MEANDRE_PENTE_DEFAUT", float, 4.0, "Pente du tronçon, en pour cent, quand la base ne porte ni slope_fraction ni mean_slope_pct_raw."),
    "column.slope_bounds": Setting("MEANDRE_PENTE_BORNES", tuple, (0.001, 0.5), "Bornes de la pente du tronçon, en fraction (m/m)."),
    "column.aspect_default_code": Setting("MEANDRE_ORIENTATION_DEFAUT", float, 7.0, "Code d'orientation (0 à 7, par pas de 45 degrés) quand la base ne porte pas sin_aspect et cos_aspect."),
    "column.penman_snow_swe_scale_mm": Setting("MEANDRE_PENMAN_NEIGE_ECHELLE", float, 5.0, "Échelle, en mm d'équivalent en eau, de la fraction enneigée f = 1 − exp(−SWE/échelle) du couvert nival dans Penman."),
    "column.penman_forcing_albedo": Setting("MEANDRE_PENMAN_ALBEDO_FORCAGE", float, 0.23, "Albédo supposé par le rayonnement net du forçage, sans dimension ; sous la neige la demande est multipliée par (1 − albédo de la neige)/(1 − cet albédo)."),
    "snow.melt_scale_reference": Setting("MEANDRE_FONTE_ECHELLE_REF", float, 4.5, "Valeur de C_f, en mm/(degré·j), qui donne une échelle de fonte spatiale de 1 : échelle = C_f / référence."),
    "snow.melt_scale_bounds": Setting("MEANDRE_FONTE_ECHELLE_BORNES", tuple, (0.15, 1.8), "Bornes de l'échelle de fonte spatiale C_f / référence, sans dimension."),
    "snow.seasonal_melt_phase_day": Setting("MEANDRE_FONTE_SAISON_PHASE", float, 81.0, "Jour de l'année où la modulation saisonnière de la fonte vaut 1 en montant, s(j) = 1 + a sin(2π(j − jour)/365) ; 81 place le maximum au solstice d'été."),
    "wetland.max_depth_m": Setting("MEANDRE_MH_WETDMAX", float, 0.3, "Profondeur maximale du milieu humide isolé (wetdmax), en mètres, quand le projet ne la donne pas."),
    "wetland.area_fraction": Setting("MEANDRE_MH_FRAC", float, 0.8, "Part de l'aire drainée par le milieu humide isolé (frac), sans dimension, quand le projet ne la donne pas."),
    "wetland.normal_depth_m": Setting("MEANDRE_MH_WETDNOR", float, 0.2, "Profondeur normale du milieu humide isolé (wetdnor), en mètres, quand le projet ne la donne pas."),
    "wetland.bottom_ksat": Setting("MEANDRE_MH_KSAT_BS", float, 0.5, "Conductivité du fond du milieu humide (ksat_bs, unité du projet Hydrotel), quand le projet ne la donne pas."),
    "wetland.evaporation_coefficient": Setting("MEANDRE_MH_C_EV", float, 0.6, "Coefficient d'évaporation du milieu humide (c_ev), sans dimension, quand le projet ne le donne pas."),
    "wetland.production_days": Setting("MEANDRE_MH_C_PROD", float, 10.0, "Durée de restitution du stock au-dessus du volume normal (c_prod), en jours, quand le projet ne la donne pas."),
    "groundwater.water_table_substeps": Setting("MEANDRE_NAPPE_SOUSPAS", int, 4, "Sous-pas par jour de la nappe libre."),
    "groundwater.water_table_coupling": Setting("MEANDRE_NAPPE_COUPLAGE", float, 0.0, "Couplage nappe-colonne : la percolation de la couche 3 cesse quand la nappe remonte jusqu'à la base du sol ; 0, pas de couplage."),
    "groundwater.water_table_learned": Setting("MEANDRE_NAPPE_APPRISE", bool, False, "Conductance, extraction et exposant de la nappe appris (trois scalaires) ; faux, valeurs posées."),
    "routed_lakes.area_km2_default": Setting("MEANDRE_LAC_AIRE_DEFAUT", float, 1.0, "Surface d'un lac routé sans bloc lac dans troncon.trl, en km²."),
    "routed_lakes.k_default": Setting("MEANDRE_LAC_K_DEFAUT", float, 1e-4, "Coefficient de vidange k_lake d'un lac routé sans loi de tarage, et centre de la tête du champ sans ancre, en 1/s (Q = k (S/A)^beta A)."),
    "routed_lakes.k_bounds": Setting("MEANDRE_LAC_K_BORNES", tuple, (1e-6, 1e-2), "Bornes de k_lake prédit par le champ, en 1/s."),
    "routed_lakes.beta_default": Setting("MEANDRE_LAC_BETA_DEFAUT", float, 1.5, "Exposant beta de la vidange d'un lac routé sans loi de tarage, et valeur de départ de la tête du champ, sans dimension."),
    "routed_lakes.beta_bounds": Setting("MEANDRE_LAC_BETA_BORNES", tuple, (1.0, 2.5), "Bornes de l'exposant beta prédit par le champ, sans dimension."),
    "routed_lakes.dead_storage_m": Setting("MEANDRE_LAC_STOCK_MORT", float, 0.0, "Hauteur d'eau morte des lacs routés, sous laquelle rien ne sort, en mètres."),
    "drainage.class_threshold": Setting("MEANDRE_DRAINAGE_SEUIL", float, 4.5, "Classe de drainage (1 très rapide à 7 très mal drainé) au-dessus de laquelle un tronçon est mal drainé, pour drainage_select des processus déclarés ; un processus peut poser son propre drainage_threshold."),
    "phenology_constants.gdd_base_c": Setting("MEANDRE_PHENO_TBASE", float, 10.0, "Température de base des degrés-jours de la phénologie, en degrés."),
    "phenology_constants.kc_min_init": Setting("MEANDRE_PHENO_KCMIN", float, 0.3, "Valeur de départ du K_c de dormance (appris), sans dimension."),
    "phenology_constants.kc_max_factor_init": Setting("MEANDRE_PHENO_KCMAX", float, 1.0, "Valeur de départ de l'amplificateur du K_c au plateau (appris), sans dimension."),
    "phenology_constants.kc_min_bounds": Setting("MEANDRE_PHENO_KCMIN_BORNES", tuple, (0.05, 1.0), "Bornes du K_c de dormance ; la borne basse est aussi le plancher du K_c effectif."),
    "phenology_constants.kc_max_factor_bounds": Setting("MEANDRE_PHENO_KCMAX_BORNES", tuple, (0.5, 2.0), "Bornes de l'amplificateur du K_c au plateau, sans dimension."),
    "phenology_constants.senescence_start_doy": Setting("MEANDRE_PHENO_JOUR_SENESCENCE", int, 172, "Jour de l'année après lequel la longueur du jour peut déclencher la sénescence (172, solstice d'été)."),
    "phenology_constants.photoperiod_center_h": Setting("MEANDRE_PHENO_PHOTO_CENTRE", float, 12.0, "Centre du seuil de longueur du jour de la sénescence, en heures : seuil = centre + demi-plage · tanh(photo_crit)."),
    "phenology_constants.photoperiod_half_range_h": Setting("MEANDRE_PHENO_PHOTO_DEMI_PLAGE", float, 3.0, "Demi-plage du seuil de longueur du jour de la sénescence, en heures."),
    "field_constraints.log_slope": Setting("MEANDRE_CHAMP_PENTE_LOG", float, 0.3, "Pente au centre, en logarithme naturel par unité de sortie brute, des sorties log-normales du champ (K_sat, k_gw, krec, k_sub) et des bornes par tronçon."),
    "field_constraints.log_center_K_sat_1": Setting("MEANDRE_CHAMP_CENTRE_KSAT1", float, 0.5, "Valeur donnée par une sortie brute nulle pour K_sat_1, en m/j."),
    "field_constraints.log_center_K_sat_2": Setting("MEANDRE_CHAMP_CENTRE_KSAT2", float, 0.1, "Valeur donnée par une sortie brute nulle pour K_sat_2, en m/j."),
    "field_constraints.log_center_K_sat_3": Setting("MEANDRE_CHAMP_CENTRE_KSAT3", float, 0.02, "Valeur donnée par une sortie brute nulle pour K_sat_3, en m/j."),
    "field_constraints.log_center_k_gw": Setting("MEANDRE_CHAMP_CENTRE_KGW", float, 0.02, "Valeur donnée par une sortie brute nulle pour k_gw, en 1/j."),
    "field_constraints.log_center_krec": Setting("MEANDRE_CHAMP_CENTRE_KREC", float, 2e-5, "Valeur donnée par une sortie brute nulle pour krec, en m/h."),
    "field_constraints.log_center_k_sub": Setting("MEANDRE_CHAMP_CENTRE_KSUB", float, 1.0e-3 / 24.0, "Valeur donnée par une sortie brute nulle pour k_sub, en m/h (1 mm/j)."),
    "field_constraints.literature_weight_shrink": Setting("MEANDRE_CHAMP_RETRECISSEMENT", float, 0.1, "Facteur appliqué aux poids de la couche de sortie du champ à l'initialisation par la littérature : tous les tronçons partent près des mêmes valeurs."),
    "physical_prior_weights.K_sat": Setting("MEANDRE_PRIOR_POIDS_KSAT", float, 0.3, "Poids relatif, dans le prior de la littérature ([training] w_prior), de l'écart de la moyenne du log de K_sat de chaque couche."),
    "physical_prior_weights.porosity": Setting("MEANDRE_PRIOR_POIDS_POROSITE", float, 1.0, "Poids relatif de l'écart de la porosité moyenne de chaque couche."),
    "physical_prior_weights.C_f": Setting("MEANDRE_PRIOR_POIDS_CF", float, 0.3, "Poids relatif de l'écart du facteur de fonte moyen C_f."),
    "physical_prior_weights.T_melt": Setting("MEANDRE_PRIOR_POIDS_TMELT", float, 0.5, "Poids relatif de l'écart du seuil de fonte moyen T_melt."),
    "physical_prior_weights.dT_canopee": Setting("MEANDRE_PRIOR_POIDS_DTCANOPEE", float, 0.1, "Poids relatif de l'écart des retards de canopée moyens (feuillus et conifères)."),
    "physical_prior_weights.vg_n": Setting("MEANDRE_PRIOR_POIDS_VGN", float, 0.3, "Poids relatif de l'écart de vg_n moyen."),
    "physical_prior_weights.k_gw": Setting("MEANDRE_PRIOR_POIDS_KGW", float, 0.3, "Poids relatif de l'écart de la moyenne du log de k_gw."),
    "physical_prior_weights.krec": Setting("MEANDRE_PRIOR_POIDS_KREC", float, 0.3, "Poids relatif de l'écart de la moyenne du log de krec, quand krec est libre."),
    "physical_prior_weights.k_sub": Setting("MEANDRE_PRIOR_POIDS_KSUB", float, 0.3, "Poids relatif de l'écart de la moyenne du log de k_sub, quand son prior est actif."),
    "physical_prior_weights.K_c": Setting("MEANDRE_PRIOR_POIDS_KC", float, 0.2, "Poids relatif de l'écart du K_c moyen."),
    "priors.safety_porosity": Setting("MEANDRE_PRIOR_GARDE_POROSITE", tuple, (0.05, 0.75), "Plancher et plafond de sécurité de la porosité des couches 2 et 3 sous prior, sans dimension."),
    "priors.safety_porosity_1": Setting("MEANDRE_PRIOR_GARDE_POROSITE1", tuple, (0.05, 0.95), "Plancher et plafond de sécurité de la porosité de la couche 1 sous prior (horizon organique possible), sans dimension."),
    "priors.safety_theta_fc_frac": Setting("MEANDRE_PRIOR_GARDE_CC", tuple, (0.10, 0.98), "Plancher et plafond de sécurité de la capacité au champ sous prior, en fraction de la porosité."),
    "priors.safety_theta_wp_frac": Setting("MEANDRE_PRIOR_GARDE_PF", tuple, (0.05, 0.95), "Plancher et plafond de sécurité du point de flétrissement sous prior, en fraction de la capacité au champ."),
    "priors.safety_K_sat": Setting("MEANDRE_PRIOR_GARDE_KSAT", tuple, (1e-5, 500.0), "Plancher et plafond de sécurité d'une conductivité sous prior, en m/j."),
    "priors.safety_Z2": Setting("MEANDRE_PRIOR_GARDE_Z2", tuple, (0.02, 5.0), "Plancher et plafond de sécurité de l'épaisseur de la couche 2 sous prior, en mètres."),
    "priors.safety_Z3": Setting("MEANDRE_PRIOR_GARDE_Z3", tuple, (0.05, 10.0), "Plancher et plafond de sécurité de l'épaisseur de la couche 3 sous prior, en mètres."),
    "loss_terms.kge_std_floor": Setting("MEANDRE_PERTE_PLANCHER_KGE", float, 1e-3, "Plancher des écarts-types du KGE, en fraction du débit moyen de la station."),
    "loss_terms.low_flow_quantile": Setting("MEANDRE_PERTE_ETIAGE_QUANTILE", float, 0.3, "Quantile du débit observé sous lequel un jour compte comme jour d'étiage (w_low_flow)."),
    "loss_terms.low_flow_eps_m3s": Setting("MEANDRE_PERTE_ETIAGE_EPS", float, 1e-3, "Décalage ajouté au débit avant le logarithme du terme d'étiage, en m³/s."),
    "loss_terms.fdc_low_quantiles": Setting("MEANDRE_PERTE_FDC_QUANTILES", tuple, (0.2, 0.5), "Quantiles bas et de référence du soutien d'étiage (w_fdc_low), rapport Q(bas)/Q(référence)."),
    "loss_terms.log_offset_m3s": Setting("MEANDRE_PERTE_LOG_DECALAGE", float, 1.0, "Décalage ajouté au débit avant le logarithme de l'écart quadratique en log (w_log_mse), en m³/s."),
    "loss_terms.et_window_days": Setting("MEANDRE_PERTE_ET_FENETRE", int, 8, "Fenêtre, en jours, de la moyenne glissante de l'ET simulée comparée aux composites MOD16 de 8 jours."),
    "loss_terms.min_station_obs": Setting("MEANDRE_PERTE_MIN_OBS", int, 30, "Nombre minimal d'observations, historique compris, pour qu'une station entre dans la perte d'un bloc."),
    "loss_terms.flat_min_days": Setting("MEANDRE_PERTE_PLAT_MIN_JOURS", int, 20, "Nombre minimal de jours des mois ciblés dans le bloc pour que le terme de platitude (w_flat) compte une station."),
    "loss_terms.peak_quantile": Setting("MEANDRE_PERTE_PICS_QUANTILE", float, 0.75, "Quantile du débit observé d'entraînement qui fixe le seuil des pics de chaque station (w_peak, w_peak_ratio)."),
    "loss_terms.station_stats_min_obs": Setting("MEANDRE_PERTE_STATS_MIN_OBS", int, 100, "Nombre d'observations d'entraînement qu'une station doit dépasser pour recevoir sa variance et son seuil de pics ; sinon variance 1 et aucun seuil."),
    "trainer.lr_mult_water_table": Setting("MEANDRE_LR_MULT_NAPPE", float, 20.0, "Multiplicateur du taux des scalaires de la nappe apprise."),
    "trainer.lr_mult_phenology": Setting("MEANDRE_LR_MULT_PHENOLOGIE", float, 100.0, "Multiplicateur du taux des seuils de la phénologie (degrés-jours, longueur du jour)."),
    # -- Diagnostics du gradient, jamais dans un modèle livré ----------------------------
    "debug.detach_dtc": Setting("MEANDRE_DTC_DETACHE", bool, False, "Détache le pas de temps adaptatif du sol ; valeur simulée identique."),
    "debug.detach_cascade": Setting("MEANDRE_CASCADE_DETACHE", bool, False, "Détache la cascade de saturation du sol, pour mesurer sa part dans le gradient."),
    "debug.zero_padding": Setting("MEANDRE_REMBOURRAGE_NUL", bool, False, "Poids nuls pour les colonnes d'entrée ajoutées à un réseau chargé : le modèle chargé reste identique."),
    "debug.kge_trace": Setting("MEANDRE_DEBUG_KGE", bool, False, "Trace de contrôle du terme KGE de la perte."),
    "debug.nan_level": Setting("MEANDRE_DEBUG_NAN", str, "0", "Recherche des valeurs non numériques : 1 nomme l'opération de la passe arrière, 2 pose le mode anomalie (lent)."),
    "debug.grad": Setting("MEANDRE_DEBUG_GRAD", bool, False, "Norme du gradient de chaque terme sur les trois premiers blocs."),
    "debug.grad_stop": Setting("MEANDRE_DEBUG_GRAD_STOP", bool, False, "Arrêt après les trois blocs de debug.grad."),
    "debug.detach_day": Setting("MEANDRE_DETACHE_JOUR", str, "", "Détache l'état de la colonne aux jours listés."),
}

# Sections lues par le pilote lui-même et transmises au modèle : `configure` ne les lit pas,
# et la valeur transmise prime sur l'alias d'environnement, qui ne sert qu'en son absence.
PILOT_SECTIONS = frozenset({"phenology"})

_configured: dict[str, object] = {}


def _parse(key: str, raw: str):
    """Convertit une valeur lue dans l'environnement, et refuse ce qui est ambigu."""
    kind = SCHEMA[key].kind
    if kind is bool:
        if raw not in ("0", "1"):
            raise ValueError(f"réglage {key} ({SCHEMA[key].env}) : '{raw}' n'est ni 0 ni 1")
        return raw == "1"
    if kind is str:
        return raw
    if raw == "":
        return SCHEMA[key].default
    if kind is tuple:
        return tuple(float(x) for x in raw.split(","))
    if kind is list:
        # Liste de noms : séparés par des virgules dans l'environnement, rendue en tuple.
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return kind(raw)


def _from_file(key: str, value):
    kind = SCHEMA[key].kind
    if kind is str:
        # Texte, comme une variable d'environnement : vrai et faux s'écrivent 1 et 0.
        return ("1" if value else "0") if isinstance(value, bool) else str(value)
    if kind is bool and not isinstance(value, bool):
        raise ValueError(f"réglage {key} : {value!r} n'est pas un booléen")
    if kind is tuple:
        return tuple(float(x) for x in value)
    if kind is list:
        if isinstance(value, str) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"réglage {key} : {value!r} n'est pas une liste de noms")
        return tuple(value)
    return kind(value) if kind in (int, float) else value


def configure(sections: dict | None) -> list[str]:
    """Résout tous les réglages à partir du fichier de configuration chargé.

    `sections` est le document TOML entier : chaque réglage y est cherché dans la section
    qui porte le début de sa clé. Retourne les clés prises dans le fichier. Une clé d'une
    section déclarée ici, absente de SCHEMA, arrête tout : elle ne serait lue par personne.
    """
    sections = sections or {}
    for name in {k.split(".")[0] for k in SCHEMA} - PILOT_SECTIONS:
        for key in (sections.get(name) or {}):
            if f"{name}.{key}" not in SCHEMA:
                raise ValueError(f"réglage inconnu [{name}] {key} : aucune entrée de SCHEMA ne le déclare")
    from_file = []
    _configured.clear()
    for key, s in SCHEMA.items():
        section, name = key.split(".", 1)
        raw = os.environ.get(s.env)
        if section in PILOT_SECTIONS:
            continue
        if raw is not None:
            _configured[key] = _parse(key, raw)
        elif name in (sections.get(section) or {}):
            _configured[key] = _from_file(key, sections[section][name])
            from_file.append(key)
        else:
            _configured[key] = s.default
    return from_file


def get(key: str):
    """Valeur d'un réglage : résolue par `configure`, sinon alias d'environnement, sinon défaut."""
    if key in _configured:
        return _configured[key]
    s = SCHEMA[key]
    raw = os.environ.get(s.env)
    return _parse(key, raw) if raw is not None else s.default


def override(key: str, value, force: bool = False) -> None:
    """Fixe un réglage en cours d'exécution, sauf si son alias d'environnement est posé.

    `force` passe outre l'environnement, pour un réglage que le pilote calcule toujours lui-même.

    Pour un pilote qui décide d'un réglage après avoir lu son fichier, par exemple des poids
    nuls pour des colonnes d'entrée qu'il vient d'ajouter. L'environnement garde la priorité.
    """
    if force or os.environ.get(SCHEMA[key].env) is None:
        _configured[key] = value


def raw(key: str, default=None):
    """Valeur d'un réglage de type texte, ou `default` s'il n'est posé nulle part.

    Remplace `os.environ.get(alias, default)` mot pour mot dans le pilote : la valeur arrive
    sous la même forme, si bien que les tests existants (`== "1"`, `float(...)`) ne changent pas.
    """
    v = get(key)
    return default if v is None else v


def snapshot() -> dict[str, object]:
    """Valeurs en vigueur qui s'écartent du défaut, pour la recette du point de reprise."""
    out = {}
    for key, s in SCHEMA.items():
        v = get(key)
        if v != s.default:
            out[key] = v
    return out
