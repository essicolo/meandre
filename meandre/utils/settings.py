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
    "column.z_tensor": Setting("MEANDRE_Z_TENSEUR", bool, True, "Épaisseurs Z2 et Z3 en tenseurs sur la voie racinaire."),
    "column.penman_snow": Setting("MEANDRE_PENMAN_NEIGE", bool, False, "Couvert nival dans l'ETP de Penman (R265)."),
    "column.transpiration_soil_temp_ref": Setting("MEANDRE_TRANSP_TSOL_REF", float, None, "Échelle, en degrés, du frein de la transpiration par le sol froid (R264) ; absent, pas de frein."),
    "column.etr_faithful": Setting("MEANDRE_ETR_FIDELE", bool, False, "ETR répartie par couche comme Hydrotel, sans correction."),
    "column.viscosity_tref": Setting("MEANDRE_VISCOSITE_TREF", float, None, "Température de référence, en degrés, de la correction de conductivité par la viscosité de l'eau (Vogel) ; absent, pas de correction."),
    "column.semi_implicit_soil": Setting("MEANDRE_SOL_SEMI_IMPLICITE", bool, False, "Schéma linéairement implicite du sol : gradient borné, graines reproductibles."),
    "column.freeze_without_snow": Setting("MEANDRE_GEL_SANS_NEIGE", bool, False, "Le gel ferme le sol même sans manteau neigeux."),
    "column.frozen_gate_amplitude": Setting("MEANDRE_GEL_AMPLITUDE", float, 1.0, "Amplitude de la porte de gel : capacité d'infiltration ks1 (1 − a · fraction gelée), et même facteur sur le drainage de la couche 1 ; 1 reproduit la porte d'origine, qui ferme le sol dès que le gel atteint l'épaisseur de la couche 1 (2026-10-23)."),
    "column.saturated_area_exponent": Setting("MEANDRE_VSA_B", float, None, "Exposant b du ruissellement sur aire saturée, part (theta1/thetas1)^b (R386) ; absent, pas de ruissellement de saturation partiel."),
    "column.dynamic_compile": Setting("MEANDRE_COMPILE_DYNAMIQUE", bool, False, "Compilation du sol en formes dynamiques ; faux garde le mode fixe mesuré à x17,6."),
    "column.field_capacity_from_curve": Setting("MEANDRE_THETA_FC_COURBE", bool, False, "Capacité au champ de la couche 3 déduite de la courbe de rétention plutôt qu'apprise."),
    "column.mass_closure": Setting("BV3C_FERMETURE", bool, True, "Fermeture de masse du sol ; faux reproduit la fuite historique d'Hydrotel."),
    # -- Routage et calcul ----------------------------------------------------------------
    "routing.muskingum_k_hours": Setting("MEANDRE_KMUSK", tuple, (4.0, 48.0, 24.0), "Temps de transfert de Muskingum, en heures : borne basse, borne haute, valeur initiale."),
    "routing.lake_evaporation": Setting("MEANDRE_EVAP_LACS", bool, True, "Évaporation des lacs routés par temps sec."),
    "compute.operator_block": Setting("MEANDRE_OPERATOR_BLOCK", int, None, "Taille maximale d'un bloc de l'opérateur de routage ; absent, un bloc par étage. Résultat identique, mémoire bornée."),
    "compute.gradient_segments": Setting("MEANDRE_SEGMENT_GRADIENT", int, 0, "Longueur, en jours, des segments recalculés à la rétropropagation ; 0, pas de recalcul. Gradient identique, mémoire bornée."),
    # -- Tête de quantiles ----------------------------------------------------------------
    "quantile.log_step": Setting("MEANDRE_QUANTILE_LOG_STEP", float, 0.05, "Largeur de chaque niveau de quantile, dans le logarithme du débit."),
    "quantile.log_bound": Setting("MEANDRE_QUANTILE_LOG_BOUND", float, 4.0, "Un quantile extrême reste entre e^-b et e^b fois la médiane."),
    "quantile.additive": Setting("MEANDRE_QUANTILE_ADDITIF", bool, False, "Enveloppe additive en débit au lieu de multiplicative en logarithme."),
    # -- Perte ----------------------------------------------------------------------------
    "loss_options.pbias_continuous": Setting("MEANDRE_PBIAS_CONTINU", bool, True, "Biais de volume calculé sur la série continue (historique plus bloc), et non sur le bloc seul."),
    # -- Diagnostics de sortie --------------------------------------------------------------
    "diagnostics.cpu": Setting("MEANDRE_DIAG_CPU", bool, False, "Diagnostics journaliers gardés sur le processeur plutôt que sur la carte graphique."),
    "diagnostics.half_precision": Setting("MEANDRE_DIAG_F16", bool, False, "Diagnostics non dérivés gardés en demi-précision sur le processeur."),
    "diagnostics.retained": Setting("MEANDRE_DIAG_DERIVES", str, None, "Diagnostics gardés avec leur graphe, séparés par des virgules ; absent, tous ceux qu'un terme peut dériver."),
    # -- Données -----------------------------------------------------------------------------
    "data.withdrawals_table": Setting("MEANDRE_WITHDRAWALS_TABLE", str, "withdrawals", "Table des prélèvements lue dans la base de chaque territoire."),
    "data.territorial_extra": Setting("MEANDRE_TERRITORIAL_EXTRA", str, "", "Tables territoriales supplémentaires ajoutées aux attributs du champ, séparées par des virgules."),
    "data.field_capacity_from_calibration": Setting("MEANDRE_THETA_CC_CALAGE", bool, False, "Capacité au champ et point de flétrissement repris du calage d'Hydrotel."),
    # -- Entraîneur ---------------------------------------------------------------------------
    "trainer.lr_mult_heads": Setting("MEANDRE_LR_MULT_TETES", float, None, "Multiplicateur commun du taux des têtes du champ et des lacs ; absent, 10 pour la sortie du champ et lake_lr_mult pour les lacs."),
    "trainer.lr_mult_soil": Setting("MEANDRE_LR_MULT_SOL", float, 50.0, "Multiplicateur du taux du profil de sol appris."),
    "trainer.lr_mult_routing": Setting("MEANDRE_LR_MULT_ROUTAGE", float, 100.0, "Multiplicateur du taux de la tête du temps de transfert."),
    "trainer.lr_mult_lakes": Setting("MEANDRE_LR_MULT_LACS", float, 50.0, "Multiplicateur du taux des scalaires des lacs hors réseau."),
    "trainer.save_every_epoch": Setting("MEANDRE_SAUVER_EPOQUES", bool, False, "Un point de reprise par époque, pour évaluer toute la trajectoire."),
    "trainer.abandon": Setting("MEANDRE_ABANDON", str, None, "Arrêt époque:seuil si la métrique de sélection reste sous le seuil après cette époque."),
    "trainer.kge_continuous": Setting("MEANDRE_KGE_CONTINU", bool, True, "KGE calculé sur la séquence continue, historique détaché plus bloc courant (R67)."),
    "trainer.state_continuous": Setting("MEANDRE_ETAT_CONTINU", bool, True, "État riche de la colonne transmis d'un bloc au suivant (R64)."),
    "trainer.step_per_block": Setting("MEANDRE_PAS_PAR_BLOC", bool, False, "Un pas d'optimisation par bloc ; faux, un pas par époque (R78, R288)."),
    "trainer.history_priming": Setting("MEANDRE_HISTORIQUE_AMORCE", bool, False, "Passe sans gradient en début d'époque pour que chaque bloc voie le KGE de toute la période."),
    "trainer.et_center_ema": Setting("MEANDRE_ET_CENTRAGE_EMA", bool, False, "Témoin : référence de l'ET centrée par moyenne mobile, comme avant le 2026-10-02."),
    "trainer.et_reference_from_spinup": Setting("MEANDRE_ET_REF_MISE_EN_REGIME", bool, False, "Référence de l'ET centrée prise sur la mise en régime."),
    "trainer.et_level_weight": Setting("MEANDRE_W_ET_NIVEAU", float, 0.0, "Poids du terme qui pousse le niveau moyen de l'ET vers MOD16."),
    "trainer.groundwater_max_share": Setting("ETL_NAPPE_PART_MAX", float, 0.15, "Part maximale de la perte que peut prendre le terme des niveaux de nappe."),
    # -- Sondes de diagnostic -------------------------------------------------------------
    "probe.terms": Setting("MEANDRE_SONDE_TERMES", str, "", "Champs dont la sonde mesure le gradient de chaque terme, séparés par des virgules."),
    "probe.terms_only": Setting("MEANDRE_SONDE_TERMES_SEULS", str, "", "Termes retenus par la sonde, séparés par des virgules ; vide, tous."),
    "probe.terms_output": Setting("MEANDRE_SONDE_TERMES_SORTIE", str, "sonde_termes.npz", "Fichier de sortie de la sonde par terme."),
    "probe.conflict": Setting("MEANDRE_SONDE_CONFLIT", str, "", "Fichier de sortie de la sonde de conflit des gradients ; vide, sonde éteinte."),
    "probe.conflict_blocks": Setting("MEANDRE_SONDE_CONFLIT_BLOCS", int, 4, "Nombre de blocs mesurés par la sonde de conflit avant l'arrêt."),
    "probe.segment": Setting("MEANDRE_SONDE_SEGMENT", bool, False, "Sonde du recalcul par segments : gradient et mémoire du deuxième bloc."),
    # -- Phénologie : la section [phenology] est lue par le pilote, qui transmet ses valeurs
    # au modèle ; ces réglages ne servent qu'en son absence (voir PILOT_SECTIONS). ------------
    "phenology.mode": Setting("MEANDRE_PHENOLOGIE_MODE", str, "gdd", "Forme saisonnière de K_c : gdd (degrés-jours), photo (longueur du jour), modis (indice foliaire observé)."),
    "phenology.params": Setting("MEANDRE_PHENOLOGIE_PARAMS", str, None, "Paramètres gelés du mode photo, « emergence,seuil,largeur1,largeur2 »."),
    "debug.simulate_end": Setting("MEANDRE_FIN_SIMULATION", str, None, "Coupe la fenêtre simulée à cette date (AAAA-MM-JJ), depuis le début du forçage : contrôles d'identité et essais de fumée en quelques minutes au lieu de 25 ans ; jamais dans un modèle livré (2026-10-26)."),
    "spatial.priors": Setting("MEANDRE_PRIORS", bool, False, "Priors par tronçon sur le sol (porosités, capacité au champ, point de flétrissement, conductivités, épaisseurs) tirés de la texture et des dépôts du domaine ; le champ apprend un écart pénalisé ([training] w_deviation_prior). Les valeurs du modèle chargé sont conservées à l'adoption (2026-10-26)."),
    "priors.mode": Setting("MEANDRE_PRIOR_MODE", str, "anomaly", "Comparaison du champ à ses priors : level (niveau de la carte), anomaly (contrastes entre tronçons, niveau libre, comme MOD16), shape (motif seul, niveau et amplitude libres, comme GRACE). La dispersion mesurée contre l'IRDA est une dispersion centrée : elle correspond au mode anomaly (2026-10-26)."),
    "priors.sigma_k_sat": Setting("MEANDRE_PRIOR_SIGMA_KSAT", float, 1.609, "Écart-type, en logarithme naturel, du prior de conductivité : ln 5. Mesuré le 2026-10-26 contre la perméabilité des levés IRDA par tronçon : dispersion d'un facteur 2 au sud (Saint-Laurent sud-ouest, Montérégie, Outaouais), 2,8 à 5 au nord ; on retient le pire."),
    "priors.sigma_porosity": Setting("MEANDRE_PRIOR_SIGMA_POROSITE", float, 0.20, "Écart-type en logarithme du prior de porosité (pédotransfert de Saxton et Rawls) ; non mesuré contre une donnée indépendante."),
    "priors.sigma_theta_fc_frac": Setting("MEANDRE_PRIOR_SIGMA_CC", float, 0.20, "Écart-type en logarithme du prior de capacité au champ, en fraction de la porosité ; non mesuré."),
    "priors.sigma_theta_wp_frac": Setting("MEANDRE_PRIOR_SIGMA_PF", float, 0.30, "Écart-type en logarithme du prior de point de flétrissement, en fraction de la capacité au champ ; non mesuré."),
    "priors.sigma_z2": Setting("MEANDRE_PRIOR_SIGMA_Z2", float, 0.693, "Écart-type en logarithme du prior d'épaisseur de la couche 2 : ln 2 ; non mesuré."),
    "priors.sigma_z3": Setting("MEANDRE_PRIOR_SIGMA_Z3", float, 0.916, "Écart-type en logarithme du prior d'épaisseur de la couche 3 (épaisseur des dépôts SIGEOM) : ln 2,5 ; non mesuré."),
    "spatial.gradient_floor": Setting("MEANDRE_PLANCHER_GRADIENT", float, None, "Plancher de la dérivée de la sigmoïde des sorties nommées dans spatial.gradient_floor_outputs, au rétropropagé seulement : la valeur ne change pas, une sortie collée à sa borne reçoit encore un gradient (2026-10-26) ; absent, sigmoïde ordinaire."),
    "spatial.gradient_floor_outputs": Setting("MEANDRE_PLANCHER_GRADIENT_SORTIES", str, None, "Sorties du champ concernées par spatial.gradient_floor, séparées par des virgules (par exemple « Z2,Z3,porosity »)."),
    "spatial.extend_below": Setting("MEANDRE_PROLONGE_SOUS", str, None, "Prolongement sous la borne basse, « sortie:nouvelle_borne,... » (par exemple « Z2:0.10,Z3:0.20 ») : sous la plus basse sortie brute du domaine au premier appel, la valeur descend doucement vers la nouvelle borne ; les valeurs de départ sont conservées exactement (2026-10-26)."),
    "phenology.params_agri": Setting("MEANDRE_PHENOLOGIE_PARAMS_AGRI", str, None, "Calendrier propre à la classe agricole en mode photo, « levée,seuil,largeur1,largeur2 » ; absent, la classe suit le calendrier commun (2026-10-24)."),
    "phenology.target": Setting("MEANDRE_PHENOLOGIE_CIBLE", str, "kc", "Grandeur modulée : kc (coefficient cultural) ou lai (indice foliaire, transpiration seule)."),
    "phenology.classes": Setting("MEANDRE_PHENOLOGIE_CLASSES", str, None, "Classes d'occupation modulées, séparées par des virgules ; absent, toutes."),
    "phenology.lai_modis_csv": Setting("MEANDRE_LAI_MODIS_CSV", str, None, "Fichier de l'indice foliaire MODIS (colonnes date, lai_median) du mode modis."),
    # -- Pilote régional : réglages lus sous forme de texte, comme dans l'environnement --------
    "snow.wet_bulb_threshold": Setting("ETL_SEUIL_TWB", str, None, "Seuil de partage pluie-neige au thermomètre mouillé, en degrés (Stull 2011) ; absent, seuil sur la température de l'air."),
    "snow.seasonal_melt_amplitude": Setting("ETL_MELT_SAISON", str, None, "Amplitude de la modulation saisonnière du facteur de fonte, s(j) = 1 + a sin(2π(j−81)/365) ; absent ou 0, clone fidèle."),
    "snow.project_threshold": Setting("ETL_SEUIL_NEIGE", str, None, "Seuil pluie-neige repris du projet Hydrotel (1) ou non (0) ; absent, 1."),
    "groundwater.aquifer": Setting("ETL_AQUIFER", str, None, "Voie souterraine active (aquifère restituant ou nappe libre)."),
    "groundwater.free_recession": Setting("ETL_KREC_LIBRE", str, None, "Constante de récession apprise par le champ, moyenne ancrée au prior."),
    "groundwater.conductivity_field": Setting("ETL_KGW_FIELD", str, None, "Champ provincial de conductivité souterraine imposé en niveau."),
    "groundwater.conductivity_modulation": Setting("MEANDRE_KGW_MODULE", str, None, "Le réseau module le contraste autour du champ imposé ; 0, il l'écrase."),
    "groundwater.water_table": Setting("ETL_NAPPE_LIBRE", str, None, "Nappe libre dont la profondeur est l'état, à la place du réservoir restituant."),
    "groundwater.exponent": Setting("ETL_NAPPE_EXP", str, None, "Exposant de vidange de la nappe, ou « mesure » pour l'ancrer sur les récessions observées ; absent, 2."),
    "groundwater.specific_yield": Setting("ETL_NAPPE_SY", str, None, "Porosité de drainage de la nappe ; absent, 0,05."),
    "groundwater.conductance": Setting("ETL_NAPPE_KB", str, None, "Conductance de vidange de la nappe ; absent, 2e-3."),
    "groundwater.river_depth_m": Setting("ETL_NAPPE_ZRIV", str, None, "Profondeur du lit de la rivière sous la surface, en mètres ; absent, 8."),
    "groundwater.reference_head_m": Setting("ETL_NAPPE_HREF", str, None, "Charge de référence de la nappe, en mètres ; absent, 4."),
    "groundwater.extinction_depth_m": Setting("ETL_NAPPE_ZEXT", str, None, "Profondeur d'extinction de l'extraction par les racines, en mètres ; absent, 9."),
    "groundwater.extraction_fraction": Setting("ETL_NAPPE_EFRAC", str, None, "Part de l'évapotranspiration prise dans la zone saturée ; absent, 0,35."),
    "spatial.no_latent_codes": Setting("ETL_NO_LATENT", str, None, "Codes latents par tronçon coupés (non transférables)."),
    "spatial.init_from_hydrotel": Setting("ETL_INIT_HYDROTEL", str, None, "Départ du sol sur le calage d'Hydrotel : 0, 1, courbe ou sauf_ks ; absent, 0."),
    "spatial.global_normalization": Setting("JOINT_GLOBAL_NORM", str, None, "Attributs du territoire normalisés sur toute la province."),
    "evaporation.open_water": Setting("ETL_EVAP_EAU_LIBRE", str, None, "Évaporation de l'eau libre par temps sec (R285, R286)."),
    "evaporation.mod16_snow_scale_mm": Setting("ETL_ET_NEIGE_MM", str, None, "Échelle, en mm, de la pondération du terme MOD16 hors neige, w = exp(−SWE/échelle) ; absent, poids 1."),
    "forcing.suffix": Setting("JOINT_FX_SUFFIX", str, None, "Forçage météo d'entrée : suffixe du fichier, -casr-brut pour CaSR brut."),
    "groundwater.frost_factor": Setting("ETL_NAPPE_GEL", str, None, "Porte de gel sur la vidange de la nappe : facteur appliqué quand le gel atteint 30 cm (R316) ; absent, pas de porte."),
    "groundwater.conductance_field": Setting("ETL_NAPPE_KB_CHAMP", str, None, "Conductance de la nappe par tronçon, « référence_1/j,exposant » : k_b × (k_gw/référence)^exposant (R380) ; absent, uniforme."),
    "groundwater.kgw_prior": Setting("ETL_KGW", str, None, "Valeur de k_gw visée par le prior de la littérature (récessions des jauges) ; absent, défaut du prior."),
    "groundwater.kgw_scale": Setting("ETL_KGW_ECHELLE", str, None, "Facteur appliqué au champ de conductivité souterraine imposé ; absent, 1."),
    "groundwater.krec": Setting("ETL_KREC", str, None, "Constante de recharge imposée, uniforme ; absent, celle du champ."),
    "groundwater.krec_frozen": Setting("ETL_KREC_GEL", str, None, "Constante de recharge imposée gelée de bout en bout, y compris après le chargement d'un point de reprise."),
    "groundwater.krec_prior": Setting("ETL_KREC_PRIOR", str, None, "Valeur de krec visée par le prior physique ; absent, défaut du prior."),
    "evaporation.demand_scale": Setting("ETL_DEMAND_SCALE", str, None, "Facteur appliqué à la demande évaporative apprise ; absent, 1."),
    # -- Diagnostics du gradient, jamais dans un modèle livré ----------------------------
    "debug.detach_dtc": Setting("MEANDRE_DTC_DETACHE", bool, False, "Détache le pas de temps adaptatif du sol ; valeur simulée identique."),
    "debug.detach_cascade": Setting("MEANDRE_CASCADE_DETACHE", bool, False, "Détache la cascade de saturation du sol, pour mesurer sa part dans le gradient."),
    "debug.zero_padding": Setting("MEANDRE_REMBOURRAGE_NUL", bool, False, "Poids nuls pour les colonnes d'entrée ajoutées à un réseau chargé : le modèle chargé reste identique."),
    "debug.nan_mask": Setting("MEANDRE_MASQUE_NAN", bool, False, "Témoin : masquage par NaN d'avant le 2026-09-08."),
    "debug.kge_trace": Setting("MEANDRE_DEBUG_KGE", bool, False, "Trace de contrôle du terme KGE de la perte."),
    "debug.tws_sd_block": Setting("MEANDRE_TWS_SD_BLOC", bool, False, "Témoin : écart-type de GRACE en forme calculé sur le bloc."),
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
