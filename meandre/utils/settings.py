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
    # -- Diagnostics du gradient, jamais dans un modèle livré ----------------------------
    "debug.detach_dtc": Setting("MEANDRE_DTC_DETACHE", bool, False, "Détache le pas de temps adaptatif du sol ; valeur simulée identique."),
    "debug.detach_cascade": Setting("MEANDRE_CASCADE_DETACHE", bool, False, "Détache la cascade de saturation du sol, pour mesurer sa part dans le gradient."),
    "debug.zero_padding": Setting("MEANDRE_REMBOURRAGE_NUL", bool, False, "Poids nuls pour les colonnes d'entrée ajoutées à un réseau chargé : le modèle chargé reste identique."),
    "debug.nan_mask": Setting("MEANDRE_MASQUE_NAN", bool, False, "Témoin : masquage par NaN d'avant le 2026-09-08."),
    "debug.kge_trace": Setting("MEANDRE_DEBUG_KGE", bool, False, "Trace de contrôle du terme KGE de la perte."),
    "debug.detach_day": Setting("MEANDRE_DETACHE_JOUR", str, "", "Détache l'état de la colonne aux jours listés."),
}

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
    for name in {k.split(".")[0] for k in SCHEMA}:
        for key in (sections.get(name) or {}):
            if f"{name}.{key}" not in SCHEMA:
                raise ValueError(f"réglage inconnu [{name}] {key} : aucune entrée de SCHEMA ne le déclare")
    from_file = []
    _configured.clear()
    for key, s in SCHEMA.items():
        section, name = key.split(".", 1)
        raw = os.environ.get(s.env)
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


def snapshot() -> dict[str, object]:
    """Valeurs en vigueur qui s'écartent du défaut, pour la recette du point de reprise."""
    out = {}
    for key, s in SCHEMA.items():
        v = get(key)
        if v != s.default:
            out[key] = v
    return out
