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
    # -- Diagnostics du gradient, jamais dans un modèle livré ----------------------------
    "debug.detach_dtc": Setting("MEANDRE_DTC_DETACHE", bool, False, "Détache le pas de temps adaptatif du sol ; valeur simulée identique."),
    "debug.detach_cascade": Setting("MEANDRE_CASCADE_DETACHE", bool, False, "Détache la cascade de saturation du sol, pour mesurer sa part dans le gradient."),
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
    return kind(raw)


def _from_file(key: str, value):
    kind = SCHEMA[key].kind
    if kind is bool and not isinstance(value, bool):
        raise ValueError(f"réglage {key} : {value!r} n'est pas un booléen")
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


def snapshot() -> dict[str, object]:
    """Valeurs en vigueur qui s'écartent du défaut, pour la recette du point de reprise."""
    out = {}
    for key, s in SCHEMA.items():
        v = get(key)
        if v != s.default:
            out[key] = v
    return out
