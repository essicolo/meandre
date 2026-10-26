"""Vocabulaire des attributs de tronçon : nom, unité, obligation, rôle.

Un domaine porte ses attributs sous ces noms et aucun autre. Les noms sont ceux de la table
`territorial` des bases québécoises, parce que l'ordre des colonnes fixe la dimension d'entrée du
champ spatial et qu'un point de reprise ne se recharge qu'avec les mêmes colonnes ; les renommer
est un chantier de l'étape 3, quand le champ apprendra des écarts à des priors.

Deux familles : les attributs `field`, normalisés et donnés au champ spatial, et les attributs
`physical`, bruts, lus par la physique et le routage. Un attribut optionnel absent est accepté ;
un attribut hors vocabulaire est refusé, pour qu'un adaptateur ne glisse pas une colonne que
personne ne lit ou, pire, que le champ lit sans qu'on le sache.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute:
    unit: str
    role: str          # "field" (normalisé, entrée du champ) ou "physical" (brut, physique)
    required: bool
    doc: str


ATTRIBUTES: dict[str, Attribute] = {
    "drainage_area_km2": Attribute("km2", "field", True, "Aire drainée en amont du tronçon, exutoire compris."),
    "strahler_order": Attribute("1", "field", True, "Ordre de Strahler du tronçon."),
    "mean_slope_pct": Attribute("%", "field", True, "Pente moyenne du versant."),
    "mean_elevation_m": Attribute("m", "field", True, "Altitude moyenne du versant."),
    "sin_aspect": Attribute("1", "field", True, "Sinus de l'exposition moyenne."),
    "cos_aspect": Attribute("1", "field", True, "Cosinus de l'exposition moyenne."),
    "f_forest": Attribute("1", "field", True, "Fraction de forêt."),
    "f_agriculture": Attribute("1", "field", True, "Fraction de terres cultivées."),
    "f_urban": Attribute("1", "field", True, "Fraction imperméable."),
    "f_wetland": Attribute("1", "field", True, "Fraction de milieux humides."),
    "f_water": Attribute("1", "field", True, "Fraction d'eau libre."),
    "f_sand": Attribute("1", "field", True, "Fraction de sable de la texture de surface."),
    "f_silt": Attribute("1", "field", True, "Fraction de limon de la texture de surface."),
    "f_clay": Attribute("1", "field", True, "Fraction d'argile de la texture de surface."),
    "depth_to_bedrock_m": Attribute("m", "physical", False, "Profondeur du roc ; nulle partout dans les bases québécoises (attribut mort, 2026-10-25)."),
    "dist_to_outlet_km": Attribute("km", "field", True, "Distance à l'exutoire le long du réseau."),
    "lake_fraction": Attribute("1", "field", True, "Fraction de lacs du versant."),
    "area_km2_physical": Attribute("km2", "physical", True, "Aire du versant lue par la physique."),
    "area_km2_local": Attribute("km2", "physical", True, "Aire locale du tronçon, sans l'amont."),
    "slope_fraction": Attribute("1", "physical", True, "Pente du versant en fraction, lue par la physique."),
    "reach_length_m": Attribute("m", "physical", False, "Longueur du tronçon."),
}

# Fractions brutes d'occupation et de texture, suffixe `_raw` : brutes, jamais au champ, lues
# par la colonne pour la répartition par classe. Optionnelles ; le nom est celui de la classe.
RAW_SUFFIX = "_raw"
# Tables d'attributs supplémentaires jointes par le chargeur (data.territorial_extra) : les
# colonnes portent le préfixe `<table>__`. Optionnelles ; acceptées telles quelles.
EXTRA_SEPARATOR = "__"


def check_columns(columns: list[str]) -> list[str]:
    """Attributs obligatoires absents et colonnes hors vocabulaire, en phrases ; vide si tout va."""
    present = set(columns)
    issues = [f"attribut obligatoire absent : {n}" for n, a in ATTRIBUTES.items() if a.required and n not in present]
    for c in columns:
        if c in ATTRIBUTES or c == "node_idx" or c.endswith(RAW_SUFFIX) or EXTRA_SEPARATOR in c:
            continue
        issues.append(f"colonne hors vocabulaire : {c}")
    return issues


def field_columns(columns: list[str]) -> list[str]:
    """Colonnes données au champ spatial, dans l'ordre reçu."""
    return [c for c in columns if (c in ATTRIBUTES and ATTRIBUTES[c].role == "field") or EXTRA_SEPARATOR in c]


def physical_columns(columns: list[str]) -> list[str]:
    """Colonnes brutes lues par la physique, dans l'ordre reçu."""
    return [c for c in columns if (c in ATTRIBUTES and ATTRIBUTES[c].role == "physical") or c.endswith(RAW_SUFFIX)]
