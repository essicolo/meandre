"""Observables déclarés : ce que le modèle compare, à quoi, et comment (étape 4, 2026-10-26).

Une recette déclare chaque observable dans sa propre section :

    [observations.et]
    weight = 0.4
    mode = "anomaly"          # centré sur la moyenne de longue durée par nœud
    snow_scale_mm = 10.0      # poids exp(-SWE/échelle) : MOD16 peu fiable sous la neige

    [observations.tws]
    weight = 0.0
    mode = "shape"            # forme seulement, chaque côté réduit par son écart-type

Le catalogue dit, pour chaque observable : l'observation du domaine qu'il lit, le diagnostic de
la colonne qu'il compare, les modes de comparaison admis, et sa traduction vers les réglages de
la perte. Le calcul des termes reste celui de l'entraîneur ; ce module en fait une déclaration
unique, vérifiée, et refuse ce qui ne serait lu par personne : un observable inconnu, un mode
non admis, une clé en trop, ou un même poids déclaré deux fois avec deux valeurs.

Le débit n'est pas ici : ses termes (KGE, volume, étiage...) restent dans `[loss]`, parce qu'ils
sont plusieurs façons de comparer une même série et non des observables distincts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Observable:
    domain_key: str              # nom de l'observation dans Domain.observations
    diagnostic: str              # diagnostic de la colonne comparé (SimDiagnostics)
    weight_key: str              # poids correspondant dans la perte ([loss])
    modes: tuple[str, ...]       # modes de comparaison admis ; le premier est le défaut
    origin: str                  # produit de référence, en clair
    doc: str
    extra: dict = field(default_factory=dict)   # clé de section -> (section de destination, clé)


CATALOGUE: dict[str, Observable] = {
    "et": Observable("et", "etr", "w_et", ("anomaly", "level", "bassin", "forme"), "MOD16", "Évapotranspiration réelle ; comparée centrée par nœud (jamais en niveau sur CaSR).", {"snow_scale_mm": ("evaporation", "mod16_snow_scale_mm")}),
    "tws": Observable("tws", "theta1+theta2+theta3+swe+s_gw+canopy+wet_vol", "w_tws", ("shape", "level"), "GRACE", "Stock d'eau total du domaine, mensuel ; en forme seulement."),
    "swe_mass": Observable("swe_mass", "swe", "w_swe_mass", ("level",), "CanSWE", "Masse du manteau aux sites du réseau au sol."),
    "snow": Observable("snow", "swe", "w_snow", ("cover",), "MOD10", "Couverture nivale ; fraction simulée 1 - exp(-SWE/référence)."),
    "wells": Observable("wells", "profondeur_nappe", "w_nappe", ("shape",), "RSESQ", "Niveaux de nappe, en anomalies réduites et centrées."),
}
KEYS = {"weight", "mode", "snow_scale_mm"}


def translate(sections: dict | None, loss: dict, training: dict) -> tuple[dict, dict, dict, list[str]]:
    """Traduit les sections [observations.*] vers les réglages de la perte et de l'entraînement.

    Retourne (loss, training, autres, messages) : copies modifiées des sections [loss] et
    [training], réglages déclarés à poser ailleurs ({section: {clé: valeur}}), et une phrase par
    observable déclaré. Lève ValueError sur tout ce qui ne serait lu par personne.
    """
    loss, training = dict(loss), dict(training)
    others: dict[str, dict] = {}
    messages = []
    for name, sec in (sections or {}).items():
        if name not in CATALOGUE:
            raise ValueError(f"[observations.{name}] : observable inconnu ; catalogue : {sorted(CATALOGUE)}")
        o = CATALOGUE[name]
        unknown = set(sec) - KEYS - set(o.extra)
        if unknown:
            raise ValueError(f"[observations.{name}] : clé(s) inconnue(s) {sorted(unknown)}")
        w = float(sec.get("weight", 0.0))
        if o.weight_key in loss and float(loss[o.weight_key]) != w:
            raise ValueError(f"[observations.{name}] weight = {w} et [loss] {o.weight_key} = {loss[o.weight_key]} : un seul endroit")
        loss[o.weight_key] = w
        mode = str(sec.get("mode", o.modes[0]))
        if mode not in o.modes:
            raise ValueError(f"[observations.{name}] mode {mode!r} non admis ; modes : {o.modes}")
        if name == "et":
            if "et_mode" in loss and loss["et_mode"] != mode:
                raise ValueError(f"[observations.et] mode = {mode!r} et [loss] et_mode = {loss['et_mode']!r} : un seul endroit")
            loss["et_mode"] = mode
        if name == "tws":
            shape = mode == "shape"
            if "tws_shape_only" in training and bool(training["tws_shape_only"]) != shape:
                raise ValueError(f"[observations.tws] mode = {mode!r} et [training] tws_shape_only = {training['tws_shape_only']} : un seul endroit")
            training["tws_shape_only"] = shape
        for k, (dest, dk) in o.extra.items():
            if k in sec:
                others.setdefault(dest, {})[dk] = sec[k]
        messages.append(f"{name} ({o.origin}) : poids {w:g}, mode {mode}")
    return loss, training, others, messages


def missing(declared: dict | None, observations: dict) -> list[str]:
    """Observables déclarés à poids positif dont le domaine ne porte pas l'observation."""
    out = []
    for name, sec in (declared or {}).items():
        o = CATALOGUE.get(name)
        if o and float(sec.get("weight", 0.0)) > 0 and o.domain_key not in observations:
            out.append(f"{name} : poids {float(sec['weight']):g} mais le domaine ne porte pas d'observation « {o.domain_key} »")
    return out
