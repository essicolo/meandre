"""Processus de sol DÉCLARÉS par couche, à la manière de Raven.

Pourquoi. Les sorties de la colonne étaient codées en dur, une branche par hypothèse, et dix-sept
variables d'environnement s'étaient accumulées pour les activer. Chaque question nouvelle ajoutait
un `if`. Raven prend le problème à l'envers : un profil de sol porte des couches, chaque couche
porte des processus, et chaque processus a une forme fonctionnelle NOMMÉE, choisie dans un
catalogue dont les plages de paramètres sont publiées. On déclare au lieu de brancher.

Les noms de formes sont ceux de Raven 3.8, délibérément, pour que la littérature s'applique sans
traduction et pour cesser de réinventer des lois. La sortie latérale ajoutée le 2026-09-19, qui
vide linéairement ce qui dépasse la capacité au champ, est exactement `BASE_THRESH_POWER` avec un
exposant de un ; elle avait été écrite à la main faute d'avoir regardé le catalogue.

Fidélité. Sans déclaration, la colonne garde le chemin de code d'origine et reproduit Hydrotel au
bit près : c'est la condition pour que les harnais de validation par UHRH contre le binaire C++
restent valables. La déclaration par défaut, `hydrotel_profile()`, décrit ce même comportement et
un test vérifie que les deux chemins coïncident exactement.

Vocabulaire. Raven nomme `baseflow` toute sortie latérale d'un réservoir de sol vers le cours
d'eau, y compris depuis une couche superficielle ; Hydrotel nomme la même chose hypodermique et
réserve le débit de base à ce qui sort de l'aquifère. Ce module suit Raven pour les noms de
formes et Hydrotel pour la destination des flux, qui reste celle du clone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Identifiants en anglais, y compris dans la configuration : c'est la convention du dépôt.
KIND_LATERAL = "lateral"
KIND_PERCOLATION = "percolation"
KINDS = (KIND_LATERAL, KIND_PERCOLATION)


def _above_field_capacity(ctx):
    """Eau gravitaire de la couche, en mètres : ce que la gravité draine par définition."""
    return torch.clamp(ctx["theta"] - ctx["theta_fc"], min=0.0) * ctx["thickness"]


def base_linear(ctx, prm):
    """Darcy le long du versant, forme d'Hydrotel pour la couche 2 : K(theta) · sin(pente) · z."""
    return ctx["conductivity"] * ctx["sin_slope"] * ctx["thickness"] * prm.get("factor", 1.0)


def base_thresh_power(ctx, prm):
    """Vidange de l'eau gravitaire avec sa propre constante de temps, exposant libre.

    Q = (S_grav / tau) · (S_grav / S_max)^(n-1), soit linéaire pour n = 1. Un exposant
    supérieur à un relâche vite à stock plein et lentement à stock bas, ce que le diagnostic
    du 2026-09-18 réclamait pour l'aquifère et qui vaut aussi ici.

    CONVERGENCE. La condition de Courant se resserre quand la couche reste pleine. Mesuré le
    2026-09-19 sur une couche de 2,70 m : à tau = 5 jours la loi est convergée dès 64 sous-pas,
    266,7 mm contre 267,9 à 512 ; à tau = 20 jours elle ne l'est pas, 168,2 contre 204,4.
    """
    s = _above_field_capacity(ctx)
    q = s / prm["tau"]
    n = prm.get("exponent", 1.0)
    if n != 1.0:
        s_max = torch.clamp(ctx["porosity"] - ctx["theta_fc"], min=1e-9) * ctx["thickness"]
        q = q * torch.clamp(s / s_max, min=0.0, max=1.0) ** (n - 1.0)
    return q


def perc_linear(ctx, prm):
    """Percolation d'Hydrotel : krec · z · theta, linéaire en teneur en eau.

    Conséquence mesurée et consignée : linéaire en theta, le flux ne s'annule jamais et la
    couche s'épingle à saturation, d'où une recharge plate. C'est la loi du clone fidèle, et
    la raison pour laquelle une forme à seuil a été ajoutée à côté.
    """
    return prm["krec"] * ctx["thickness"] * ctx["theta"]


def perc_thresh_power(ctx, prm):
    """Percolation de l'eau gravitaire seule, avec constante de temps et exposant libre."""
    return base_thresh_power(ctx, prm)


def perc_power_law(ctx, prm):
    """Percolation en puissance de la saturation : krec · z · theta_s · (theta/theta_s)^n."""
    ths = ctx["porosity"]
    return (prm["krec"] * ctx["thickness"] * ths
            * torch.clamp(ctx["theta"] / ths, min=0.0) ** prm["exponent"])


# Catalogue. La clé est le nom Raven, la valeur la fonction et les paramètres exigés.
FORMS = {
    "BASE_LINEAR": (base_linear, ()),
    "BASE_THRESH_POWER": (base_thresh_power, ("tau",)),
    "PERC_LINEAR": (perc_linear, ("krec",)),
    "PERC_THRESH_POWER": (perc_thresh_power, ("tau",)),
    "PERC_POWER_LAW": (perc_power_law, ("krec", "exponent")),
}

FORMS_BY_KIND = {
    KIND_LATERAL: ("BASE_LINEAR", "BASE_THRESH_POWER"),
    KIND_PERCOLATION: ("PERC_LINEAR", "PERC_THRESH_POWER", "PERC_POWER_LAW"),
}


@dataclass(frozen=True)
class SoilProcess:
    """Un processus attaché à une couche. `layer` est indexé à partir de 1, comme en TOML."""
    layer: int
    kind: str
    form: str
    params: dict = field(default_factory=dict)
    ceiling: float | None = None       # plafond du flux, mm/j ; None pour aucun

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"processus de type « {self.kind} » inconnu, attendu {KINDS}")
        if self.form not in FORMS:
            raise ValueError(f"forme « {self.form} » absente du catalogue : {sorted(FORMS)}")
        if self.form not in FORMS_BY_KIND[self.kind]:
            raise ValueError(f"la forme « {self.form} » ne convient pas a un processus "
                             f"« {self.kind} » ; formes valides {FORMS_BY_KIND[self.kind]}")
        for key in FORMS[self.form][1]:
            if key not in self.params:
                raise ValueError(f"forme « {self.form} » : parametre « {key} » manquant")
        if self.layer < 1:
            raise ValueError("les couches sont numerotees a partir de 1")

    def flux(self, ctx):
        """Flux en mètres par heure, dans les unités internes de la boucle de sous-pas."""
        q = FORMS[self.form][0](ctx, self.params)
        if self.ceiling is not None:
            q = torch.minimum(q, torch.as_tensor(self.ceiling, dtype=q.dtype, device=q.device))
        return q


@dataclass
class SoilProfile:
    """Le profil complet : un nombre de couches et les processus qui y sont attachés.

    Le nombre de couches est porté ici et non codé en dur, de sorte qu'en généraliser la
    valeur soit une extension et non une réécriture. La boucle du clone en traite trois
    aujourd'hui ; ce champ dit lesquelles sont déclarées, et la vérification refuse une
    couche hors du profil.
    """
    layers: int = 3
    processes: tuple[SoilProcess, ...] = ()

    def __post_init__(self):
        for proc in self.processes:
            if proc.layer > self.layers:
                raise ValueError(f"processus sur la couche {proc.layer} d'un profil qui en "
                                 f"compte {self.layers}")

    def of(self, layer: int, kind: str):
        """Les processus d'un type attachés à une couche, dans l'ordre de déclaration."""
        return tuple(p for p in self.processes if p.layer == layer and p.kind == kind)

    def total(self, layer: int, kind: str, ctx):
        """Somme des flux d'un type sur une couche, ou None si aucun n'est déclaré."""
        selected = self.of(layer, kind)
        if not selected:
            return None
        q = selected[0].flux(ctx)
        for proc in selected[1:]:
            q = q + proc.flux(ctx)
        return q


# Conversions entre les unites LISIBLES de la configuration et les unites INTERNES de la
# boucle de sous-pas, qui travaille en heures et en metres. Elles vivent ici pour que la
# section TOML se lise sans connaitre la boucle : un hydrologue ecrit des jours et des
# millimetres par jour.
HOURS_PER_DAY = 24.0
M_PER_MM = 1.0e-3
RESERVED_KEYS = {"layer", "kind", "form", "ceiling_mm_per_day", "tau_days"}


def from_toml(section: dict | None) -> SoilProfile | None:
    """Construit un profil depuis une section `[soil]` de TOML. None si la section est absente.

        [soil]
        layers = 3
        [[soil.process]]
        layer = 3
        kind = "percolation"
        form = "PERC_THRESH_POWER"
        tau_days = 2.0
        ceiling_mm_per_day = 3.0

    Les unites de la configuration sont le jour et le millimetre par jour ; la conversion
    vers les heures et les metres par heure de la boucle se fait ici.
    """
    if not section:
        return None
    raw = section.get("process") or section.get("processes") or []
    if isinstance(raw, dict):
        raw = [raw]
    declared = []
    for entry in raw:
        params = {k: v for k, v in entry.items() if k not in RESERVED_KEYS}
        if "tau_days" in entry:
            params["tau"] = float(entry["tau_days"]) * HOURS_PER_DAY
        ceiling = entry.get("ceiling_mm_per_day")
        if ceiling is not None:
            ceiling = float(ceiling) * M_PER_MM / HOURS_PER_DAY
        declared.append(SoilProcess(layer=int(entry["layer"]), kind=str(entry["kind"]),
                                    form=str(entry["form"]), params=params, ceiling=ceiling))
    return SoilProfile(layers=int(section.get("layers", 3)), processes=tuple(declared))


def hydrotel_profile(krec) -> SoilProfile:
    """La déclaration qui reproduit Hydrotel, pour servir de départ et de test de fidélité.

    Couche 2 : sortie latérale de Darcy le long du versant. Couche 3 : percolation linéaire en
    teneur en eau, et AUCUNE sortie latérale. C'est tout le profil du clone.
    """
    return SoilProfile(layers=3, processes=(
        SoilProcess(layer=2, kind=KIND_LATERAL, form="BASE_LINEAR"),
        SoilProcess(layer=3, kind=KIND_PERCOLATION, form="PERC_LINEAR", params={"krec": krec}),
    ))
