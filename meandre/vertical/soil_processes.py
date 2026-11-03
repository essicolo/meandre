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
KIND_RUNOFF = "runoff"
KINDS = (KIND_LATERAL, KIND_PERCOLATION, KIND_RUNOFF)


def _storage_above(ctx, prm):
    """Stock au-dessus d'un niveau de la couche, en mètres, coupé net ou lissé (2026-10-02).

    `threshold` choisit le niveau : « field_capacity » (défaut, eau gravitaire) ou
    « wilting_point » (eau disponible, que le sol rend lentement par capillarité). `smooth`
    (mètres, donné en `smooth_mm`) remplace la coupure par une transition douce,
    smooth · softplus(stock / smooth) : le flux reste continu et dérivable au passage du
    niveau, sans porte ni bascule de conductivité. Sans `smooth`, la coupure d'origine.
    """
    level = ctx["theta_wp"] if prm.get("threshold", "field_capacity") == "wilting_point" else ctx["theta_fc"]
    s = (ctx["theta"] - level) * ctx["thickness"]
    w = prm.get("smooth")
    if w is None:
        return torch.clamp(s, min=0.0)
    return w * torch.nn.functional.softplus(s / w)


def _conductivity(ctx, prm):
    """Conductivité à la teneur en eau courante, K(theta). Si le processus déclare `conductivity`
    (un nombre en m/h, ou une valeur par tronçon posée par le pilote, par exemple celle du
    substratum tirée de la classe de drainage), la courbe K(theta) de la couche est rééchelonnée
    pour valoir cette conductivité à saturation : la dynamique est celle du dépôt, la forme reste
    celle du sol (2026-10-27)."""
    k = ctx["conductivity"]
    c = prm.get("conductivity")
    if c is None:
        return k
    return k / torch.clamp(ctx["k_sat"], min=1e-12) * torch.as_tensor(c, dtype=k.dtype, device=k.device)


def base_linear(ctx, prm):
    """Darcy le long du versant, forme d'Hydrotel pour la couche 2 : K(theta) · sin(pente) · z.

    `gravitational = true` (2026-11-03) : seule l'eau gravitaire s'écoule le long du versant. Le
    flux est multiplié par la part de la porosité drainable occupée, (theta − capacité au champ) /
    (porosité − capacité au champ), bornée entre 0 et 1 : sous la capacité au champ l'eau est tenue
    par capillarité et ne part plus latéralement, à saturation le flux est celui d'Hydrotel. Aucun
    paramètre nouveau : les deux niveaux sont ceux du sol.
    """
    q = _conductivity(ctx, prm) * ctx["sin_slope"] * ctx["thickness"] * prm.get("factor", 1.0)
    if prm.get("gravitational"):
        fc, ths = ctx["theta_fc"], ctx["porosity"]
        if fc is None:
            raise ValueError("BASE_LINEAR gravitational : capacité au champ de la couche absente")
        q = q * torch.clamp((ctx["theta"] - fc) / torch.clamp(ths - fc, min=1e-6), 0.0, 1.0)
    return q


def base_thresh_power(ctx, prm):
    """Vidange de l'eau gravitaire avec sa propre constante de temps, exposant libre.

    Q = (S_grav / tau) · (S_grav / S_max)^(n-1), soit linéaire pour n = 1. Un exposant
    supérieur à un relâche vite à stock plein et lentement à stock bas, ce que le diagnostic
    du 2026-09-18 réclamait pour l'aquifère et qui vaut aussi ici.

    CONVERGENCE. La condition de Courant se resserre quand la couche reste pleine. Mesuré le
    2026-09-19 sur une couche de 2,70 m : à tau = 5 jours la loi est convergée dès 64 sous-pas,
    266,7 mm contre 267,9 à 512 ; à tau = 20 jours elle ne l'est pas, 168,2 contre 204,4.
    """
    s = _storage_above(ctx, prm)
    q = s / prm["tau"]
    n = prm.get("exponent", 1.0)
    if n != 1.0:
        # ECHELLE (2026-09-29). Rapportee a la capacite gravitaire de toute la couche, une
        # couche profonde de 2,65 m donne 185 mm d'echelle : quelques millimetres d'exces font
        # une fraction infime, et la sortie s'eteint (mesure : l'hypodermique tombe de 1,6 a
        # 0,3 mm/j, KGE de 0,64 a 0,38). `scale_mm` donne l'echelle de l'exces, en millimetres,
        # au-dela de laquelle la loi amplifie ; sans elle, l'ancien comportement.
        s_ref = prm.get("scale")
        if s_ref is None:
            s_ref = torch.clamp(ctx["porosity"] - ctx["theta_fc"], min=1e-9) * ctx["thickness"]
            ratio = torch.clamp(s / s_ref, min=0.0, max=1.0)
        else:
            # `max_ratio` borne l'amplification : sans borne, une pluie de 60 mm sur une
            # echelle de 5 mm multiplie la sortie par douze et double la pointe annuelle
            # (mesure sur la Chateauguay, 2026-09-29).
            ratio = torch.clamp(s / s_ref, min=0.0, max=float(prm.get("max_ratio", float("inf"))))
        q = q * ratio ** (n - 1.0)
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


def perc_gravity(ctx, prm):
    """Écoulement gravitaire sous gradient unitaire : le flux vaut la conductivité de la couche à
    sa teneur en eau, K(theta) (2026-10-27). La vidange suit ainsi le sol, sans constante de
    temps : rapide quand la couche est humide, lente sous la capacité au champ, où K(theta)
    s'effondre. Un plafond déclaré représente le substratum."""
    return _conductivity(ctx, prm) * prm.get("factor", 1.0)


# Catalogue. La clé est le nom Raven, la valeur la fonction et les paramètres exigés.
def inf_pdm(ctx, prm):
    """Ruissellement par distribution de capacités de stockage (Moore 1985, INF_PDM chez Raven),
    forme intégrée équivalente à Xinanjiang (2026-10-30, R447).

    Le tronçon est une mosaïque de réservoirs de capacité c, répartie en F(c) = 1 − (1 − c/cmax)^b
    sur [0, cmax] ; la capacité moyenne vaut S_max = cmax / (b + 1). Pour un stock S et un apport
    P, les réservoirs déjà pleins et ceux que P remplit débordent ; le reste s'infiltre. Deux
    propriétés mesurées sur 175 stations motivent la forme : le coefficient de ruissellement croît
    avec la taille de l'averse (pente log-log 1,2 observée) et avec l'humidité antérieure (0,6).

    Contexte : `storage` et `capacity` en mètres (eau au-dessus du point de flétrissement des
    couches incluses, et son maximum à saturation), `input` en mètres (apport du jour). Paramètres :
    `exponent` b, et `factor` (1 par défaut) qui multiplie la capacité tirée du sol : c'est le
    paramètre propre de la loi, distinct de la porosité et de la conductivité, apprenable.
    Retourne le ruissellement en mètres pour le pas.
    """
    b = torch.as_tensor(prm["exponent"], dtype=ctx["input"].dtype, device=ctx["input"].device)
    s_max = torch.clamp(ctx["capacity"] * torch.as_tensor(prm.get("factor", 1.0), dtype=ctx["input"].dtype, device=ctx["input"].device), min=1e-6)
    s = torch.clamp(ctx["storage"], torch.zeros_like(s_max), s_max)
    c_max = s_max * (b + 1.0)
    # Capacité critique : les réservoirs de capacité inférieure sont pleins.
    c_star = c_max * (1.0 - torch.clamp(1.0 - s / s_max, min=1e-9) ** (1.0 / (b + 1.0)))
    x = torch.clamp((c_star + ctx["input"]) / c_max, max=1.0)
    s_after = s_max * (1.0 - (1.0 - x) ** (b + 1.0))
    return torch.clamp(ctx["input"] - (s_after - s), min=0.0)


def topmodel_runoff(ctx, prm):
    """Ruissellement par aire saturée commandée par la nappe libre (forme TOPMODEL, R447).

    Une portion du tronçon voit la nappe affleurer et ruisselle tout l'apport du jour ; le reste
    s'infiltre. La portion croît avec le stock de nappe : a = (s_gw / scale)^exponent, bornée à
    un. Le levier est mesuré sans simulation : sur les épisodes d'été de R447, le stock de nappe du
    modèle gelé prédit le coefficient de ruissellement observé à pente log-log 0,63 en Montérégie
    et 0,33 sur l'Outaouais, au niveau du débit de la veille, et il est redondant avec lui. Le
    contexte porte `s_gw`, le stock de nappe de la veille en mètres, posé par la colonne quand la
    nappe libre est activée ; sans nappe, la forme n'a pas de levier et refuse. Paramètres propres :
    `exponent` et `scale` (le TOML le donne en `scale_mm`, un stock de nappe en millimètres, et
    from_toml le convertit en mètres comme pour la loi en exposant). Retourne le ruissellement
    en mètres pour le pas.
    """
    s = ctx.get("s_gw")
    if s is None:
        raise ValueError("la forme TOPMODEL demande la nappe libre : le contexte doit porter s_gw")
    a = torch.clamp(s / torch.as_tensor(prm["scale"], dtype=s.dtype, device=s.device), 0.0, 1.0)
    a = a.pow(torch.as_tensor(float(prm["exponent"]), dtype=s.dtype, device=s.device))
    return ctx["input"] * a


def drain_hooghoudt(ctx, prm):
    """Drainage souterrain agricole, formule de Hooghoudt (2026-10-04).

    q = f · K · z · min(1, (8·d_e·m + 4·m²) / L²), f la fraction drainee (part `share` des terres
    agricoles du troncon), K la conductivite de la couche, z son epaisseur, m la charge de la
    zone saturee au-dessus des drains, d_e la profondeur equivalente sous les drains (par defaut
    l'epaisseur de la couche) et L l'espacement. La hauteur saturee se lit comme z·theta/theta_s
    depuis la base de la couche, et le drain est pose a `depth_m` sous la surface. Le facteur de
    Hooghoudt, sans dimension, est borne a 1 : un drain ne transmet pas plus que la capacite
    laterale saturee de la couche. Meme forme que la branche historique de bv3c2, qu'un profil
    declare n'execute pas.
    """
    z = ctx["thickness"]
    agri = ctx.get("agri_frac")
    if agri is None:
        return torch.zeros_like(ctx["theta"])
    frac = torch.clamp(prm["share"] * agri, 0.0, 1.0)
    s = z * ctx["theta"] / (ctx["porosity"] + 1e-9)
    h_drain = torch.clamp(ctx["z_top"] + z - prm["depth_m"], min=0.0)
    m = torch.clamp(s - h_drain, min=0.0)
    de = prm.get("equiv_depth_m", z)
    L = prm["spacing_m"]
    hoog = (8.0 * de * m + 4.0 * m * m) / (L * L)
    return frac * ctx["conductivity"] * z * torch.clamp(hoog, max=1.0)


FORMS = {
    "BASE_LINEAR": (base_linear, ()),
    "BASE_THRESH_POWER": (base_thresh_power, ("tau",)),
    "PERC_LINEAR": (perc_linear, ("krec",)),
    "PERC_THRESH_POWER": (perc_thresh_power, ("tau",)),
    "PERC_POWER_LAW": (perc_power_law, ("krec", "exponent")),
    "PERC_GRAVITY": (perc_gravity, ()),
    "DRAIN_HOOGHOUDT": (drain_hooghoudt, ("spacing_m", "depth_m", "share")),
    "INF_PDM": (inf_pdm, ("exponent",)),
    "TOPMODEL": (topmodel_runoff, ("exponent", "scale")),
}

FORMS_BY_KIND = {
    KIND_LATERAL: ("BASE_LINEAR", "BASE_THRESH_POWER", "DRAIN_HOOGHOUDT"),
    KIND_PERCOLATION: ("PERC_LINEAR", "PERC_THRESH_POWER", "PERC_POWER_LAW", "PERC_GRAVITY"),
    # Un processus « runoff » porte sur l'apport du jour, avant le sol ; `layer` est la couche la
    # plus profonde dont le stock entre dans la distribution de capacités (1 : couche 1 seule).
    # TOPMODEL ignore ce stock : sa manette est le stock de nappe du contexte.
    KIND_RUNOFF: ("INF_PDM", "TOPMODEL"),
}


@dataclass(frozen=True)
class SoilProcess:
    """Un processus attaché à une couche. `layer` est indexé à partir de 1, comme en TOML."""
    layer: int
    kind: str
    form: str
    params: dict = field(default_factory=dict)
    # Plafond du flux, en unites internes, ou le NOM d'une sortie du champ spatial a
    # resoudre par troncon. None pour aucun plafond.
    ceiling: float | str | None = None
    # Parametres APPRIS (2026-10-01) : noms parmi ceux de `params` et « ceiling ». La valeur
    # declaree sert de depart ; la colonne cree un scalaire en espace logarithmique.
    learn: tuple = ()

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
        for key in self.learn:
            if key != "ceiling" and key not in self.params:
                raise ValueError(f"parametre appris « {key} » absent du processus : {sorted(self.params)} ou ceiling")
            if key == "ceiling" and self.ceiling is None:
                raise ValueError("plafond appris sans plafond declare")

    def describe(self) -> str:
        """Une ligne lisible dans un journal, quel que soit le type du plafond.

        Le formatage vit ICI et non dans le pilote : une premiere version imprimait le
        plafond avec un format numerique, ce qui plante des qu'il nomme une sortie du champ
        au lieu de porter un nombre. Le defaut a coute deux passes le 2026-09-21, dans un
        script qu'aucun test ne couvre. Sous test, il ne peut plus se reproduire.
        """
        end = f"couche {self.layer} {self.kind} {self.form}"
        if self.params:
            end += " " + " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                   for k, v in sorted(self.params.items()))
        if self.ceiling is None:
            return end
        if isinstance(self.ceiling, str):
            return f"{end} | plafond pris du champ spatial : {self.ceiling}"
        return f"{end} | plafond {float(self.ceiling):.3e} m/h"

    def resolved(self, fields: dict):
        """Remplace les valeurs SYMBOLIQUES par les tenseurs du champ spatial.

        Une valeur de configuration peut nommer une sortie du champ au lieu de donner un
        nombre, par exemple `ceiling_mm_per_day = "k_sub"`. C'est ce qui rend un plafond
        spatial : le champ le prédit par tronçon depuis la texture du sol, qui l'explique à
        27 % contre le témoin, là où l'exposant et les constantes de temps ne sont prédits
        par aucune covariable disponible et restent donc uniformes.
        """
        from dataclasses import replace

        ceiling_value = self.ceiling
        if isinstance(ceiling_value, str):
            if ceiling_value not in fields:
                raise KeyError(f"plafond « {ceiling_value} » absent du champ spatial : "
                               f"sorties connues {sorted(fields)}")
            ceiling_value = fields[ceiling_value]
        params = {k: (fields[v] if isinstance(v, str) and v in fields else v)
                  for k, v in self.params.items()}
        if ceiling_value is self.ceiling and params == self.params:
            return self
        return replace(self, ceiling=ceiling_value, params=params)

    def flux(self, ctx):
        """Flux en mètres par heure, dans les unités internes de la boucle de sous-pas."""
        q = FORMS[self.form][0](ctx, self.params)
        # PORTE DE GEL (2026-10-06, R316) : `frost_factor` multiplie le flux par ce facteur
        # quand le sol est gelé sur toute la couche de surface, et par une valeur
        # intermédiaire quand le gel est partiel (poids = profondeur de gel de Rankinen
        # rapportée à la couche 1, indépendant du manteau). Opt-in : sans la clé, rien ne
        # change. Motif : en décembre-janvier, la voie latérale lente de la couche 3 porte
        # plus de la moitié du débit des grands bassins du nord, trop fort de 30 à 40 %.
        # VISCOSITE (R384) : les vidanges a constante de temps sont des ecoulements gravitaires,
        # proportionnels a la conductivite donc a l'inverse de la viscosite. Les formes qui lisent
        # deja la conductivite (BASE_LINEAR, DRAIN_HOOGHOUDT) la recoivent corrigee.
        if ctx.get("visc") is not None and self.form not in ("BASE_LINEAR", "DRAIN_HOOGHOUDT", "PERC_GRAVITY", "INF_PDM"):
            q = q * ctx["visc"]
        f = self.params.get("frost_factor")
        if f is not None and ctx.get("frost_frac") is not None:
            q = q * (1.0 - (1.0 - f) * ctx["frost_frac"])
        # FRACTION D'OCCUPATION (2026-10-08, R346) : `cover = "agri"` restreint le flux à la part
        # du tronçon portant cette occupation, multipliée par `share` (1 par défaut). Un chemin
        # d'événement propre aux terres cultivées drainées se déclare ainsi sans borner le flux
        # par la conductivité apprise, ce que fait DRAIN_HOOGHOUDT. Le contexte expose
        # aujourd'hui `agri_frac` ; une autre occupation demande sa fraction dans le contexte.
        cover = self.params.get("cover")
        if cover is not None:
            frac = ctx.get(f"{cover}_frac")
            if frac is None:
                raise KeyError(f"occupation « {cover} » absente du contexte du sol (fractions connues : "
                               f"{sorted(k[:-5] for k in ctx if k.endswith('_frac') and k != 'frost_frac')})")
            q = q * torch.clamp(frac * self.params.get("share", 1.0), 0.0, 1.0)   # `share` peut etre appris (tenseur)
        elif "share" in self.params and self.form != "DRAIN_HOOGHOUDT":
            # Sans occupation, `share` est la part du flux prise sur toutes les occupations
            # (2026-10-15). DRAIN_HOOGHOUDT lit deja `share` dans sa forme.
            q = q * torch.clamp(torch.as_tensor(self.params["share"], dtype=q.dtype, device=q.device), 0.0, 1.0)
        if self.ceiling is not None:
            if isinstance(self.ceiling, str):
                raise RuntimeError(f"plafond « {self.ceiling} » non resolu : appeler "
                                   f"SoilProfile.resolved() avec les sorties du champ")
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

    def resolved(self, fields: dict):
        """Profil dont les valeurs symboliques sont remplacées par les tenseurs du champ."""
        taken = tuple(p.resolved(fields) for p in self.processes)
        if all(a is b for a, b in zip(taken, self.processes)):
            return self
        return SoilProfile(layers=self.layers, processes=taken)

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
RESERVED_KEYS = {"layer", "kind", "form", "ceiling_mm_per_day", "tau_days", "learn"}


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
    # PIEGE EVITE DE JUSTESSE (2026-09-20). Une section `[soil]` existe DEJA dans les
    # configurations regionales, pour `hydrotel_calib_dir`. Rendre un profil des qu'elle est
    # presente aurait donne un profil VIDE, donc un sol sans aucun flux, sur toutes les
    # configurations existantes, et silencieusement. Le profil n'existe que si des processus
    # sont effectivement declares.
    if not section:
        return None
    raw = section.get("process") or section.get("processes") or []
    if not raw:
        return None
    if isinstance(raw, dict):
        raw = [raw]
    declared = []
    for entry in raw:
        params = {k: v for k, v in entry.items() if k not in RESERVED_KEYS}
        if "tau_days" in entry:
            params["tau"] = float(entry["tau_days"]) * HOURS_PER_DAY
        if "scale_mm" in entry:
            params["scale"] = float(entry["scale_mm"]) * M_PER_MM
            params.pop("scale_mm", None)
        if "smooth_mm" in entry:
            params["smooth"] = float(entry["smooth_mm"]) * M_PER_MM
            params.pop("smooth_mm", None)
        ceiling = entry.get("ceiling_mm_per_day")
        if isinstance(ceiling, str):
            # Nom d'une sortie du champ spatial, resolu plus tard contre les parametres du
            # troncon. Le champ travaille deja en metres par heure : aucune conversion.
            pass
        elif ceiling is not None:
            ceiling = float(ceiling) * M_PER_MM / HOURS_PER_DAY
        learn = tuple("tau" if k == "tau_days" else "ceiling" if k == "ceiling_mm_per_day" else k
                      for k in entry.get("learn", ()))
        declared.append(SoilProcess(layer=int(entry["layer"]), kind=str(entry["kind"]),
                                    form=str(entry["form"]), params=params, ceiling=ceiling,
                                    learn=learn))
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
