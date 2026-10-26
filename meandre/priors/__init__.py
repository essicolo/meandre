"""Catalogue des priors : la valeur centrale et l'écart de chaque paramètre physique, par tronçon,
tirés des attributs du domaine (étape 3 de l'architecture cible, 2026-10-26).

Le champ spatial n'apprend plus une valeur libre entre deux bornes mais un écart au prior, en
logarithme, pénalisé en proportion de l'écart divisé par l'écart-type du prior. Ce module ne
fait que calculer les centres : la texture par couche donne porosité, capacité au champ, point
de flétrissement et conductivité par la pédotransfert de Saxton et Rawls ; l'épaisseur des dépôts
quaternaires donne celle de la couche profonde. Où la donnée manque, le centre est celui d'un
loam de la littérature, avec le même écart : le prior dit alors seulement « un sol ordinaire ».

Les écarts sont des écarts-types en logarithme : 0,10 sur la porosité veut dire qu'un écart de
10 % coûte une unité de pénalité, un écart de 20 % quatre. Ils sont déclarés ici et non dans le
code du champ, pour qu'un adaptateur ou une recette puisse les remplacer.
"""
from __future__ import annotations

import math

import numpy as np

from meandre.data.pedotransfer import saxton_rawls

# Écart-type en logarithme de chaque famille : réglages déclarés ([priors] sigma_*), parce que ce
# sont des choix de confiance dans une carte, pas des constantes. Seul celui de la conductivité est
# mesuré (perméabilité des levés IRDA, 2026-10-26) ; les autres sont à mesurer.
def sigmas() -> dict[str, float]:
    from meandre.utils import settings
    return {"porosity": settings.get("priors.sigma_porosity"), "theta_fc_frac": settings.get("priors.sigma_theta_fc_frac"), "theta_wp_frac": settings.get("priors.sigma_theta_wp_frac"), "K_sat": settings.get("priors.sigma_k_sat"), "Z2": settings.get("priors.sigma_z2"), "Z3": settings.get("priors.sigma_z3")}
# Texture de repli : loam (sable 0,40, argile 0,20), le sol forestier tempéré de la littérature.
LOAM = (0.40, 0.20)
# Épaisseurs de repli, en mètres : couche 2 fine (Hydrotel pose 0,16 m au sud), couche 3 d'un mètre.
Z1_M, Z2_DEFAULT_M, Z3_DEFAULT_M = 0.10, 0.30, 1.0
LAYERS = (1, 2, 3)
KEYS = tuple(f"{f}_{l}" for f in ("porosity", "theta_fc_frac", "theta_wp_frac", "K_sat") for l in LAYERS) + ("Z2", "Z3")


def _fill(values: np.ndarray | None, n: int, default: float) -> np.ndarray:
    out = np.full(n, float(default))
    if values is not None:
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        out[ok] = v[ok]
    return out


def soil_priors(n_nodes: int, texture: dict[int, tuple] | None = None, deposit_thickness_m: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, float]]:
    """Centres et écarts par tronçon.

    `texture` : par couche, (fraction de sable, fraction d'argile), NaN où la texture manque.
    `deposit_thickness_m` : épaisseur des dépôts meubles, NaN où elle manque.
    Retourne {clé: (centre (n_nodes,), sigma)} pour les clés de `KEYS`.
    """
    SIGMAS = sigmas()
    priors: dict[str, tuple[np.ndarray, float]] = {}
    for layer in LAYERS:
        sand, clay = (texture or {}).get(layer, (None, None))
        s = _fill(sand, n_nodes, LOAM[0])
        c = _fill(clay, n_nodes, LOAM[1])
        sr = saxton_rawls(s, c)
        priors[f"porosity_{layer}"] = (np.asarray(sr["theta_s"], float), SIGMAS["porosity"])
        priors[f"theta_fc_frac_{layer}"] = (np.asarray(sr["theta_fc"] / sr["theta_s"], float), SIGMAS["theta_fc_frac"])
        priors[f"theta_wp_frac_{layer}"] = (np.asarray(sr["theta_wp"] / sr["theta_fc"], float), SIGMAS["theta_wp_frac"])
        priors[f"K_sat_{layer}"] = (np.asarray(sr["k_sat"], float), SIGMAS["K_sat"])
    z2 = np.full(n_nodes, Z2_DEFAULT_M)
    z3 = np.full(n_nodes, Z3_DEFAULT_M)
    if deposit_thickness_m is not None:
        t = np.asarray(deposit_thickness_m, dtype=float)
        ok = np.isfinite(t)
        z3[ok] = np.clip(t[ok] - Z1_M - Z2_DEFAULT_M, 0.2, 4.0)
    priors["Z2"] = (z2, SIGMAS["Z2"])
    priors["Z3"] = (z3, SIGMAS["Z3"])
    return priors


def summary(priors: dict[str, tuple[np.ndarray, float]]) -> str:
    parts = []
    for k in ("porosity_1", "theta_fc_frac_1", "K_sat_1", "Z2", "Z3"):
        c, s = priors[k]
        parts.append(f"{k} médiane {np.median(c):.3g} (sigma log {s:.2f})")
    return "priors : " + " ; ".join(parts)
