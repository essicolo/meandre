"""Log-ratios isometriques (ilr) pour les predicteurs COMPOSITIONNELS du champ spatial.

Remarque d'Essi (2026-08-31) : les fractions granulometriques sont compositionnelles, leur
somme est contrainte a 1, et les traiter comme des variables euclidiennes independantes
est incorrect depuis Aitchison (1986). Mesure sur OUTV : les trois colonnes normalisees
f_sand, f_silt, f_clay ont un rang de 2, le champ recoit un triplet exactement degenere.
Consequence secondaire : une sensibilite par fraction (E14 du registre) n'est pas bien
definie, on ne peut pas varier une part en tenant les autres fixes sur un simplexe.

PORTEE, et elle est deliberee. La transformation s'applique aux compositions utilisees
comme PREDICTEURS du champ. La ou une fraction entre dans la PHYSIQUE comme ponderation
reelle (parts de conifere/feuillu/decouvert qui ponderent la fonte, fraction de milieu
humide qui dimensionne un reservoir), elle reste une fraction : c'est sa signification.

ZEROS. Un log-ratio exige des parts strictement positives, or l'occupation porte beaucoup
de zeros exacts (urbain nul sur la plupart des troncons) et la granulometrie en porte
(63 troncons sans sable sur OUTV). Remplacement multiplicatif de Martin-Fernandez 2003 :
chaque zero recoit delta, les parts non nulles sont reduites au prorata, la somme est
preservee.

IMPLEMENTATION. Au branchement (1.1), remplacer les fonctions locales par le package
`nuee` d'Essi (PyPI, 0.2.1) : nuee.ilr(), nuee.multiplicative_replacement(),
nuee.sbp_basis() pour des bases de balances choisies plutot que codees en dur, et
nuee.closure(). Les fonctions locales ci-dessous n'existent que pour verrouiller les
proprietes par les tests en attendant ; elles portent la meme convention et devront
etre validees contre nuee avant d'etre supprimees. Ajouter nuee aux dependances a ce
moment-la, pas avant.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def remplacement_zeros(parts: Tensor, delta: float = 0.005) -> Tensor:
    """Remplace les zeros d'une composition (n, k) en preservant la somme.

    Martin-Fernandez et coll. 2003, remplacement multiplicatif : les zeros recoivent
    delta fois la somme, les parts non nulles sont reduites au prorata.
    """
    s = parts.sum(dim=1, keepdim=True).clamp(min=1e-12)
    p = parts / s
    zero = p <= 0
    n_zero = zero.sum(dim=1, keepdim=True).to(p.dtype)
    p = torch.where(zero, torch.full_like(p, delta), p * (1.0 - delta * n_zero))
    return p / p.sum(dim=1, keepdim=True).clamp(min=1e-12)


def ilr(parts: Tensor, delta: float = 0.005) -> Tensor:
    """Coordonnees ilr d'une composition (n, k) -> (n, k-1), base par bifurcation.

    La coordonnee j est la balance entre les parts 0..j et la part j+1 :
        z_j = sqrt(j+1 / j+2) * ln( moyenne geometrique(parts 0..j) / part j+1 )
    """
    p = remplacement_zeros(parts, delta)
    n, k = p.shape
    lg = torch.log(p)
    out = []
    for j in range(k - 1):
        gauche = lg[:, : j + 1].mean(dim=1)
        coef = math.sqrt((j + 1) / (j + 2))
        out.append(coef * (gauche - lg[:, j + 1]))
    return torch.stack(out, dim=1)


# Les groupes compositionnels parmi les attributs territoriaux. L'occupation est fermee
# par un complement (sol nu, roc, routes...) pour que la composition somme a 1 ; sans
# lui, le sous-vecteur n'est pas une composition et l'ilr n'a pas de sens.
GROUPES = {
    "granulo": ("f_sand", "f_silt", "f_clay"),
    "occupation": ("f_forest", "f_agriculture", "f_urban", "f_wetland", "f_water"),
}


def transformer_territorial(t, delta: float = 0.005):
    """Remplace les colonnes compositionnelles d'un TerritorialFeatures par leurs ilr.

    Attend les colonnes en FRACTIONS BRUTES ; si la table ne porte que des colonnes
    normalisees (moyenne nulle), les fractions sont reconstruites depuis `physical`
    quand elles y sont, sinon la transformation est refusee plutot que fausse.
    Retourne un nouveau TerritorialFeatures ; l'objet d'entree n'est pas modifie.
    """
    from meandre.spatial.territorial import TerritorialFeatures

    cols = list(t.columns)
    data = t.data
    garde = [i for i, c in enumerate(cols) if not any(c in g for g in GROUPES.values())]
    nouveaux, noms = [data[:, garde]], [cols[i] for i in garde]

    for nom, parts_noms in GROUPES.items():
        if not all(c in cols for c in parts_noms):
            continue
        brut = torch.stack([_fraction_brute(t, c) for c in parts_noms], dim=1)
        if nom == "occupation":
            compl = (1.0 - brut.sum(dim=1, keepdim=True)).clamp(min=0.0)
            brut = torch.cat([brut, compl], dim=1)
        z = ilr(brut, delta)
        z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True).clamp(min=1e-6)
        nouveaux.append(z)
        noms += [f"ilr_{nom}_{j + 1}" for j in range(z.shape[1])]

    return TerritorialFeatures(torch.cat(nouveaux, dim=1), noms, dict(t.physical))


def _fraction_brute(t, col: str) -> Tensor:
    """La fraction en [0, 1], depuis physical si la colonne du tenseur est normalisee."""
    if col in t.physical:
        return t.physical[col].to(t.data.dtype)
    v = t.data[:, list(t.columns).index(col)]
    if float(v.min()) >= -1e-6 and float(v.max()) <= 1.0 + 1e-6:
        return v.clamp(0.0, 1.0)
    raise ValueError(
        f"{col} est normalisee (min {float(v.min()):.2f}) et absente de `physical` : "
        "impossible de reconstruire la fraction brute, la table doit etre rebatie "
        "avec ses fractions physiques.")
