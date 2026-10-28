"""Combien de contraintes INDÉPENDANTES chaque source de neige apporte-t-elle.

Une donnée auxiliaire se juge sur l'identifiabilité, pas sur le KGE. Comparer 35 millions de
valeurs de NEISIM à 2154 relevés du réseau au sol ne dit rien : les 3917 séries de NEISIM
viennent d'une grille de deux kilomètres et demi et sont massivement redondantes. Ce qui
compte est le nombre de directions indépendantes que chaque source contraint.

La mesure est le RANG EFFECTIF de la matrice des anomalies, nœuds ou sites en colonnes, jours
en lignes : le nombre de composantes principales portant 95 % de la variance, et le rapport
de participation, qui pèse les composantes par leur variance et ne dépend d'aucun seuil.

CE QUE CELA BORNE. C'est le nombre de directions que la donnée POURRAIT contraindre, donc une
borne supérieure. Que le modèle s'en serve dépend du poids de la contrainte et de l'existence
d'un chemin de gradient, ce que ce banc ne mesure pas.

RÉSERVE DE FOND. NEISIM est un modèle : sa structure spatiale est la sienne. Un rang effectif
élevé y désigne la richesse de NEISIM, pas nécessairement celle de la réalité.

    .venv/bin/python .runs/quebec/information_de_la_neige.py outv gasp
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.basin_cache import BasinCache
from meandre.data.canswe_loader import build_swe_targets
from meandre.utils import paths as _paths

DERIVES = f"{os.environ.get('MEANDRE_DERIVES', _paths.DERIVED_ROOT)}/auxiliaires"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"
MOIS_NEIGE = (11, 12, 1, 2, 3, 4, 5)


def _mensualise(x, times, hiver):
    """Moyennes MENSUELLES par colonne, sur les mois de neige.

    Les relevés du réseau au sol sont espacés de une à deux semaines : à l'échelle
    journalière presque aucun jour n'est complet et la matrice est inexploitable. Le pas
    mensuel est le plus fin auquel les deux sources existent, et il est appliqué aux DEUX,
    faute de quoi la comparaison porterait sur deux objets différents.
    """
    cle = times.year.to_numpy() * 100 + times.month.to_numpy()
    cle = cle[hiver]
    x = x[hiver]
    mois = np.unique(cle)
    out = np.full((len(mois), x.shape[1]), np.nan)
    for i, m in enumerate(mois):
        bloc = x[cle == m]
        with np.errstate(invalid="ignore"):
            n = np.isfinite(bloc).sum(axis=0)
            out[i] = np.where(n > 0, np.nansum(np.nan_to_num(bloc), axis=0) / np.maximum(n, 1), np.nan)
    return out


def _rang_effectif(x, recouvrement_min=24):
    """Rang effectif d'une matrice mois x colonnes, par corrélation deux à deux.

    Retourne (composantes pour 95 %, rapport de participation, colonnes retenues). Le
    rapport de participation vaut (somme des valeurs propres)^2 sur somme de leurs carrés :
    il vaut 1 quand une seule direction porte tout, et le nombre de colonnes quand elles
    sont indépendantes et de même variance. La corrélation se calcule sur les mois communs
    à chaque paire, seule façon de traiter une matrice trouée sans inventer de valeurs.
    """
    x = np.asarray(x, dtype="float64")
    garde = np.isfinite(x).sum(axis=0) >= recouvrement_min
    x = x[:, garde]
    if x.shape[1] < 2:
        return np.nan, np.nan, int(garde.sum())
    v = np.isfinite(x)
    xc = np.where(v, x, 0.0)
    n = v.astype("float64").T @ v.astype("float64")
    s1 = xc.T @ v.astype("float64")
    s2 = (xc ** 2).T @ v.astype("float64")
    sxy = xc.T @ xc
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (s1 / n) * (s1.T / n)
        var = s2 / n - (s1 / n) ** 2
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        c = cov / np.outer(d, d)
    c[~np.isfinite(c)] = 0.0
    c = np.clip((c + c.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    # Une matrice construite par paires n'est pas forcement definie positive : on tronque
    # les valeurs propres negatives, qui sont l'artefact de cette construction.
    vp = np.clip(np.linalg.eigvalsh(c), 0.0, None)
    vp = np.sort(vp)[::-1]
    vp = vp[vp > 1e-10]
    if vp.size == 0:
        return np.nan, np.nan, int(garde.sum())
    part = np.cumsum(vp) / vp.sum()
    return int(np.searchsorted(part, 0.95) + 1), float(vp.sum() ** 2 / (vp ** 2).sum()), int(garde.sum())


def _correlation_par_paires(x, recouvrement_min):
    """Matrice de corrélation sur les mois communs à chaque paire de colonnes."""
    v = np.isfinite(x)
    garde = v.sum(axis=0) >= recouvrement_min
    x, v = x[:, garde], v[:, garde]
    if x.shape[1] < 2:
        return None
    xc = np.where(v, x, 0.0)
    vf = v.astype("float64")
    n = vf.T @ vf
    n[n < 3] = np.nan
    s1 = xc.T @ vf
    s2 = (xc ** 2).T @ vf
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = xc.T @ xc / n - (s1 / n) * (s1.T / n)
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        c = cov / np.outer(d, d)
    c[~np.isfinite(c)] = 0.0
    c = np.clip((c + c.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c, garde


def _reproductibilite(x, k=4, recouvrement_min=12):
    """La structure spatiale des k premières directions se retrouve-t-elle sur l'autre moitié.

    Un bruit indépendant gonfle le rang effectif sans porter d'information : il ne se
    reproduit pas. On coupe la période en deux, on prend les k premières directions de
    chaque moitié sur les COLONNES COMMUNES, et on mesure le recouvrement des deux
    sous-espaces, qui vaut 1 s'ils coïncident et k/p s'ils sont sans rapport.
    """
    m = x.shape[0] // 2
    a, b = x[:m], x[m:]
    commun = (np.isfinite(a).sum(axis=0) >= recouvrement_min) & (np.isfinite(b).sum(axis=0) >= recouvrement_min)
    if commun.sum() < k + 2:
        return np.nan, int(commun.sum())
    ra = _correlation_par_paires(a[:, commun], recouvrement_min)
    rb = _correlation_par_paires(b[:, commun], recouvrement_min)
    if ra is None or rb is None:
        return np.nan, int(commun.sum())
    ca, ga = ra
    cb, gb = rb
    ok = ga & gb
    ca, cb = ca[np.ix_(ok[ga], ok[ga])], cb[np.ix_(ok[gb], ok[gb])]
    if ca.shape[0] < k + 2:
        return np.nan, int(ca.shape[0])
    va = np.linalg.eigh(ca)[1][:, -k:]
    vb = np.linalg.eigh(cb)[1][:, -k:]
    # Recouvrement des sous-espaces : moyenne des carres des correlations canoniques.
    sv = np.linalg.svd(va.T @ vb, compute_uv=False)
    p = ca.shape[0]
    return float((sv ** 2).mean()), p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    a = p.parse_args()
    times = pd.date_range(DATE_START, DATE_END, freq="D")
    hiver = np.isin(times.month, MOIS_NEIGE)

    print(f"{'territoire':<12s} {'source':<18s} {'series':>8s} {'95 %':>7s} "
          f"{'participation':>14s} {'reproductible':>14s} {'hasard':>8s}")
    for reg in [r.lower() for r in a.regions]:
        lignes = []
        noeuds_reseau = None
        cache = BasinCache(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb")
        if cache.has_canswe():
            mes, sit = cache.load_canswe(DATE_START, DATE_END)
            obs, node_idx, gardes = build_swe_targets(mes, sit, times)
            if obs is not None:
                lignes.append(("reseau", obs.cpu().numpy()))
                noeuds_reseau = node_idx.cpu().numpy()
        f = f"{DERIVES}/neisim-{reg}.npz"
        if os.path.exists(f):
            z = np.load(f)
            lignes.append(("neisim", z["valeurs"]))
            # CONTROLE DE POPULATION : NEISIM lu aux SEULS nœuds que le reseau occupe. Sans
            # lui, un ecart de rang pourrait venir de ce qu'on compare tout un territoire a
            # une poignee de sites, et non des donnees elles-memes.
            if noeuds_reseau is not None:
                rang = {int(n): i for i, n in enumerate(z["node_idx"])}
                col = [rang[int(n)] for n in noeuds_reseau if int(n) in rang]
                if col:
                    lignes.append(("neisim-aux-sites", z["valeurs"][:, col]))
        for nom, m in lignes:
            mm = _mensualise(m, times, hiver)
            n95, part, cols = _rang_effectif(mm)
            rep, p = _reproductibilite(mm)
            n95_t = f"{n95:>7d}" if np.isfinite(n95) else f"{'—':>7s}"
            part_t = f"{part:>14.1f}" if np.isfinite(part) else f"{'—':>14s}"
            rep_t = f"{rep:>14.2f}" if np.isfinite(rep) else f"{'—':>14s}"
            # Ce qu'un sous-espace de dimension k obtiendrait au hasard dans p dimensions.
            hasard = f"{4.0 / p:>8.2f}" if p and np.isfinite(rep) else f"{'—':>8s}"
            print(f"{reg:<12s} {nom:<18s} {cols:>8d} {n95_t} {part_t} {rep_t} {hasard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
