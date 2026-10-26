"""Candidate 5.1, épreuves 3.3, 3.6 et 3.7 : une fonction de fonte non linéaire apprise contre
l'ablation MESURÉE bat-elle la loi des degrés-jours réajustée, aux sites et aux territoires
jamais vus ?

Cible : baisse de masse du manteau CanSWE entre deux relevés consécutifs à un jour, un jour
sans précipitation (CaSR brut, le forçage du modèle), bornée à zéro. Territoires où le réseau
existe : Outaouais, Saguenay, Saint-Laurent nord-ouest, Gaspésie.

Lois comparées, toutes avec la même cible :
  DJ      loi d'Hydrotel réajustée : cf · max(T − T_melt, 0) · (1 + 0,5 sin saisonnier), cf et
          T_melt ajustés par moindres carrés (témoin paramétrique) ;
  ETI     degrés-jours plus rayonnement net, linéaire ;
  G       arbres de gradient MONOTONES croissants en T et en rayonnement (sklearn
          monotonic_cst), sur T, rayonnement, vent, vapeur, stock, âge du manteau, saison ;
  G+cum   G avec cumuls de T et de rayonnement sur 3 et 7 jours (épreuve 3.6 : la mémoire
          ajoute-t-elle quelque chose aux variables du jour ?) ;
  G+att   G+cum conditionné par les attributs du site (couvert, latitude, altitude) ;
  G+xy    G+cum conditionné par les coordonnées seules (ce qu'un réseau de coordonnées
          mémoriserait).

Validations : sites retirés (cinq plis par site) ; années retirées (une sur cinq) ; territoire
retiré (apprentissage sur trois, évaluation sur le quatrième). Erreur : écart absolu moyen de
l'ablation (mm/j) sur les jours de fonte (ablation observée > 0,5 mm), et biais.

    .venv/Scripts/python.exe experiments/plan_ml/fonte_observee.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REGIONS = ["outv", "sagu", "slno"]   # la Gaspésie n'a que 19 jours-sites secs consécutifs


def charger(reg):
    bc = BasinCache(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb")
    h = bc.load(device=torch.device("cpu"))
    mes, sites = bc.load_canswe("2000-01-01", "2024-12-31")
    fx = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-casr-brut.nc")
    F = fx["forcing"].values
    temps = pd.DatetimeIndex(fx.time.values)
    fx.close()
    m = mes[mes.swe_mm.notna()].copy()
    m["date"] = pd.to_datetime(m.date)
    m = m.sort_values(["swe_station_id", "date"])
    m["dswe"] = m.groupby("swe_station_id").swe_mm.diff()
    m["dj"] = m.groupby("swe_station_id").date.diff().dt.days
    m = m[(m.dj == 1) & m.dswe.notna()]
    pos = pd.Series(np.arange(len(temps)), index=temps.normalize())
    m["t"] = pos.reindex(pd.DatetimeIndex(m.date).normalize()).to_numpy()
    m = m[np.isfinite(m.t)]
    m["t"] = m.t.astype(int)
    m["node"] = m.node_idx.astype(int)
    t, n = m.t.to_numpy(), m.node.to_numpy()
    P = F[t, n, 0]
    m = m[P < 0.5].copy()
    t, n = m.t.to_numpy(), m.node.to_numpy()
    tm = (F[t, n, 1] + F[t, n, 2]) / 2
    # Cumuls et âge du manteau par nœud, calculés sur la série pleine du nœud.
    Tn = (F[:, :, 1] + F[:, :, 2]) / 2
    Rn = F[:, :, 3]
    Pn = F[:, :, 0]
    neige = (Pn > 2.0) & (Tn < 1.0)

    def cumul(A, w):
        c = np.cumsum(A, axis=0)
        out = c.copy()
        out[w:] = c[w:] - c[:-w]
        return out

    T3, T7 = cumul(np.clip(Tn, 0, None), 3), cumul(np.clip(Tn, 0, None), 7)
    R3, R7 = cumul(Rn, 3), cumul(Rn, 7)
    age = np.zeros_like(Tn)
    for k in range(1, len(temps)):
        age[k] = np.where(neige[k], 0.0, age[k - 1] + 1.0)
    terr = h["territorial"]
    col = {c: j for j, c in enumerate(terr.columns)}
    coords = h["node_coords"].numpy()
    doy = pd.DatetimeIndex(m.date).dayofyear.to_numpy()
    X = pd.DataFrame({
        "tmoy": tm, "tmax": F[t, n, 2], "rn": F[t, n, 3], "u2": F[t, n, 4], "ea": F[t, n, 5],
        "swe": m.swe_mm.to_numpy(), "age": age[t, n],
        "sais_s": np.sin(2 * np.pi * doy / 365.25), "sais_c": np.cos(2 * np.pi * doy / 365.25),
        "T3": T3[t, n], "T7": T7[t, n], "R3": R3[t, n], "R7": R7[t, n],
        "f_forest": terr.data[n, col["f_forest"]].numpy(), "f_agri": terr.data[n, col["f_agriculture"]].numpy(),
        "elev": terr.data[n, col["mean_elevation_m"]].numpy(), "lat": coords[n, 1], "lon": coords[n, 0],
        "doy": doy,
    })
    X["y"] = np.clip(-m.dswe.to_numpy(), 0.0, None)
    X["site"] = m.swe_station_id.astype(str).to_numpy()
    X["annee"] = pd.DatetimeIndex(m.date).year.to_numpy()
    X["region"] = reg
    return X


def predire_dj(p, X):
    """Loi d'Hydrotel : cf · max(T − T_melt, 0) · (1 + 0,5 sin(2π(j − 81)/365)), plafonnée au stock."""
    sais = 1 + 0.5 * np.sin(2 * np.pi * (X.doy.to_numpy() - 81) / 365.25)
    return np.minimum(p[0] * np.clip(X.tmoy.to_numpy() - p[1], 0, None) * sais, X.swe.to_numpy())


def loi_dj(X, y, fonte_seule=False):
    """cf et T_melt ajustés par moindres carrés ; `fonte_seule` n'ajuste que sur les jours où il
    fond (borne haute de ce que la loi peut faire, puisqu'elle ne décide pas s'il fond)."""
    if fonte_seule:
        g = y > 0.5
        X, y = X[g], y[g]

    def res(p):
        return predire_dj(p, X) - y

    meilleur = None
    for t0 in (-2.0, 0.0, 2.0):
        sol = least_squares(res, x0=[3.0, t0], bounds=([0.0, -5.0], [30.0, 8.0]))
        if meilleur is None or sol.cost < meilleur.cost:
            meilleur = sol
    return meilleur.x


BASE = ["tmoy", "tmax", "rn", "u2", "ea", "swe", "age", "sais_s", "sais_c"]
CUM = ["T3", "T7", "R3", "R7"]
ATT = ["f_forest", "f_agri", "elev", "lat"]
XY = ["lat", "lon"]


def arbres(cols):
    mono = [1 if c in ("tmoy", "tmax", "rn", "T3", "T7", "R3", "R7") else 0 for c in cols]
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, monotonic_cst=mono, random_state=0)


MODELES = {
    "DJ réajustée (témoin)": None,
    "DJ ajustée jours de fonte": "fonte",
    "ETI linéaire": (["tmoy", "rn"], lambda c: Ridge(1.0)),
    "G monotone, jour": (BASE, arbres),
    "G + cumuls 3 et 7 j": (BASE + CUM, arbres),
    "G + cumuls + attributs": (BASE + CUM + ATT, arbres),
    "G + cumuls + coordonnées": (BASE + CUM + XY, arbres),
}


def ajuster_predire(nom, Xa, Xe):
    ya = Xa.y.to_numpy()
    if MODELES[nom] is None or MODELES[nom] == "fonte":
        p = loi_dj(Xa, ya, fonte_seule=MODELES[nom] == "fonte")
        return np.clip(predire_dj(p, Xe), 0, None)
    cols, fab = MODELES[nom]
    m = fab(cols)
    m.fit(Xa[cols].to_numpy(), ya)
    return np.clip(m.predict(Xe[cols].to_numpy()), 0, None)


def evaluer(nom, paires):
    y = np.concatenate([p[0] for p in paires])
    h = np.concatenate([p[1] for p in paires])
    fond = y > 0.5
    eam = np.abs(h - y)[fond].mean()
    biais = (h - y)[fond].mean()
    eam_tous = np.abs(h - y).mean()
    print(f"  {nom:28s} jours de fonte : écart absolu moyen {eam:5.2f} mm/j, biais {biais:+5.2f} ; tous jours secs : {eam_tous:4.2f} ; n fonte {fond.sum()}")
    return eam


def main():
    X = pd.concat([charger(r) for r in REGIONS], ignore_index=True)
    # SAISON DE FONTE SEULE (argument `fonte`) : mars à juin, stock de plus de 10 mm. Hors de
    # cette fenêtre, 57 % des baisses de masse surviennent par température moyenne négative
    # (vent, tassement, mesure) et une loi apprise sur tout apprend ce bruit.
    if len(sys.argv) > 1 and sys.argv[1] == "fonte":
        X = X[(X.doy >= 60) & (X.doy <= 181) & (X.swe > 10)].reset_index(drop=True)
        print("SAISON DE FONTE SEULE : mars à juin, stock > 10 mm")
    print(f"{len(X)} jours-sites sans précipitation, {X.site.nunique()} sites, ablation moyenne {X.y.mean():.2f} mm/j, {100 * (X.y > 0.5).mean():.0f} % de jours de fonte")
    for r in REGIONS:
        s = X[X.region == r]
        print(f"  {r} : {len(s)} jours-sites, {s.site.nunique()} sites")
    res = {}
    print("\nA. sites retirés (cinq plis)")
    for nom in MODELES:
        paires = []
        for tr, te in GroupKFold(5).split(X, groups=X.site):
            paires.append((X.y.to_numpy()[te], ajuster_predire(nom, X.iloc[tr], X.iloc[te])))
        res[("sites", nom)] = evaluer(nom, paires)
    print("\nB. années retirées (une sur cinq)")
    for nom in MODELES:
        paires = []
        for k in range(5):
            te = (X.annee % 5 == k).to_numpy()
            paires.append((X.y.to_numpy()[te], ajuster_predire(nom, X[~te], X[te])))
        res[("annees", nom)] = evaluer(nom, paires)
    print("\nC. territoire retiré (apprentissage sur les trois autres)")
    for nom in MODELES:
        paires = []
        for r in REGIONS:
            te = (X.region == r).to_numpy()
            paires.append((X.y.to_numpy()[te], ajuster_predire(nom, X[~te], X[te])))
        res[("territoire", nom)] = evaluer(nom, paires)
    print("\n  par territoire retiré, écart absolu moyen sur les jours de fonte :")
    for r in REGIONS:
        te = (X.region == r).to_numpy()
        ligne = []
        for nom in ("DJ réajustée (témoin)", "G monotone, jour", "G + cumuls 3 et 7 j", "G + cumuls + attributs"):
            h = ajuster_predire(nom, X[~te], X[te])
            y = X.y.to_numpy()[te]
            f = y > 0.5
            ligne.append(f"{nom.split(' (')[0]} {np.abs(h - y)[f].mean():.2f}")
        print(f"    {r} : " + " ; ".join(ligne))
    p = loi_dj(X, X.y.to_numpy())
    print(f"\nloi des degrés-jours réajustée sur tout : cf {p[0]:.2f} mm/j/°C, T_melt {p[1]:+.2f} °C")
    pd.Series({f"{k[0]}|{k[1]}": v for k, v in res.items()}).to_csv(f"{DOSSIER}/fonte_observee.csv")


if __name__ == "__main__":
    main()
