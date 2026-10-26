"""Étapes 1 et 2 : quelle part de l'écart entre débit observé et simulé est prévisible, et
quelle est sa structure. Aucune simulation, quelques minutes.

Cible : y = log((Q_obs + e) / (Q_sim + e)) par station et par jour, e = 1 % du débit moyen
observé de la station. Une correction parfaite multiplie Q_sim + e par exp(y).

Prédicteurs, tous connus sans observation de débit (ni débit observé retardé : ce serait de
l'assimilation, une autre question) :
  - minimal : log Q_sim centré par station et ses retards, jour de l'année (le GRU minimal
    de l'été) ;
  - forçage : cumuls de pluie et moyennes de température sur 1 à 90 jours, degrés-jours ;
  - neige : un manteau de degrés-jours calculé sur le forçage seul, indicateur d'état ;
  - statiques : attributs du tronçon exutoire et paramètres du champ spatial.

Deux validations :
  A. années : on retire une année et on apprend sur les cinq autres, toutes stations vues ;
     plafond d'une correction aux stations jaugées.
  B. stations et futur : apprentissage sur 2019-2021 avec 80 % des stations, évaluation sur
     2022-2024 aux 20 % retirés, cinq plis ; c'est ce qu'exigerait une carte provinciale.

    .venv/Scripts/python.exe experiments/gru_residual/plafond.py [log|lin]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/gru_residual"
JOURS_MIN = 365
BORNE = np.log(2.0)   # correction plafonnée à un facteur deux, pour le chiffrage seulement
# Cible : "log" (rapport, comme le GRU de l'été) ou "lin" (écart en débit divisé par le débit
# moyen SIMULÉ de la station, connu même sans station : y = (Q_obs - Q_sim) / moy(Q_sim)).
CIBLE = sys.argv[1] if len(sys.argv) > 1 else "log"


def kge(s, o):
    ok = np.isfinite(s) & np.isfinite(o)
    s, o = s[ok], o[ok]
    if len(o) < 30 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def etiage(s, o, temps):
    """Médiane sur les années du rapport des minimums estivaux (juin-octobre) de la moyenne
    glissante sur sept jours, simulé sur observé ; jours mesurés seulement."""
    d = pd.DataFrame({"s": s, "o": o}, index=temps)
    d.loc[d.o.isna(), "s"] = np.nan
    r7 = d.rolling(7, min_periods=7).mean()
    r7 = r7[(r7.index.month >= 6) & (r7.index.month <= 10)]
    m = r7.groupby(r7.index.year).min()
    m = m[(m.o > 0) & m.s.notna()]
    return float(np.median(m.s / m.o)) if len(m) else np.nan


def neige_degres_jours(p, tmoy, ddf=3.0, seuil=0.0):
    """Manteau de degrés-jours sur le forçage seul : accumulation sous 0 °C, fonte de ddf mm
    par degré-jour au-dessus."""
    swe = np.zeros_like(p)
    s = 0.0
    for t in range(len(p)):
        if tmoy[t] < seuil:
            s += p[t]
        else:
            s = max(0.0, s - ddf * (tmoy[t] - seuil))
        swe[t] = s
    return swe


def construire():
    q = pd.read_parquet(f"{DOSSIER}/debits.parquet")
    fx = pd.read_parquet(f"{DOSSIER}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{DOSSIER}/attributs.parquet")
    temps = q.index
    statiques = [c for c in at.columns if c not in ("region", "n_troncons")]
    blocs = []
    for st in q["q_obs"].columns:
        o = q["q_obs"][st].to_numpy(float)
        s = q["q_sim"][st].to_numpy(float)
        if np.isfinite(o).sum() < JOURS_MIN or np.nanmean(o) <= 0:
            continue
        e = 0.01 * np.nanmean(o)
        f = fx[st]
        p = f["P"].to_numpy(float)
        tm = ((f["Tmin"] + f["Tmax"]) / 2).to_numpy(float)
        swe = neige_degres_jours(p, tm)
        F = pd.DataFrame(index=f.index)
        for w in (1, 3, 7, 30, 90):
            F[f"P{w}"] = pd.Series(p, index=f.index).rolling(w).sum()
        for w in (1, 7, 30):
            F[f"T{w}"] = pd.Series(tm, index=f.index).rolling(w).mean()
        for w in (7, 30):
            F[f"DJ{w}"] = pd.Series(np.maximum(tm, 0), index=f.index).rolling(w).sum()
        F["swe"] = swe
        F["fonte7"] = -pd.Series(swe, index=f.index).diff(7).clip(upper=0)
        F["Rn30"] = f["R_n"].rolling(30).mean()
        F = F.reindex(temps)
        ls = np.log(s + e)
        lsc = ls - np.nanmean(ls)
        d = pd.DataFrame(index=temps)
        d["station"] = st
        d["region"] = at.loc[st, "region"]
        d["annee"] = temps.year
        d["mois"] = temps.month
        d["qm"] = np.nanmean(s)
        d["y"] = np.log(o + e) - ls if CIBLE == "log" else (o - s) / np.nanmean(s)
        d["q_obs"] = o
        d["q_sim"] = s
        d["e"] = e
        d["lqs"] = lsc
        for k in (1, 3, 7, 30):
            d[f"lqs_l{k}"] = pd.Series(lsc, index=temps).shift(k)
        d["dlqs"] = d["lqs"] - d["lqs_l1"]
        d["doy_s"] = np.sin(2 * np.pi * temps.dayofyear / 365.25)
        d["doy_c"] = np.cos(2 * np.pi * temps.dayofyear / 365.25)
        d["lqs_niveau"] = np.nanmean(ls) - np.log(max(at.loc[st, "n_troncons"], 1))
        d = pd.concat([d, F], axis=1)
        for c in statiques:
            d[f"st_{c}"] = at.loc[st, c]
        blocs.append(d)
    D = pd.concat(blocs)
    D = D[D["lqs_l30"].notna()]
    return D


def groupes(D):
    minimal = ["lqs", "lqs_l1", "lqs_l3", "lqs_l7", "lqs_l30", "dlqs", "doy_s", "doy_c"]
    forcage = [c for c in D.columns if c[:1] in "PT" and c[1:].isdigit()] + ["DJ7", "DJ30", "Rn30"]
    neige = ["swe", "fonte7"]
    stat = [c for c in D.columns if c.startswith("st_")] + ["lqs_niveau"]
    return {"minimal": minimal,
            "minimal+forçage": minimal + forcage,
            "minimal+forçage+neige": minimal + forcage + neige,
            "tout+statiques": minimal + forcage + neige + stat}


def modeles():
    return {"ridge": lambda: make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))),
            "arbres": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)}


def evaluer(T, yhat):
    """R² de la cible, puis KGE, volume et étiage par station avant et après correction."""
    ok = T["q_obs"].notna().to_numpy()
    y = T["y"].to_numpy()
    r2 = 1 - np.nansum((y - yhat)[ok] ** 2) / np.nansum((y[ok] - np.nanmean(y[ok])) ** 2)
    if CIBLE == "log":
        yb = np.clip(yhat, -BORNE, BORNE)
        qc = np.maximum((T["q_sim"] + T["e"]).to_numpy() * np.exp(yb) - T["e"].to_numpy(), 0)
    else:
        qc = np.maximum(T["q_sim"].to_numpy() + yhat * T["qm"].to_numpy(), 0)
    T = T.assign(qc=qc)
    lignes = []
    for st, g in T.groupby("station"):
        o = g["q_obs"].to_numpy()
        if np.isfinite(o).sum() < 180:
            continue
        m = np.isfinite(o)
        yy, hh = g["y"].to_numpy()[m], yhat[T["station"].to_numpy() == st][m]
        lignes.append(dict(station=st,
                           r2=1 - np.sum((yy - hh) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-12),
                           kge_av=kge(g["q_sim"].to_numpy(), o), kge_ap=kge(g["qc"].to_numpy(), o),
                           vol_av=np.nansum(g["q_sim"].to_numpy()[m]) / np.nansum(o[m]),
                           vol_ap=np.nansum(g["qc"].to_numpy()[m]) / np.nansum(o[m]),
                           et_av=etiage(g["q_sim"].to_numpy(), o, g.index), et_ap=etiage(g["qc"].to_numpy(), o, g.index)))
    L = pd.DataFrame(lignes)
    return dict(r2_global=r2, r2_median=L.r2.median(),
                kge_av=L.kge_av.median(), kge_ap=L.kge_ap.median(),
                part_ameliorees=(L.kge_ap > L.kge_av).mean(),
                vol_av=(L.vol_av - 1).abs().median(), vol_ap=(L.vol_ap - 1).abs().median(),
                etiage_av=np.abs(np.log(L.et_av)).median(), etiage_ap=np.abs(np.log(L.et_ap)).median(),
                n=len(L)), L


def validation_annees(D, cols, fabrique):
    yhat = np.full(len(D), np.nan)
    for a in sorted(D.annee.unique()):
        tr = (D.annee != a) & D.y.notna()
        te = (D.annee == a).to_numpy()
        m = fabrique()
        X = D[cols]
        m.fit(X[tr], D.y[tr])
        yhat[te] = m.predict(X[te])
    return yhat


def validation_stations(D, cols, fabrique, plis=5):
    """Apprentissage 2019-2021 sur les stations des autres plis, évaluation 2022-2024 aux
    stations du pli. Plis par station, pas par territoire."""
    stations = np.array(sorted(D.station.unique()))
    gkf = GroupKFold(n_splits=plis)
    yhat = np.full(len(D), np.nan)
    for _, ite in gkf.split(stations, groups=stations):
        tenues = set(stations[ite])
        dans = D.station.isin(tenues).to_numpy()
        tr = (~dans) & (D.annee <= 2021).to_numpy() & D.y.notna().to_numpy()
        te = dans & (D.annee >= 2022).to_numpy()
        m = fabrique()
        X = D[cols]
        m.fit(X[tr], D.y[tr])
        yhat[te] = m.predict(X[te])
    return yhat


def correction_saisonniere(D, par_station=True):
    """Témoin sans réseau : moyenne de y par (station, mois) apprise sur 2019-2021, posée sur
    2022-2024. Sans par_station : moyenne par mois sur toutes les stations."""
    tr = D[(D.annee <= 2021) & D.y.notna()]
    cle = ["station", "mois"] if par_station else ["mois"]
    moy = tr.groupby(cle).y.mean().rename("yh")
    return D.join(moy, on=cle)["yh"].fillna(0).to_numpy()


def structure(D):
    """Décomposition de la variance de y sur 2019-2024 : biais constant de station, cycle
    saisonnier de station, reste ; persistance du reste ; écart par saison et par régime."""
    d = D[D.y.notna()].copy()
    v = d.y.var()
    d["b"] = d.groupby("station").y.transform("mean")
    d["s"] = d.groupby(["station", "mois"]).y.transform("mean")
    d["reste"] = d.y - d.s
    print(f"  variance de y : {v:.4f}")
    print(f"  part du biais constant par station : {d.b.var() / v:.3f}")
    print(f"  part du cycle mensuel moyen par station (au-delà du biais) : {(d.s - d.b).var() / v:.3f}")
    print(f"  part du reste (dans l'année, hors cycle moyen) : {d.reste.var() / v:.3f}")
    ac = []
    for _, g in d.groupby("station"):
        r = g["reste"].asfreq("D")
        ac.append([r.autocorr(1), r.autocorr(7), r.autocorr(30)])
    ac = np.nanmedian(np.array(ac), axis=0)
    print(f"  autocorrélation médiane du reste : 1 j {ac[0]:.2f}, 7 j {ac[1]:.2f}, 30 j {ac[2]:.2f}")
    print("  y moyen et écart-type par mois (toutes stations ; y > 0 : le modèle sous-estime) :")
    t = d.groupby("mois").y.agg(["mean", "std"]).T.round(2)
    print(t.to_string())
    d["regime"] = d.groupby("station").q_obs.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=["très bas", "bas", "moyen", "haut", "très haut"]))
    print("  y moyen par quintile du débit observé de chaque station :")
    print(d.groupby("regime", observed=True).y.agg(["mean", "std"]).T.round(2).to_string())
    print("  y moyen par territoire (biais, positif = sous-estimation) :")
    print(d.groupby("region").y.agg(["mean", "std", "count"]).round(2).T.to_string())


def main():
    D = construire()
    print(f"cible {CIBLE}")
    print(f"{D.station.nunique()} stations, {int(D.y.notna().sum())} jours-station mesurés, 2019-2024")
    print("\nSTRUCTURE DE L'ÉCART (étape 2)")
    structure(D)
    G = groupes(D)
    M = modeles()
    res = []
    par_station = {}
    print("\nPLAFOND (étape 1)")
    for nom_v, fn in (("A années", validation_annees), ("B stations et futur", validation_stations)):
        for nom_g, cols in G.items():
            if nom_v.startswith("A") and nom_g == "tout+statiques":
                continue
            for nom_m, fab in M.items():
                yhat = fn(D, cols, fab)
                masque = np.isfinite(yhat)
                r, L = evaluer(D[masque], yhat[masque])
                r.update(validation=nom_v, predicteurs=nom_g, modele=nom_m)
                res.append(r)
                par_station[(nom_v, nom_g, nom_m)] = L
                print(f"  {nom_v:20s} {nom_g:22s} {nom_m:7s} R² {r['r2_global']:+.3f} (médiane par station {r['r2_median']:+.3f}) ; KGE médian {r['kge_av']:.3f} -> {r['kge_ap']:.3f}, {100 * r['part_ameliorees']:.0f} % améliorées ; |volume-1| {r['vol_av']:.3f} -> {r['vol_ap']:.3f} ; |log étiage| {r['etiage_av']:.3f} -> {r['etiage_ap']:.3f}", flush=True)
    print("\nTÉMOINS SANS APPRENTISSAGE SÉQUENTIEL (2022-2024)")
    te = (D.annee >= 2022).to_numpy()
    for nom, ps in (("cycle mensuel par station", True), ("cycle mensuel commun", False)):
        yh = correction_saisonniere(D, ps)
        r, _ = evaluer(D[te], yh[te])
        print(f"  {nom:28s} R² {r['r2_global']:+.3f} ; KGE médian {r['kge_av']:.3f} -> {r['kge_ap']:.3f}, {100 * r['part_ameliorees']:.0f} % améliorées ; |volume-1| {r['vol_av']:.3f} -> {r['vol_ap']:.3f} ; |log étiage| {r['etiage_av']:.3f} -> {r['etiage_ap']:.3f}", flush=True)
        res.append(dict(r, validation="témoin 2022-2024", predicteurs=nom, modele="moyenne"))
    pd.DataFrame(res).to_csv(f"{DOSSIER}/plafond-{CIBLE}.csv", index=False)
    pd.concat({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in par_station.items()}).to_parquet(f"{DOSSIER}/plafond_par_station-{CIBLE}.parquet")
    print(f"\nécrit : {DOSSIER}/plafond-{CIBLE}.csv")


if __name__ == "__main__":
    main()
