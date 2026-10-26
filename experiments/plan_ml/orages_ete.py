"""Candidate 5.3, épreuve 3.1 : la part ruisselée d'un orage d'été dépend-elle de l'humidité
antécédente, et le modèle la reproduit-il ? Sans simulation nouvelle : débits observés et
simulés du modèle provincial gelé aux stations, 2019-2024.

Événement : jour de juin à septembre avec pluie de bassin d'au moins 10 mm, précédé de deux jours
à moins de 2 mm. Volume d'événement : somme sur cinq jours de la hausse du débit spécifique
au-dessus du débit de la veille (mm), rapportée à la pluie (coefficient d'écoulement). Humidité
antécédente : indice de précipitation antérieure sur 30 jours (API, décroissance 0,9 par jour),
rapporté à sa médiane de la station pour comparer les stations.

Ce qu'une part (theta/thetas)^b uniforme prédit : un coefficient d'écoulement qui croît avec
l'humidité antécédente, de la même façon partout ; ce qui est en cause (R386, R389) : le volume
d'événement simulé vaut 0,64 de l'observé, 0,32 en Montérégie.

    .venv/Scripts/python.exe experiments/plan_ml/orages_ete.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"


def evenements(p, q_obs, q_sim):
    api = np.zeros(len(p))
    pv = p.to_numpy()
    for k in range(1, len(p)):
        api[k] = 0.9 * api[k - 1] + pv[k - 1]
    api = pd.Series(api, index=p.index)
    sec = (p.shift(1) < 2.0) & (p.shift(2) < 2.0)
    ev = (p >= 10.0) & sec & p.index.month.isin((6, 7, 8, 9))
    lignes = []
    for t in p.index[ev]:
        i = p.index.get_loc(t)
        if i + 5 >= len(p) or i < 1:
            continue
        fen = slice(i, i + 5)
        o, s = q_obs.iloc[fen], q_sim.iloc[fen]
        if o.isna().any() or s.isna().any() or not np.isfinite(q_obs.iloc[i - 1]):
            continue
        pl = p.iloc[fen].sum()
        lignes.append(dict(date=t, pluie=pl, api=api.iloc[i], vol_obs=(o - q_obs.iloc[i - 1]).clip(lower=0).sum() / pl, vol_sim=(s - q_sim.iloc[i - 1]).clip(lower=0).sum() / pl, q0_obs=q_obs.iloc[i - 1], q0_sim=q_sim.iloc[i - 1]))
    return pd.DataFrame(lignes)


def main():
    q = pd.read_parquet(f"{REF}/debits.parquet")
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    blocs = []
    for st in q["q_obs"].columns:
        o, s = q["q_obs"][st], q["q_sim"][st]
        if o.notna().sum() < 365 or at.loc[st, "aire_km2"] <= 0:
            continue
        conv = 86.4 / float(at.loc[st, "aire_km2"])
        p = fx[st]["P"].reindex(q.index)
        e = evenements(p, o * conv, s * conv)
        if len(e) < 5:
            continue
        e["station"] = st
        e["region"] = st.split(":")[0]
        e["api_rel"] = e.api / e.api.median()
        for a in ("f_agriculture", "f_forest", "f_wetland", "lake_fraction", "mean_slope_pct", "f_clay", "f_sand"):
            e[a] = at.loc[st, a]
        blocs.append(e)
    E = pd.concat(blocs, ignore_index=True)
    E.to_parquet(f"{DOSSIER}/orages_ete.parquet")
    print(f"{len(E)} orages d'été (≥ 10 mm après deux jours secs), {E.station.nunique()} stations, 2019-2024")
    print(f"coefficient d'écoulement médian : observé {E.vol_obs.median():.3f}, simulé {E.vol_sim.median():.3f} ; rapport médian par station sim/obs {E.groupby('station').apply(lambda d: d.vol_sim.median() / max(d.vol_obs.median(), 1e-4)).median():.2f}")
    E["classe_api"] = pd.cut(E.api_rel, [0, 0.5, 0.8, 1.2, 1.6, 2.5, 100], labels=["< 0,5", "0,5-0,8", "0,8-1,2", "1,2-1,6", "1,6-2,5", "> 2,5"])
    print("\ncoefficient d'écoulement médian par classe d'humidité antécédente (API rapporté à la médiane de la station) :")
    t = E.groupby("classe_api", observed=True).agg(n=("vol_obs", "size"), obs=("vol_obs", "median"), sim=("vol_sim", "median"))
    t["sim/obs"] = t.sim / t.obs
    print(t.round(3).to_string())
    E["classe_pluie"] = pd.cut(E.pluie, [0, 15, 25, 40, 1000], labels=["10-15", "15-25", "25-40", "> 40"])
    print("\npar classe de pluie d'événement (mm sur cinq jours) :")
    t = E.groupby("classe_pluie", observed=True).agg(n=("vol_obs", "size"), obs=("vol_obs", "median"), sim=("vol_sim", "median"))
    t["sim/obs"] = t.sim / t.obs
    print(t.round(3).to_string())
    print("\npar territoire : coefficient observé, simulé, rapport ; pente de l'observé et du simulé en fonction de l'API relatif (régression de rang) :")
    for r in sorted(E.region.unique()):
        g = E[E.region == r]
        if len(g) < 30:
            continue
        co = g[["api_rel", "vol_obs"]].corr(method="spearman").iloc[0, 1]
        cs = g[["api_rel", "vol_sim"]].corr(method="spearman").iloc[0, 1]
        print(f"  {r}: obs {g.vol_obs.median():.3f}, sim {g.vol_sim.median():.3f}, rapport {g.vol_sim.median() / g.vol_obs.median():.2f} ; corrélation de rang avec l'API : obs {co:+.2f}, sim {cs:+.2f} (n {len(g)}, {g.station.nunique()} stations)")
    par_st = E.groupby("station").agg(obs=("vol_obs", "median"), sim=("vol_sim", "median"), **{a: (a, "first") for a in ("f_agriculture", "f_forest", "f_wetland", "lake_fraction", "mean_slope_pct", "f_clay", "f_sand")})
    par_st["rapport"] = par_st.sim / par_st.obs.clip(lower=1e-4)
    print("\ncorrélation de rang entre stations du coefficient OBSERVÉ et des attributs (transformés) : " + ", ".join(f"{a} {par_st[['obs', a]].corr(method='spearman').iloc[0, 1]:+.2f}" for a in ("f_agriculture", "f_forest", "f_wetland", "lake_fraction", "mean_slope_pct", "f_clay", "f_sand")))
    print("corrélation de rang du RAPPORT sim/obs et des attributs : " + ", ".join(f"{a} {par_st[['rapport', a]].corr(method='spearman').iloc[0, 1]:+.2f}" for a in ("f_agriculture", "f_forest", "f_wetland", "lake_fraction", "mean_slope_pct", "f_clay", "f_sand")))


if __name__ == "__main__":
    main()
