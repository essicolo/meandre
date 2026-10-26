"""Un sol gelé fait-il ruisseler la pluie, et où ? Test croisé sur toutes les stations, sans simulation.

Deux sous-bassins veulent des portes de gel opposées : l'Outaouais forestier veut un sol gelé
imperméable, la Châteauguay agricole drainée veut qu'il reste perméable (R221). Avant d'écrire
une porte qui dépende de l'occupation, on demande aux stations. Pour chaque pluie de novembre
à décembre d'au moins 5 mm par jour tombée par température positive, la réponse est la hausse
du débit sur les trois jours suivants rapportée à la pluie, en écoulement. Le sol est dit gelé
quand la somme des degrés-jours négatifs depuis le 15 octobre dépasse 20 °C·jour, non gelé
quand elle est sous 5. Le rapport de la réponse gelée à la réponse non gelée, par station,
est mis en face de la part agricole, de la pente et de la forêt du bassin amont.

    python .runs/quebec/reponse_sol_gele_stations.py outv mont slno sagu gasp
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from meandre.utils import paths as _p
from soutien_etiage_stations import ATTRIBUTS, amont


def reponses(p, tm, q):
    """Réponse aux pluies de novembre-décembre, séparée selon l'état du sol.

    L'état du sol est un indice de gel cumulé : somme des degrés-jours négatifs depuis le
    15 octobre. Sol gelé au-delà de 20 °C·jour, non gelé sous 5. Aucune condition sur la neige :
    en Outaouais, toute semaine froide de novembre porte déjà un manteau, et l'exiger vide
    l'échantillon (mesuré le 2026-09-29 : zéro événement sur 24 ans).
    """
    saison = pd.Series(np.where(p.index.month >= 10, p.index.year, p.index.year - 1), index=p.index)
    froid = (-tm).clip(lower=0.0)
    froid[(p.index.month < 10) | ((p.index.month == 10) & (p.index.day < 15))] = 0.0
    indice = froid.groupby(saison).cumsum().shift(1)
    dq = (q.shift(-1).rolling(3).max().shift(-2) - q.shift(1)).clip(lower=0.0)
    ev = (p >= 5.0) & (tm > 0.5) & p.index.month.isin((11, 12)) & q.notna() & dq.notna()
    rep = (dq / p)[ev]
    gel = indice >= 20.0
    chaud = indice < 5.0
    return rep[gel[ev]], rep[chaud[ev]]


def territoire(reg):
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
    terr = con.execute("select * from territorial").df().set_index("node_idx")
    edges = [(e[0], e[1]) for e in con.execute("select * from edges").fetchall()]
    obs = con.execute("select station_id, date, discharge, reconstructed from observations where discharge is not null").df()
    con.close()
    _amont = amont(edges, len(terr))
    aire = terr["area_km2_local"].to_numpy()
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
    F = ds["forcing"]
    temps = pd.DatetimeIndex(ds.time.values)
    lignes = []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id]
        o = o[~o.reconstructed.fillna(True).astype(bool)]
        if len(o) < 2000:
            continue
        up = _amont(int(s.node_idx))
        w = aire[up]; w = w / w.sum() if w.sum() > 0 else np.ones(len(up)) / len(up)
        f = F.isel(node=up).values
        p = pd.Series((f[:, :, 0] * w[None, :]).sum(axis=1), index=temps)
        tm = pd.Series((0.5 * (f[:, :, 1] + f[:, :, 2]) * w[None, :]).sum(axis=1), index=temps)
        conv = 86400.0 * 1000.0 / (float(s.drainage_area_km2) * 1e6)
        q = pd.Series(o.discharge.values * conv, index=pd.DatetimeIndex(o.date)).reindex(temps)
        rg, rc = reponses(p, tm, q)
        if len(rg) < 5 or len(rc) < 5:
            continue
        ligne = {"territoire": reg, "station": str(s.station_id), "aire_km2": float(s.drainage_area_km2), "n_gel": len(rg), "n_chaud": len(rc), "rep_gel": float(rg.median()), "rep_chaud": float(rc.median())}
        ligne["rapport"] = ligne["rep_gel"] / max(ligne["rep_chaud"], 1e-3)
        for a in ATTRIBUTS:
            if a in terr.columns:
                ligne[a] = float((terr[a].to_numpy()[up] * w).sum())
        lignes.append(ligne)
    ds.close()
    return pd.DataFrame(lignes)


def main(regs):
    t = pd.concat([territoire(r) for r in regs], ignore_index=True)
    print(f"{len(t)} stations avec au moins 5 pluies de chaque etat ; reponse = hausse du debit sur 3 jours / pluie")
    print(t.groupby("territoire")[["n_gel", "n_chaud", "rep_gel", "rep_chaud", "rapport"]].median().round(3).to_string())
    print(f"  toutes stations : reponse gelee {t.rep_gel.median():.3f}, non gelee {t.rep_chaud.median():.3f}, rapport median {t.rapport.median():.2f}, part des stations ou le gel repond plus : {100 * (t.rapport > 1).mean():.0f} %")
    attrs = [a for a in ("f_agriculture", "mean_slope_pct", "f_forest", "f_wetland", "lake_fraction", "f_clay", "f_sand") if a in t.columns]
    print("  correlation (Spearman) du rapport gele / non gele avec les attributs amont :")
    print("    " + ", ".join(f"{a} {t[[a, 'rapport']].dropna().corr(method='spearman').iloc[0, 1]:+.2f}" for a in attrs))
    m = t[t.station.str.contains("40110|30905")]
    print(m[["territoire", "station", "n_gel", "n_chaud", "rep_gel", "rep_chaud", "rapport", "f_agriculture", "mean_slope_pct"]].round(3).to_string())
    f = f"{_p.DERIVED_ROOT}/auxiliaires/reponse-sol-gele-stations.csv"
    t.to_csv(f, index=False)
    print(f"  ecrit : {f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["outv", "mont", "slno", "sagu", "gasp"])
