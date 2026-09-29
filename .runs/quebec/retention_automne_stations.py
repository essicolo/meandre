"""Quelle part de la pluie de septembre-octobre le bassin réel retient-il, et qu'est-ce qui la porte ?

Sur l'Outaouais 040110, le modèle rend en octobre 66 % de la précipitation en débit quand la
station en rend 36 % : le bassin réel retient environ 2 mm/j que MOD16 n'évapore pas. Avant de
chercher la pièce du modèle qui manque, on mesure sur toutes les stations la rétention
d'automne observée, (P − Q) / P sur septembre-octobre, avec P de CaSR pondérée sur le bassin
amont, et on la met en face des lacs, milieux humides, forêt et texture amont. Une rétention
qui suit les lacs désigne un remplissage de plans d'eau ; qui ne suit rien, un déficit de sol
ou de nappe à combler, ou une précipitation d'automne surestimée par le forçage.

    python .runs/quebec/retention_automne_stations.py outv mont slso slno sagu gasp
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

FENETRES = {"sep-oct": (9, 10), "nov-dec": (11, 12), "avr-mai": (4, 5)}
# Jours observes exiges par annee et par fenetre ; 25 suffit pour une fenetre d un mois.
MIN_JOURS = 50


def territoire(reg):
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
    terr = con.execute("select * from territorial").df().set_index("node_idx")
    edges = [(e[0], e[1]) for e in con.execute("select * from edges").fetchall()]
    obs = con.execute("select station_id, date, discharge from observations where discharge is not null").df()
    con.close()
    _amont = amont(edges, len(terr))
    aire = terr["area_km2_local"].to_numpy()
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
    P = ds["forcing"].sel(var="P")
    temps = pd.DatetimeIndex(ds.time.values)
    lignes = []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id]
        if len(o) < 2000:
            continue
        up = _amont(int(s.node_idx))
        w = aire[up]; w = w / w.sum() if w.sum() > 0 else np.ones(len(up)) / len(up)
        p = pd.Series((P.isel(node=up).values * w[None, :]).sum(axis=1), index=temps)
        conv = 86400.0 * 1000.0 / (float(s.drainage_area_km2) * 1e6)
        q = pd.Series(o.discharge.values * conv, index=pd.DatetimeIndex(o.date)).reindex(temps)
        ligne = {"territoire": reg, "station": str(s.station_id), "aire_km2": float(s.drainage_area_km2)}
        for nom, mois in FENETRES.items():
            m = temps.month.isin(mois)
            df = pd.DataFrame({"p": p[m], "q": q[m]}).dropna()
            par_an = df.groupby(df.index.year).agg(p=("p", "mean"), q=("q", "mean"), n=("q", "size"))
            par_an = par_an[par_an.n >= MIN_JOURS]
            if len(par_an) < 5:
                ligne[f"ret_{nom}"] = np.nan
                continue
            ligne[f"ret_{nom}"] = float(np.median((par_an.p - par_an.q) / par_an.p))
            ligne[f"p_{nom}"] = float(par_an.p.median())
            ligne[f"q_{nom}"] = float(par_an.q.median())
        for a in ATTRIBUTS:
            if a in terr.columns:
                ligne[a] = float((terr[a].to_numpy()[up] * w).sum())
        lignes.append(ligne)
    ds.close()
    return pd.DataFrame(lignes)


def main(regs):
    t = pd.concat([territoire(r) for r in regs], ignore_index=True)
    print(f"{len(t)} stations, retention (P - Q) / P par fenetre, mediane des annees ; P de CaSR sur le bassin amont")
    print(t.groupby("territoire")[["ret_sep-oct", "ret_nov-dec", "ret_avr-mai", "p_sep-oct", "q_sep-oct"]].median().round(2).to_string())
    attrs = [a for a in ATTRIBUTS if a in t.columns and t[a].notna().any() and t[a].std() > 0]
    for nom in FENETRES:
        col = f"ret_{nom}"
        print(f"  correlation (Spearman) de la retention {nom} avec les attributs amont :")
        print("    " + ", ".join(f"{a} {t[[a, col]].dropna().corr(method='spearman').iloc[0, 1]:+.2f}" for a in attrs))
    m = t[t.station.str.contains("40110|30905")]
    print("  sous-bassins du banc :")
    print(m[["territoire", "station", "ret_sep-oct", "p_sep-oct", "q_sep-oct", "ret_nov-dec", "ret_avr-mai"]].round(2).to_string())
    f = f"{_p.DERIVED_ROOT}/auxiliaires/retention-automne-stations.csv"
    t.to_csv(f, index=False)
    print(f"  ecrit : {f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["outv", "mont", "slso", "slno", "sagu", "gasp"])
