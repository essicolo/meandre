"""Orages d'été : la colonne produit-elle trop peu, ou le routage étale-t-il la production ?
Moyennes de bassin (ETL_DUMP_STATIONS, préfixe basin_), modèle provincial gelé, 2011-2024.

Pour chaque orage (pluie de bassin d'au moins 10 mm, juin à septembre, après deux jours à moins de
2 mm), trois coefficients rapportés à la pluie des cinq jours :
  production : surplus de production du bassin (surface, hypodermique, base) sur cinq jours
               au-dessus de la production de la veille ;
  simulé : hausse du débit simulé à la station au-dessus de la veille, sur cinq jours ;
  observé : même chose sur le débit observé.
Production et simulé proches désignent la colonne ; production nettement au-dessus du simulé
désigne le routage (et les lacs), qui étale l'événement au-delà de cinq jours.

    .venv/Scripts/python.exe experiments/plan_ml/bassin-orages.py slso
"""
import sys

import numpy as np
import pandas as pd

DOSSIER = "D:/meandre-data/experiments/plan_ml"
REF = "D:/meandre-data/experiments/gru_residual"


def aires_modele(reg):
    import duckdb
    from collections import defaultdict
    con = duckdb.connect(f"D:/meandre-data/quebec/{reg}.duckdb", read_only=True)
    st = con.sql("select station_id, node_idx from stations").fetchall()
    aire = con.sql("select area_km2_local from territorial order by node_idx").fetchnumpy()["area_km2_local"]
    ed = con.sql("select src, dst from edges").df()
    con.close()
    par = defaultdict(list)
    for a, b in zip(ed.src, ed.dst):
        par[int(b)].append(int(a))
    out = {}
    for sid, n0 in st:
        vus, pile = {int(n0)}, [int(n0)]
        while pile:
            for p in par.get(pile.pop(), []):
                if p not in vus:
                    vus.add(p)
                    pile.append(p)
        out[f"{reg}:{sid}"] = float(aire[np.fromiter(vus, int)].sum())
    return out


SUFFIXE = sys.argv[2] if len(sys.argv) > 2 else ""


def main(reg):
    global AIRES_MODELE
    AIRES_MODELE = aires_modele(reg)
    D = np.load(f"{DOSSIER}/diag-bassin{SUFFIXE}-{reg}.npz", allow_pickle=True)
    Z = np.load(f"{DOSSIER}/q-bassin{SUFFIXE}-{reg}.npz", allow_pickle=True)
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    td, tq = pd.DatetimeIndex(D["dates"]), pd.DatetimeIndex(Z["dates"])
    idq = [str(s) for s in Z["station_ids"]]
    lignes = []
    for j, s in enumerate([str(x) for x in D["station_ids"]]):
        st = f"{reg}:{s}"
        if st not in fx.columns.get_level_values(0) or s not in idq:
            continue
        k = idq.index(s)
        # Conversion par l'aire du bassin DANS LE MODÈLE (somme des aires locales en amont) : le débit
        # simulé et la production sont alors sur la même aire. Le débit observé, lui, se rapporte à
        # l'aire déclarée de la station ; un écart d'aire fausse sinon toute comparaison.
        conv = 86.4 / AIRES_MODELE[st]
        conv_obs = 86.4 / float(at.loc[st, "aire_km2"])
        o = pd.Series(Z["q_obs"][:, k] * conv_obs, index=tq)
        q = pd.Series(Z["q_sim"][:, k] * conv, index=tq)
        prod = pd.Series(D["basin_prod_surf"][:, j] + D["basin_prod_hypo"][:, j] + D["basin_prod_base"][:, j], index=td).reindex(tq)
        surf = pd.Series(D["basin_prod_surf"][:, j], index=td).reindex(tq)
        etr = pd.Series(D["basin_etr"][:, j], index=td).reindex(tq)
        th2 = pd.Series(D["basin_theta2"][:, j], index=td).reindex(tq)
        p = fx[st]["P"].reindex(tq)
        sec = (p.shift(1) < 2) & (p.shift(2) < 2)
        ev = (p >= 10) & sec & tq.month.isin([6, 7, 8, 9])
        for t in tq[ev]:
            i = tq.get_loc(t)
            if i < 1 or i + 5 >= len(tq) or o.iloc[i:i + 5].isna().any() or np.isnan(o.iloc[i - 1]):
                continue
            w = slice(i, i + 5)
            pl = p.iloc[w].sum()
            lignes.append(dict(station=st, rapport_aire=AIRES_MODELE[st] / float(at.loc[st, "aire_km2"]), pluie=pl, c_prod=(prod.iloc[w] - prod.iloc[i - 1]).clip(lower=0).sum() / pl, c_surf=(surf.iloc[w] - surf.iloc[i - 1]).clip(lower=0).sum() / pl, c_sim=(q.iloc[w] - q.iloc[i - 1]).clip(lower=0).sum() / pl, c_obs=(o.iloc[w] - o.iloc[i - 1]).clip(lower=0).sum() / pl, prod_annee=prod.loc[str(t.year)].mean(), q_annee=q.loc[str(t.year)].mean(), theta2=th2.iloc[i - 1], etr=etr.iloc[i - 5:i].mean()))
    E = pd.DataFrame(lignes).dropna()
    E.to_parquet(f"{DOSSIER}/bassin-orages{SUFFIXE}-{reg}.parquet")
    print(f"{reg} : {len(E)} orages, {E.station.nunique()} stations, 2011-2024 ; médianes")
    print(E[["pluie", "c_prod", "c_surf", "c_sim", "c_obs"]].median().round(3).to_string())
    print(f"rapport simulé / production (part de la production d'orage qui arrive en cinq jours) : {np.median(E.c_sim / E.c_prod.clip(lower=1e-4)):.2f}")
    print(f"débit moyen annuel / production moyenne annuelle (bilan sur l'année, doit valoir environ 1) : {np.median(E.q_annee / E.prod_annee):.2f}")
    E["classe"] = pd.cut(E.pluie, [0, 20, 35, 1000], labels=["10-20 mm", "20-35 mm", "> 35 mm"])
    print(E.groupby("classe", observed=True)[["c_prod", "c_sim", "c_obs"]].median().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
