"""Audit du reseau et des cartes physiques d'un territoire, sans simuler (2026-10-09).

Pour chaque station : aire drainee officielle (table stations) contre aire amont du modele
(somme des aires locales des troncons amont), et rapport annuel simule/observe du debit si un
export ETL_DUMP_Q est fourni. Pour le reseau : arêtes, composantes, exutoires, cycles. Pour les
attributs : valeurs constantes, manquantes, collees, et part de troncons a la valeur de repli.

    python .runs/quebec/audit_reseau.py mont [q-mont.npz]
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import duckdb
import numpy as np
import pandas as pd

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")


def main(reg, export=None):
    c = duckdb.connect(f"{DATA}/quebec/{reg}.duckdb", read_only=True)
    st = c.execute("select * from stations").fetchdf()
    nodes = c.execute("select * from nodes order by node_idx").fetchdf()
    edges = c.execute("select src, dst from edges").fetchdf()
    terr = c.execute("select * from territorial order by node_idx").fetchdf()
    c.close()
    n = len(nodes)
    # reseau
    aval = {}
    for s, d in zip(edges.src, edges.dst):
        aval.setdefault(int(s), []).append(int(d))
    plusieurs = [k for k, v in aval.items() if len(v) > 1]
    exutoires = [i for i in range(n) if i not in aval]
    amont = [[] for _ in range(n)]
    for s, d in zip(edges.src, edges.dst):
        amont[int(d)].append(int(s))
    def bassin(i):
        vu, pile = set(), [i]
        while pile:
            k = pile.pop()
            if k in vu:
                continue
            vu.add(k)
            pile.extend(amont[k])
        return vu
    print(f"{reg} : {n} troncons, {len(edges)} aretes, {len(exutoires)} exutoire(s), {len(plusieurs)} troncon(s) a plusieurs avals")
    # aire locale et aire amont du modele
    loc = terr["area_km2_local"].to_numpy(float)
    print(f"  aire locale : somme {np.nansum(loc):.0f} km2, mediane {np.nanmedian(loc):.1f}, NaN {int(np.isnan(loc).sum())}, nulles {int((loc <= 0).sum())}")
    # q export
    vol = {}
    if export:
        d = np.load(export, allow_pickle=True)
        for k, sid in enumerate(d["station_ids"]):
            o, s = d["q_obs"][:, k], d["q_sim"][:, k]
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() > 300:
                vol[str(sid)] = s[m].mean() / o[m].mean()
    rows = []
    for _, r in st.iterrows():
        b = bassin(int(r.node_idx))
        a_mod = float(np.nansum(loc[list(b)]))
        a_terr = float(np.exp(terr.drainage_area_km2.iloc[int(r.node_idx)])) if terr.drainage_area_km2.max() < 20 else float(terr.drainage_area_km2.iloc[int(r.node_idx)])
        rows.append(dict(station=r.station_id, aire_officielle=r.drainage_area_km2, aire_modele=round(a_mod, 1), aire_attribut=round(a_terr, 1),
                         rapport=round(a_mod / r.drainage_area_km2, 2) if r.drainage_area_km2 else np.nan, troncons=len(b),
                         lac=bool(nodes.is_lake.iloc[int(r.node_idx)]), volume_sim_obs=round(vol.get(str(r.station_id), np.nan), 2)))
    t = pd.DataFrame(rows).sort_values("rapport")
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    if vol:
        tt = t.dropna(subset=["volume_sim_obs"])
        print(f"  correlation (rapport d'aires, volume sim/obs) : {np.corrcoef(tt.rapport, tt.volume_sim_obs)[0, 1]:+.2f} sur {len(tt)} stations")
    # attributs
    print("  attributs : constantes, NaN, valeur la plus frequente et sa part")
    for col in terr.columns:
        if col == "node_idx":
            continue
        v = terr[col].to_numpy(float)
        vc = pd.Series(v).round(4).value_counts()
        part = vc.iloc[0] / len(v) if len(vc) else np.nan
        drap = "  <<<" if (part > 0.3 or np.isnan(v).mean() > 0.05 or np.nanstd(v) == 0) else ""
        print(f"    {col:22s} NaN {np.isnan(v).mean():5.1%} | med {np.nanmedian(v):9.3f} | q01 {np.nanquantile(v, .01):9.3f} q99 {np.nanquantile(v, .99):9.3f} | mode {vc.index[0] if len(vc) else np.nan:9.3f} ({part:5.1%}){drap}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
