"""Fermeture du bilan d'eau annuel par bassin jaugé, sans modèle (2026-10-09).

Pour chaque station : pluie CaSR brut moyenne sur le bassin amont, débit observé, évapotranspiration
implicite P - Q, et évapotranspiration de MOD16 sur le même bassin. Si l'évapotranspiration implicite
tombe sous ce que deux produits indépendants mesurent (SSEBop, MOD16 ; R355), c'est la pluie du
forçage qui manque. Années entières où la station a au moins 330 jours mesurés, 2003-2024.

    python .runs/quebec/fermeture_bassins.py mont outv
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".runs/quebec")
import duckdb
import numpy as np
import pandas as pd
import xarray as xr

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
SFX = os.environ.get("JOINT_FX_SUFFIX", "-casr-brut")


def main(territoires):
    for reg in territoires:
        con = duckdb.connect(f"{DATA}/quebec/{reg}.duckdb", read_only=True)
        st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").fetchdf()
        edges = con.execute("select src, dst from edges").fetchdf()
        loc = con.execute("select area_km2_local from territorial order by node_idx").fetchdf().area_km2_local.to_numpy(float)
        obs = con.execute("select station_id, date, discharge from observations where date >= '2003-01-01' and date <= '2024-12-31'").fetchdf()
        et = con.execute("select node_idx, year(date) an, avg(etr_mm_day) e from modis_et where quality_ok and year(date) between 2003 and 2024 group by 1, 2").fetchdf()
        con.close()
        n = len(loc)
        amont = [[] for _ in range(n)]
        for s, d in zip(edges.src, edges.dst):
            amont[int(d)].append(int(s))
        def bassin(i):
            vu, pile = set(), [i]
            while pile:
                k = pile.pop()
                if k not in vu:
                    vu.add(k)
                    pile.extend(amont[k])
            return np.array(sorted(vu))
        ds = xr.open_dataset(f"{DATA}/quebec/forcing-{reg}{SFX}.nc")
        P = ds["forcing"].sel(var="P").values
        t = pd.DatetimeIndex(ds.time.values)
        ds.close()
        obs["date"] = pd.to_datetime(obs.date)
        rows = []
        for _, r in st.iterrows():
            b = bassin(int(r.node_idx))
            w = loc[b] / loc[b].sum()
            p_j = pd.Series(P[:, b] @ w, index=t)
            o = obs[obs.station_id == r.station_id].set_index("date").discharge
            if len(o) == 0:
                continue
            q_j = o * 86400 / (loc[b].sum() * 1e6) * 1000
            ans = [a for a in range(2003, 2025) if q_j[q_j.index.year == a].notna().sum() >= 330]
            if len(ans) < 3:
                continue
            pa = np.mean([p_j[p_j.index.year == a].sum() for a in ans])
            qa = np.mean([q_j[q_j.index.year == a].mean() * 365.25 for a in ans])
            ea = et[et.node_idx.isin(b) & et.an.isin(ans)].groupby("an").e.mean().mean() * 365.25
            rows.append(dict(station=r.station_id, annees=len(ans), pluie=pa, debit=qa, et_implicite=pa - qa, et_mod16=ea))
        tb = pd.DataFrame(rows)
        print(f"\n{reg} : bilan annuel par bassin jaugé, mm/an, forçage {SFX}")
        print(tb.round(0).to_string(index=False))
        print(f"  mediane : pluie {tb.pluie.median():.0f} | debit {tb.debit.median():.0f} | ET implicite {tb.et_implicite.median():.0f} | ET MOD16 {tb.et_mod16.median():.0f} | pluie manquante pour fermer avec MOD16 {(tb.et_mod16 + tb.debit - tb.pluie).median():.0f} mm/an ({((tb.et_mod16 + tb.debit) / tb.pluie - 1).median():+.0%})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mont", "outv"])
