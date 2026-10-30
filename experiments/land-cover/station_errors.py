"""Erreurs du modèle gelé par station contre l'occupation du sol du bassin (PHYSITEL et WorldCover).

Erreurs mesurées sur 2019-2024 (dump des stations) : excès aux petites pluies (coefficient
d'épisode simulé sur observé, pluies de 10 à 20 mm), déficit aux grosses (55 mm et plus),
volume de juillet-août simulé sur observé, KGE. Occupation : moyenne pondérée par la surface sur
le bassin amont de chaque station.

Usage : python station_errors.py <territoire> <projet PHYSITEL> <comparaison par UHRH> <dossier des dumps>
"""
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, "C:/Users/parse01/documents-locaux/GitHub/meandre-analyse")
import data as dt
from meandre.adapters.physitel.physitel_loader import _parse_reach


def reach_cover(project, compare):
    from pathlib import Path
    tr = _parse_reach(Path(project) / "physitel" / "troncon.trl")
    m = pd.read_parquet(compare).set_index("uhrh")
    cols = ["impermeable", "wc_built", "eau", "wc_water", "tourbiere", "milieu_humide", "wc_wetland", "agricole_paturage", "wc_crop", "wc_tree"]
    rows = []
    for t in tr:
        u = [abs(int(i)) for i in t["uhrh_ids"] if abs(int(i)) in m.index]
        if not u:
            continue
        g = m.loc[u]
        w = g["pix"].to_numpy()
        rows.append({"node_id": t["id"], "area": w.sum(), **{c: float(np.average(g[c], weights=w)) for c in cols}})
    return pd.DataFrame(rows)


def main(territory, project, compare, dumps):
    cover = reach_cover(project, compare)
    st = dt.stations([territory])
    up = dt.station_upstream(st)
    import duckdb
    con = duckdb.connect(str(dt.basin_db(territory)), read_only=True)
    nodes = con.sql("select node_idx, node_id from nodes").df()
    con.close()
    cover = cover.merge(nodes, on="node_id")
    s = np.load(f"{dumps}/stations-{territory}.npz", allow_pickle=True)
    q = np.load(f"{dumps}/q-{territory}.npz", allow_pickle=True)
    S, O, P = q["q_sim"], q["q_obs"], s["basin_snowmelt"]
    d = np.array(q["dates"], dtype="datetime64[D]")
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    ids = list(s["station_ids"])
    rows = []
    for row in up.iter_rows(named=True):
        code = row["station"].split(":")[1]
        if code not in ids:
            continue
        k = ids.index(code)
        c = cover[cover.node_idx.isin(row["upstream"])]
        if c.empty:
            continue
        basin = {col: float(np.average(c[col], weights=c["area"])) for col in c.columns if col not in ("node_id", "node_idx", "area")}
        small, big = [], []
        for t in range(3, len(d) - 7):
            if not (6 <= m[t] <= 10) or P[t, k] < 10 or P[t - 3:t, k].sum() > 2 or np.isnan(O[t - 1:t + 7, k]).any():
                continue
            vo = np.clip(O[t:t + 7, k] - O[t - 1, k], 0, None).sum()
            vs = np.clip(S[t:t + 7, k] - S[t - 1, k], 0, None).sum()
            pw = P[t:t + 7, k].sum()
            if vo > 0 and vs > 0:
                (small if pw < 20 else big if pw >= 55 else []).append(np.log(vs / vo))
        ok = np.isfinite(O[:, k])
        jul_aug = ok & np.isin(m, (7, 8))
        sim, obs = S[ok, k], O[ok, k]
        r = np.corrcoef(sim, obs)[0, 1]
        kge = 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)
        rows.append({"station": code, **basin, "small_rain_ratio": np.exp(np.median(small)) if small else np.nan, "big_rain_ratio": np.exp(np.median(big)) if big else np.nan,
                     "summer_volume": S[jul_aug, k].sum() / O[jul_aug, k].sum(), "kge": kge})
    r = pd.DataFrame(rows)
    print(f"{territory} : {len(r)} stations")
    covers = ["impermeable", "wc_built", "eau", "wc_water", "milieu_humide", "wc_wetland", "agricole_paturage", "wc_crop", "wc_tree"]
    errors = ["small_rain_ratio", "big_rain_ratio", "summer_volume", "kge"]
    table = pd.DataFrame({e: [spearmanr(r[c], r[e], nan_policy="omit").correlation for c in covers] for e in errors}, index=covers)
    print(table.round(2).to_string())
    r.to_parquet(f"{dumps}/station-errors-{territory}.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:5])
