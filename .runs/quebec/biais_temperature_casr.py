"""Le forcage se trompe-t-il sur la TEMPERATURE, et si oui quand ?

Un collegue avance le 2026-09-15 que CaSR n'a pas vraiment de probleme de volume d'eau
mais plutot de calendrier de fonte, peut-etre par la temperature. La question se repond
sans entrainer : le depot porte 1 832 stations d'Environnement Canada avec Tmin et Tmax
journaliers sur 2016-2019, et le forcage de chaque region porte les memes grandeurs par
troncon. On apparie chaque station au troncon le plus proche et on mesure l'ecart.

Ce qui compte pour la fonte n'est pas le biais annuel mais le biais AUTOUR DE ZERO au
printemps : un degre de trop en mars avance la fonte de plusieurs jours, sans rien changer
au volume annuel d'eau. On rapporte donc le biais par mois, le biais des degres-jours
au-dessus de zero, et la date a laquelle le cumul de degres-jours franchit un seuil, qui
est l'indicateur direct du calendrier de fonte.

    .venv/Scripts/python.exe .runs/quebec/biais_temperature_casr.py sagu gasp outv
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

MOIS = ["jan", "fév", "mar", "avr", "mai", "jun", "jul", "aoû", "sep", "oct", "nov", "déc"]


def apparie(reg, dmax_km=15.0):
    """Stations d'Environnement Canada appariees au troncon le plus proche."""
    import duckdb
    base = _paths.data_path("quebec", f"{reg}.duckdb")
    cx = duckdb.connect(base, read_only=True)
    nd = cx.sql("select node_idx, lon, lat from nodes order by node_idx").fetchdf()
    cx.close()
    st = duckdb.sql(f"""select climate_id, any_value(lon) lon, any_value(lat) lat
        from read_parquet('{_paths.data_path("quebec", "eccc_daily_2016_2019.parquet")}')
        where Tmin is not null group by 1""").fetchdf()
    lat0 = float(nd.lat.mean())
    kx = 111.0 * np.cos(np.radians(lat0))
    d = np.hypot((st.lon.values[:, None] - nd.lon.values[None, :]) * kx,
                 (st.lat.values[:, None] - nd.lat.values[None, :]) * 111.0)
    j = d.argmin(1)
    dd = d[np.arange(len(st)), j]
    ok = dd <= dmax_km
    return st.climate_id.values[ok], nd.node_idx.values[j[ok]], dd[ok]


def main(regions):
    import duckdb
    obs_all = duckdb.sql(f"""select climate_id, date, Tmin, Tmax
        from read_parquet('{_paths.data_path("quebec", "eccc_daily_2016_2019.parquet")}')
        where Tmin is not null and Tmax is not null""").fetchdf()
    obs_all["date"] = pd.to_datetime(obs_all["date"]).dt.normalize()

    lignes, cal = [], []
    for reg in regions:
        f = _paths.data_path("quebec", f"forcing-{reg}-budyko.nc")
        if not os.path.exists(f):
            print(f"{reg}: forçage absent")
            continue
        ids, noeuds, dist = apparie(reg)
        if len(ids) == 0:
            print(f"{reg}: aucune station à moins de 15 km d'un tronçon")
            continue
        ds = xr.open_dataset(f)
        sim = ds.forcing.sel(var=["Tmin", "Tmax"]).sel(
            node=xr.DataArray(noeuds, dims="s")).load()
        t = pd.DatetimeIndex(ds.time.values)
        ds.close()
        for k, (cid, nj, dk) in enumerate(zip(ids, noeuds, dist)):
            o = obs_all[obs_all.climate_id == cid].set_index("date")
            if len(o) < 365:
                continue
            idx = t.intersection(o.index)
            if len(idx) < 365:
                continue
            pos = t.get_indexer(idx)
            smin = sim.values[pos, k, 0]
            smax = sim.values[pos, k, 1]
            omin = o.loc[idx, "Tmin"].values
            omax = o.loc[idx, "Tmax"].values
            m = np.isfinite(smin) & np.isfinite(omin) & np.isfinite(smax) & np.isfinite(omax)
            if m.sum() < 365:
                continue
            smoy, omoy = (smin + smax) / 2, (omin + omax) / 2
            mois = idx.month.to_numpy()
            for mo in range(1, 13):
                mm = m & (mois == mo)
                if mm.sum() < 20:
                    continue
                lignes.append({"region": reg, "station": cid, "mois": mo,
                               "d_moy": float(np.mean(smoy[mm] - omoy[mm])),
                               "d_min": float(np.mean(smin[mm] - omin[mm])),
                               "d_max": float(np.mean(smax[mm] - omax[mm])),
                               "dj_sim": float(np.mean(np.maximum(smoy[mm], 0))),
                               "dj_obs": float(np.mean(np.maximum(omoy[mm], 0)))})
            # Date de franchissement d'un cumul de 50 degres-jours, par annee.
            an = idx.year.to_numpy()
            doy = idx.dayofyear.to_numpy()
            for a in np.unique(an):
                aa = m & (an == a) & (doy <= 200)
                if aa.sum() < 150:
                    continue
                for nom, serie in (("simulé", smoy), ("observé", omoy)):
                    cum = np.cumsum(np.maximum(serie[aa], 0))
                    w = np.flatnonzero(cum >= 50)
                    if len(w):
                        cal.append({"region": reg, "station": cid, "an": int(a),
                                    "source": nom, "jour": int(doy[aa][w[0]])})

    d = pd.DataFrame(lignes)
    if d.empty:
        print("aucune station appariée")
        return 1
    print(f"\n{d.station.nunique()} stations appariées, {', '.join(regions)}, 2016-2019\n")
    print("écart du forçage à la station, en degrés Celsius, moyenne sur les stations")
    print(f"{'mois':>5} {'T moyenne':>10} {'Tmin':>8} {'Tmax':>8} "
          f"{'degrés-jours sim':>17} {'obs':>7}")
    for mo in range(1, 13):
        q = d[d.mois == mo]
        if q.empty:
            continue
        print(f"{MOIS[mo - 1]:>5} {q.d_moy.mean():+10.2f} {q.d_min.mean():+8.2f} "
              f"{q.d_max.mean():+8.2f} {q.dj_sim.mean():17.2f} {q.dj_obs.mean():7.2f}")

    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        _m = d.groupby("mois")[["d_moy", "d_min", "d_max", "dj_sim", "dj_obs"]].mean()
        _m["stations"] = d.groupby("mois").station.nunique()
        _m.to_csv(f"{sortie}/biais-temperature-casr.csv")
        print(f"cache : {sortie}/biais-temperature-casr.csv")

    c = pd.DataFrame(cal)
    if not c.empty:
        p = c.pivot_table(index=["region", "station", "an"], columns="source",
                          values="jour").dropna()
        ecart = (p["simulé"] - p["observé"]).values
        print(f"\ndate de franchissement de 50 degrés-jours cumulés depuis le 1er janvier,"
              f" sur {len(ecart)} couples station-année")
        print(f"  écart simulé moins observé : médiane {np.median(ecart):+.1f} j | "
              f"moyenne {ecart.mean():+.1f} j | q25-q75 "
              f"{np.quantile(ecart, .25):+.1f} à {np.quantile(ecart, .75):+.1f} j")
        print(f"  part des couples où le forçage est EN AVANCE : "
              f"{100 * (ecart < 0).mean():.0f} pour cent")
        if os.path.isdir(sortie):
            pd.DataFrame({"ecart_jours": ecart}).to_csv(
                f"{sortie}/calendrier-degres-jours.csv", index=False)
            print(f"  cache : {sortie}/calendrier-degres-jours.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["sagu", "gasp", "outv"]))
