"""Calendrier phénologique des cultures, ajusté sans simulation sur MOD16 : sous un même climat, le
rapport de l'évapotranspiration MOD16 des tronçons agricoles (plus de 60 % de cultures) à celle des
tronçons forestiers (moins de 10 % de cultures, plus de 70 % de forêt) suit le rapport des deux
formes saisonnières, à un facteur de plateau près. La forme forestière est celle de la recette
(levée 56,81 degrés-jours base 10, largeur 21,1) ; on ajuste la levée et la largeur des cultures et
le facteur de plateau, de mars au solstice (avant la sénescence), par territoire et ensemble.

    .venv/Scripts/python.exe experiments/plan_ml/calendrier-agricole.py
"""
import duckdb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares

FORET = (56.81, 21.1)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def donnees(reg):
    raw = pd.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")
    r = raw[raw.region == reg].reset_index(drop=True)
    agri = np.flatnonzero(r.f_agriculture > 0.6)[:400]
    foret = np.flatnonzero((r.f_agriculture < 0.1) & (r.f_forest > 0.7))[:400]
    con = duckdb.connect(f"D:/meandre-data/quebec/{reg}.duckdb", read_only=True)
    et = {}
    for nom, idx in (("agri", agri), ("foret", foret)):
        m = con.sql(f"select date, avg(etr_mm_day) as et from modis_et where node_idx in ({','.join(map(str, idx))}) and date >= '2011-01-01' group by date").df()
        m["date"] = pd.to_datetime(m["date"])
        et[nom] = m.set_index("date").et.sort_index()
    con.close()
    f = xr.open_dataset(f"D:/meandre-data/quebec/forcing-{reg}-casr-brut.nc")
    t = pd.DatetimeIndex(f.time.values)
    tm = f.forcing.isel(node=np.concatenate([agri, foret])).sel(var=["Tmin", "Tmax"]).values.mean(axis=2).mean(axis=1)
    gdd = pd.Series(np.clip(tm - 10, 0, None), index=t).groupby(t.year).cumsum()
    d = pd.DataFrame({"agri": et["agri"], "foret": et["foret"]}).dropna()
    d = d[(d.index.dayofyear >= 60) & (d.index.dayofyear <= 172)]
    # MOD16 en composites de 8 jours : degrés-jours moyens du composite
    d["gdd"] = [gdd.loc[x:x + pd.Timedelta(days=7)].mean() for x in d.index]
    d["rapport"] = d.agri / d.foret
    return d.dropna()


def ajuste(d):
    def res(p):
        e, s, c = p
        forme_agri = sig((d.gdd - e) / s)
        forme_foret = sig((d.gdd - FORET[0]) / FORET[1])
        # sol nu : fond commun (k_c_min) ; rapport des demandes ≈ (fond + c·forme_agri) / (fond + forme_foret)
        fond = 0.3
        return (fond + c * forme_agri) / (fond + forme_foret) - d.rapport
    sol = least_squares(res, x0=[150, 50, 1.1], bounds=([40, 10, 0.5], [500, 200, 2.0]))
    return sol.x, float(np.sqrt(np.mean(sol.fun ** 2)))


def main():
    tous = []
    for reg in ("mont", "slso"):
        d = donnees(reg)
        tous.append(d)
        (e, s, c), rmse = ajuste(d)
        print(f"{reg} : levée {e:.0f} degrés-jours, largeur {s:.0f}, facteur de plateau {c:.2f} ; écart quadratique {rmse:.3f} sur {len(d)} composites")
    d = pd.concat(tous)
    (e, s, c), rmse = ajuste(d)
    print(f"ensemble : levée {e:.0f}, largeur {s:.0f}, facteur {c:.2f} ; écart quadratique {rmse:.3f}")
    # validation par année : ajusté sans l'année, éprouvé sur elle
    err = []
    for an in sorted(d.index.year.unique()):
        p, _ = ajuste(d[d.index.year != an])
        t = d[d.index.year == an]
        pred = (0.3 + p[2] * sig((t.gdd - p[0]) / p[1])) / (0.3 + sig((t.gdd - FORET[0]) / FORET[1]))
        err.append(float(np.sqrt(np.mean((pred - t.rapport) ** 2))))
    print(f"années retirées : écart quadratique médian {np.median(err):.3f} (rapport moyen observé {d.rapport.mean():.2f})")


if __name__ == "__main__":
    main()
