"""Un indicateur de manteau nival predit sur tout le territoire ? Idee d'Essi, 2026-09-14.

Le reseau nival ne couvre que cinq regions sur neuf, et les deux ou le modele est le plus
faible n'ont aucun site. L'idee est d'apprendre la dynamique du manteau a partir de ce que
l'on possede partout, le forcage CaSR et les attributs territoriaux de PHYSITEL, sur les
stations disponibles, puis d'appliquer la relation au territoire entier. La contrainte
d'entrainement cesserait alors d'etre ponctuelle.

LA VALIDATION DECIDE DE TOUT. Predire un hiver jamais vu sur une station connue n'apprend
rien : l'usage est de predire une station jamais vue. La validation est donc par STATION
retiree, groupe par groupe, jamais par hiver.

Deux cibles, celles qu'Essi designe :
  la DUREE de la fonte, du maximum a la disparition, qui gouverne le calendrier de la crue ;
  la MASSE au maximum, qui gouverne son volume.
La duree est normalisee par construction ; la masse ne l'est pas et porte les 21 pour cent
d'ecart de representativite mesures entre stations voisines.

Une reserve de fond a garder en tete : un indicateur construit sur CaSR et PHYSITEL, puis
impose a un modele qui consomme CaSR et PHYSITEL, apprend au modele a reproduire une
regression sur ses propres entrees. Ce qu'il apporte est la relation EMPIRIQUE observee aux
stations, que le schema degre-jour n'encode pas ; ce qu'il ne peut pas apporter est une
information que les entrees ne contiennent pas.

    .venv/Scripts/python.exe .runs/quebec/indicateur_manteau.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from meandre.utils import paths as _p

REGS = ["outv", "gasp", "sagu", "slno", "abit", "cndb", "cndc"]
MINI_OBS = 3
SORTIE = ".reports/quebec/caches/indicateur-manteau.csv"


def cibles(g):
    """Duree de la fonte en jours et masse au maximum, pour un hiver d'une station."""
    g = g.sort_values("date")
    v, t = g.swe_mm.values, g.date.values
    ip = int(v.argmax())
    if v[ip] < 50:
        return None
    ap, at = v[ip:], t[ip:]
    f = np.flatnonzero(ap <= 10.0)
    if not len(f) or f[0] < 1:
        return None
    d = int((at[f[0]] - at[0]).astype("timedelta64[D]").astype(int))
    if not (5 <= d <= 150):
        return None
    return dict(duree=d, masse=float(v[ip]), n_obs=int(f[0]) + 1,
                debut=pd.Timestamp(at[0]))


def predicteurs(P, Tmin, Tmax, Rn, u2, dts, noeud, debut, an):
    """Ce que l'on possede partout : le forcage du noeud et la saison."""
    t = dts
    hiv = (t >= pd.Timestamp(f"{an - 1}-10-01")) & (t <= pd.Timestamp(f"{an}-07-31"))
    if hiv.sum() < 200:
        return None
    tm = (Tmin[hiv, noeud] + Tmax[hiv, noeud]) / 2.0
    pr = P[hiv, noeud]
    neige = pr[tm < -1.0].sum()
    pluie_hiv = pr[tm >= -1.0].sum()
    # fenetre de fonte : les soixante jours suivant le maximum observe
    fon = (t >= debut) & (t <= debut + pd.Timedelta(days=60))
    tmf = (Tmin[fon, noeud] + Tmax[fon, noeud]) / 2.0
    dj = np.clip(tmf, 0.0, None).sum()
    amp = (Tmax[fon, noeud] - Tmin[fon, noeud]).mean()
    return dict(neige_hiver=float(neige), pluie_hiver=float(pluie_hiv),
                t_moy_hiver=float(tm.mean()), t_min_hiver=float(tm.min()),
                degres_jour_fonte=float(dj),
                t_moy_fonte=float(tmf.mean()), t_max_fonte=float(tmf.max()),
                pluie_fonte=float(P[fon, noeud].sum()),
                amplitude_diurne_fonte=float(amp),
                rad_fonte=float(Rn[fon, noeud].mean()),
                rad_max_fonte=float(Rn[fon, noeud].max()),
                vent_fonte=float(u2[fon, noeud].mean()),
                rad_hiver=float(Rn[hiv, noeud].mean()),
                jour_debut=float(debut.dayofyear))


def main():
    lig = []
    for reg in REGS:
        try:
            c = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
            ob = c.execute("select swe_station_id,date,swe_mm from snow_obs "
                           "where quality_ok and swe_mm is not null").fetchdf()
            si = c.execute("select swe_station_id,node_idx,lat,lon,elevation_m from snow_sites").fetchdf()
            te = c.execute("select node_idx,mean_slope_pct,mean_elevation_m,sin_aspect,"
                           "cos_aspect,f_forest,f_wetland,f_water from territorial").fetchdf()
            c.close()
        except Exception as e:
            print(f"{reg} : base indisponible ({type(e).__name__})")
            continue
        if ob.empty or si.empty:
            continue
        try:
            ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
        except FileNotFoundError:
            print(f"{reg} : forçage absent")
            continue
        dts = pd.DatetimeIndex(ds["time"].values)
        F = ds["forcing"].values
        ds.close()
        P, Tmin, Tmax, Rn, u2 = F[..., 0], F[..., 1], F[..., 2], F[..., 3], F[..., 4]
        te = te.set_index("node_idx")
        # OCCUPATION DU SOL REELLE. Les colonnes territoriales de la base sont centrees et
        # reduites PAR REGION : leur fraction forestiere va de -3,67 a 1,27 et ne se compare
        # pas d'une region a l'autre. Le couvert vrai, qui separe resineux, feuillus et
        # decouvert, est dans les fichiers de la plateforme Hydrotel.
        si = si.set_index("swe_station_id")
        ob["date"] = pd.to_datetime(ob.date)
        ob["an"] = ob.date.dt.year + (ob.date.dt.month >= 8).astype(int)
        for (s, an), g in ob.groupby(["swe_station_id", "an"]):
            if s not in si.index:
                continue
            cib = cibles(g)
            if cib is None or cib["n_obs"] < MINI_OBS:
                continue
            nd = int(si.node_idx[s])
            if nd not in te.index or nd >= P.shape[1]:
                continue
            pr = predicteurs(P, Tmin, Tmax, Rn, u2, dts, nd, cib["debut"], an)
            if pr is None:
                continue
            r = dict(region=reg, station=str(s), an=int(an))
            r.update(cib)
            r.pop("debut")
            r.update(pr)
            r["altitude"] = float(si.elevation_m[s])
            r["latitude"] = float(si.lat[s])
            r["longitude"] = float(si.lon[s])
            for k in ("mean_slope_pct", "mean_elevation_m", "sin_aspect", "cos_aspect",
                      "f_forest", "f_wetland", "f_water"):
                r[k] = float(te[k].loc[nd])
            lig.append(r)
    d = pd.DataFrame(lig)
    if d.empty:
        print("aucun couple station-hiver exploitable")
        return 1
    d.to_csv(SORTIE, index=False)
    print(f"{len(d)} couples station-hiver, {d.station.nunique()} stations, {d.region.nunique()} régions\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.dummy import DummyRegressor

    PRED = [c for c in d.columns if c not in
            ("region", "station", "an", "duree", "masse", "n_obs")]
    X = d[PRED].values
    grp = d.station.values
    cv = GroupKFold(n_splits=min(8, d.station.nunique()))
    for cible, nom in (("duree", "durée de la fonte (jours)"), ("masse", "masse au maximum (mm)")):
        y = d[cible].values
        for mod, etiq in ((HistGradientBoostingRegressor(max_iter=300, random_state=0), "gradient boosté"),
                          (DummyRegressor(strategy="mean"), "moyenne, témoin")):
            p = cross_val_predict(mod, X, y, cv=cv, groups=grp)
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            mae = float(np.mean(np.abs(y - p)))
            print(f"  {nom:30s} {etiq:18s} R2 par station retirée {r2:+.3f} | "
                  f"erreur absolue moyenne {mae:.1f}")
        print()
    print("  Effet du nombre de relevés sur la durée : une cible mal mesurée ne se prédit pas.")
    for nmin in (3, 5, 8):
        g = d[d.n_obs >= nmin]
        if g.station.nunique() < 10:
            continue
        Xg, yg, gg = g[PRED].values, g.duree.values, g.station.values
        cvg = GroupKFold(n_splits=min(8, g.station.nunique()))
        pg = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, random_state=0),
                               Xg, yg, cv=cvg, groups=gg)
        r2 = 1 - ((yg - pg) ** 2).sum() / ((yg - yg.mean()) ** 2).sum()
        print(f"    au moins {nmin} relevés : {len(g):5d} hivers, {g.station.nunique():3d} stations, "
              f"R2 {r2:+.3f}, erreur {np.mean(np.abs(yg - pg)):.1f} j, "
              f"écart-type de la cible {yg.std():.1f} j")
    print()
    print("  lecture : le R2 est calculé en retirant des STATIONS entières, pas des hivers.")
    print("  Un R2 négatif signifie que la relation ne transfère pas à une station inconnue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
