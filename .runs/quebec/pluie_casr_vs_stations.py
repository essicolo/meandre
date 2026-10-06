"""La pluie d'été de CaSR est-elle trop forte sur les sous-bassins du banc ? Comparaison aux stations GHCN.

Sur le Saint-Laurent nord-ouest 052805, le débit d'août simulé vaut deux fois l'observé sous toutes
les recettes, et le bilan annuel demande plus d'évapotranspiration que MOD16 n'en donne : un surplus
de pluie d'été du forçage est le premier suspect. Pour chaque sous-bassin, on prend les stations GHCN
à moins de 0,3 degré de son emprise ayant au moins 25 jours mesurés dans le mois, et on compare la
précipitation mensuelle de la station à celle du nœud CaSR le plus proche, 2005 à 2013. Rapport
CaSR sur station par mois, médiane des couples station-mois.

    python .runs/quebec/pluie_casr_vs_stations.py outv 040110 mont 030905 slno 052805
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xarray as xr

from meandre.utils import paths as _p

# Fenetre de comparaison ; MEANDRE_PLUIE_ANNEES="2010,2024" la change (2026-10-01).
A0, A1 = (int(x) for x in os.environ.get("MEANDRE_PLUIE_ANNEES", "2005,2013").split(","))


def stations_ghcn():
    cols = [(0, 11), (12, 20), (21, 30)]
    st = pd.read_fwf(f"{_p.DATA_ROOT}/ghcn/ghcnd-stations.txt", colspecs=cols, header=None, names=["id", "lat", "lon"])
    dispo = {os.path.basename(f)[:-4] for f in glob.glob(f"{_p.DATA_ROOT}/ghcn/access/*.csv")}
    return st[st.id.isin(dispo)]


def main(paires):
    from banc_sousbassin import extraire
    st = stations_ghcn()
    for reg, station in paires:
        s = extraire(reg, station)
        c = s["node_coords"].cpu().numpy()
        idx = np.asarray(s["idx"])
        _mg = float(os.environ.get("MEANDRE_PLUIE_MARGE", "0.3"))   # marge autour de l'emprise, degres
        lon0, lon1 = c[:, 0].min() - _mg, c[:, 0].max() + _mg
        lat0, lat1 = c[:, 1].min() - _mg, c[:, 1].max() + _mg
        proches = st[(st.lon >= lon0) & (st.lon <= lon1) & (st.lat >= lat0) & (st.lat <= lat1)]
        # Le forcage compare suit JOINT_FX_SUFFIX (defaut -budyko, celui de la mesure du 1er octobre).
        ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}{os.environ.get('JOINT_FX_SUFFIX', '-budyko')}.nc")
        temps = pd.DatetimeIndex(ds.time.values)
        P = ds["forcing"].sel(var="P").isel(node=idx)
        couples = []
        n_st = 0
        for _, r in proches.iterrows():
            g = pd.read_csv(f"{_p.DATA_ROOT}/ghcn/access/{r.id}.csv", usecols=lambda c: c in ("DATE", "PRCP"), parse_dates=["DATE"])
            if "PRCP" not in g.columns:
                continue   # station sans precipitation (temperature seule)
            g = g[(g.DATE.dt.year >= A0) & (g.DATE.dt.year <= A1)].dropna(subset=["PRCP"])
            if len(g) < 365:
                continue
            p_st = pd.Series(g.PRCP.to_numpy(dtype=float) / 10.0, index=pd.DatetimeIndex(g.DATE))
            d = np.hypot((c[:, 0] - r.lon) * np.cos(np.deg2rad(r.lat)), c[:, 1] - r.lat)
            j = int(np.argmin(d))
            p_cs = pd.Series(P.isel(node=j).values, index=temps).reindex(p_st.index)
            df = pd.DataFrame({"st": p_st, "cs": p_cs}).dropna()
            m = df.groupby([df.index.year, df.index.month]).agg(st=("st", "sum"), cs=("cs", "sum"), n=("st", "size"))
            m = m[m.n >= 25]
            for (an, mois), row in m.iterrows():
                couples.append({"station": r.id, "annee": an, "mois": mois, "st": row.st, "cs": row.cs})
            n_st += 1
        ds.close()
        t = pd.DataFrame(couples)
        if t.empty:
            print(f"{reg} {station} : aucune station GHCN utilisable dans l'emprise")
            continue
        t = t[t.st > 5.0]
        t["rapport"] = t.cs / t.st
        par_mois = t.groupby("mois").agg(rapport=("rapport", "median"), st=("st", "mean"), cs=("cs", "mean"), n=("rapport", "size"))
        print(f"{reg} {station} : {n_st} stations GHCN, {len(t)} couples station-mois, {A0}-{A1}")
        print("  rapport CaSR / station par mois (mediane), pluie mensuelle moyenne station et CaSR en mm :")
        print("  " + " ".join(f"{int(m):2d}:{r.rapport:.2f}({r.st:.0f}/{r.cs:.0f})" for m, r in par_mois.iterrows()))
        ete = t[t.mois.isin((6, 7, 8, 9))]; hiv = t[t.mois.isin((12, 1, 2, 3))]
        print(f"  juin-septembre {ete.rapport.median():.2f} | decembre-mars {hiv.rapport.median():.2f} | annee {t.rapport.median():.2f}")
        # Par annee : sommes CaSR et station sur les memes couples station-mois, annee entiere et
        # juin-octobre. Dit si le surplus de pluie de CaSR suit les annees ou le debit simule deborde.
        an_t = t.groupby("annee").agg(st=("st", "sum"), cs=("cs", "sum"))
        ete_t = t[t.mois.isin((6, 7, 8, 9, 10))].groupby("annee").agg(st_ete=("st", "sum"), cs_ete=("cs", "sum"))
        par_an = an_t.join(ete_t)
        par_an["CaSR/station annee"] = par_an.cs / par_an.st
        par_an["CaSR/station juin-oct"] = par_an.cs_ete / par_an.st_ete
        print("  par annee, memes couples station-mois :")
        print(par_an[["CaSR/station annee", "CaSR/station juin-oct"]].round(2).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(list(zip(a[::2], a[1::2])))
