"""D'ou vient l'erreur d'amplitude : de la pluie d'entree ou du modele ?

L'erreur d'amplitude par evenement vaut 0,57 en mediane, et a ce niveau tout objectif
ponctuel rabote les pointes. Deux leviers se disputent la cause : la precipitation
d'entree, CaSR corrige, et la physique du modele. Une correction de la moyenne ne peut
pas reduire une dispersion ; seule une information propre a chaque evenement le peut.
Il faut donc savoir si les evenements ou le pic simule se trompe sont ceux ou la pluie
d'entree se trompe.

Pour chaque pointe observee au-dessus du quantile 0,90 d'une station hydrometrique, on
compare, sur les quatre jours qui la precedent, le cumul de precipitation de CaSR corrige
moyenne sur le bassin amont au cumul moyen des stations meteorologiques d'Environnement
Canada situees dans ce bassin. Le logarithme de leur rapport est l'erreur de pluie de
l'evenement ; celui du rapport des pics est l'erreur de debit. Si les deux vont ensemble,
le levier est la pluie et le krigeage sur stations est la voie. Sinon, c'est le modele.

Les stations meteorologiques sont des points et le bassin demande une pluie de surface :
leur moyenne porte sa propre erreur, si bien que la part de l'erreur de debit expliquee
par la pluie est une borne INFERIEURE.

    .venv/Scripts/python.exe .runs/quebec/decompose_amplitude.py gasp
"""
import sys
from collections import defaultdict, deque

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from meandre.utils import paths as _p

DOS = f"{_p.DATA_ROOT}/quebec/flotte"
# L'emprise des stations meteorologiques se DEDUIT du reseau de la region, avec une marge.
# Une emprise devinee a la main avait rate la Gaspesie entiere le 2026-09-14, les stations
# hydrometriques se trouvant a l'ouest de la fenetre supposee.
MARGE = 0.3
FENETRE = 4
MINI_PLUIE = 5.0
# Une jauge ponctuelle porte une grande erreur de representativite pour une pluie de
# surface, et cette erreur DILUE la correlation vers zero. Exiger plusieurs jauges par
# bassin reduit la dilution ; le biais, lui, y est bien moins sensible que la dispersion.
MINI_JAUGES = 3


def amont(edges, depart):
    pere = defaultdict(list)
    for s, t in edges:
        pere[int(t)].append(int(s))
    vus, q = {int(depart)}, deque([int(depart)])
    while q:
        u = q.popleft()
        for v in pere[u]:
            if v not in vus:
                vus.add(v)
                q.append(v)
    return sorted(vus)


def evenements(o, seuil, ecart=7):
    idx = [i for i in range(1, len(o) - 1)
           if o[i] > seuil and o[i] >= o[i - 1] and o[i] >= o[i + 1]]
    garde = []
    for i in idx:
        if not garde or i - garde[-1] >= ecart:
            garde.append(i)
        elif o[i] > o[garde[-1]]:
            garde[-1] = i
    return garde


def main(reg):
    z = np.load(f"{DOS}/q-{reg}-A-v4.npz", allow_pickle=True)
    dts = pd.to_datetime([str(v)[:10] for v in z["dates"]])
    sids = [str(x) for x in z["station_ids"]]
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    st = con.execute("select station_id, node_idx, lon, lat from stations").fetchdf()
    ed = con.execute("select * from edges").fetchdf()
    nd = con.execute("select * from nodes").fetchdf()
    con.close()
    st["station_id"] = st.station_id.astype(str)
    cs, ct = ed.columns[0], ed.columns[1]
    edges = list(zip(ed[cs].astype(int), ed[ct].astype(int)))
    lon_c = "lon" if "lon" in nd.columns else nd.columns[1]
    lat_c = "lat" if "lat" in nd.columns else nd.columns[2]

    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
    tf = pd.DatetimeIndex(ds["time"].values)
    sel = np.isin(tf, dts)
    P = ds["forcing"].isel(forcing=0).values[sel] if "forcing" in ds["forcing"].dims else ds["forcing"].values[sel][..., 0]
    ds.close()
    assert P.shape[0] == len(dts), (P.shape, len(dts))

    from meandre.data.eccc_loader import fetch_daily
    bbox = (float(nd[lon_c].min()) - MARGE, float(nd[lat_c].min()) - MARGE,
            float(nd[lon_c].max()) + MARGE, float(nd[lat_c].max()) + MARGE)
    ec = fetch_daily(tuple(round(v, 1) for v in bbox), dts.year.min(), dts.year.max())
    ec["date"] = pd.to_datetime(ec.date)
    ec = ec[ec.date.isin(dts) & ec.P.notna()]
    pg = ec.pivot_table(index="date", columns="climate_id", values="P").reindex(dts)
    coords = ec.groupby("climate_id")[["lon", "lat"]].first()

    lig = []
    for j, sid in enumerate(sids):
        r = st[st.station_id == sid]
        if r.empty:
            continue
        noeuds = amont(edges, int(r.node_idx.iloc[0]))
        if len(noeuds) < 3:
            continue
        lo0, lo1 = nd[lon_c].iloc[noeuds].min() - 0.1, nd[lon_c].iloc[noeuds].max() + 0.1
        la0, la1 = nd[lat_c].iloc[noeuds].min() - 0.1, nd[lat_c].iloc[noeuds].max() + 0.1
        ids = coords[(coords.lon.between(lo0, lo1)) & (coords.lat.between(la0, la1))].index
        ids = [i for i in ids if i in pg.columns and pg[i].notna().mean() > 0.8]
        if len(ids) < MINI_JAUGES:
            continue
        p_bassin = P[:, noeuds].mean(axis=1)
        p_jauges = pg[ids].mean(axis=1).to_numpy()
        o, s = z["q_obs"][:, j].astype(float), z["q_sim"][:, j].astype(float)
        m = np.isfinite(o) & np.isfinite(s)
        if m.sum() < 700:
            continue
        oo = np.where(m, o, -1.0)
        for i in evenements(oo, np.nanquantile(np.where(m, o, np.nan), 0.90)):
            if i < FENETRE:
                continue
            w = slice(i - FENETRE + 1, i + 1)
            pj, pb = np.nansum(p_jauges[w]), float(p_bassin[w].sum())
            if not np.isfinite(pj) or pj < MINI_PLUIE or pb <= 0:
                continue
            lo, hi = max(0, i - 2), min(len(s), i + 3)
            if not np.isfinite(s[lo:hi]).any() or o[i] <= 0:
                continue
            lig.append(dict(station=sid, n_jauges=len(ids), mois=int(dts[i].month),
                            err_pluie=float(np.log(pb / pj)),
                            err_debit=float(np.log(np.nanmax(s[lo:hi]) / o[i]))))
    d = pd.DataFrame(lig)
    if d.empty:
        print("aucun evenement apparie")
        return
    d.to_csv(f".reports/quebec/caches/decompose-amplitude-{reg}.csv", index=False)
    print(f"{reg.upper()} : {len(d)} evenements sur {d.station.nunique()} stations, "
          f"{d.n_jauges.median():.0f} jauges par bassin en mediane")

    def bloc(g, nom):
        if len(g) < 20:
            return
        rho = np.corrcoef(g.err_pluie, g.err_debit)[0, 1]
        pente = np.polyfit(g.err_pluie, g.err_debit, 1)[0]
        print(f"  {nom:18s} n={len(g):4d} | erreur de pluie : biais {g.err_pluie.mean():+.2f}, "
              f"dispersion {g.err_pluie.std():.2f} | erreur de debit : biais {g.err_debit.mean():+.2f}, "
              f"dispersion {g.err_debit.std():.2f} | corrélation {rho:+.2f}, pente {pente:+.2f}, "
              f"part expliquée {100 * rho ** 2:.0f} %")

    bloc(d, "toutes saisons")
    bloc(d[d.mois.isin((4, 5))], "crue d'avril-mai")
    bloc(d[~d.mois.isin((4, 5))], "autres saisons")
    print("  lecture : la part expliquee est une borne inferieure de la part du forcage, les "
          "jauges portant leur propre erreur de representativite.")


if __name__ == "__main__":
    for _r in (sys.argv[1:] or ["gasp"]):
        try:
            main(_r)
        except Exception as _e:
            print(f"{_r.upper()} : ECHEC {type(_e).__name__} {_e}")
        print()
