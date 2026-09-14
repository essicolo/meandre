"""La physique de meandre predit-elle bien la periode de fonte ? Question d'Essi.

Un indicateur statistique construit sur CaSR et PHYSITEL predit la MASSE du manteau a 0,77
de variance sur des stations jamais vues, mais la DUREE de la fonte a 0,09 seulement. Reste
a savoir ce que la physique du modele en fait : si elle predit deja correctement le
calendrier, l'echec de la regression ne coute rien ; si elle echoue aussi, le calendrier de
la crue reste sans contrainte, et c'est lui qui porte 59 pour cent de la variance annuelle.

On simule un sous-bassin jauge contenant des stations nivales, sans entrainement, et on
compare aux releves CanSWE, aux memes noeuds et aux memes dates. Trois grandeurs :
  la masse au maximum, ce que le modele accumule ;
  la date du maximum, quand il cesse d'accumuler ;
  la duree de la fonte, du maximum a la disparition.

    .venv/Scripts/python.exe .runs/quebec/fonte_simulee.py sagu 062101
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def carac(v, t, seuil_pic=50.0, seuil_fin=10.0):
    """Maximum, date du maximum, duree jusqu'a la disparition."""
    v = np.asarray(v, dtype=float)
    ip = int(np.nanargmax(v))
    if not np.isfinite(v[ip]) or v[ip] < seuil_pic:
        return None
    ap, at = v[ip:], t[ip:]
    f = np.flatnonzero(ap <= seuil_fin)
    if not len(f) or f[0] < 1:
        return None
    d = int((at[f[0]] - at[0]).astype("timedelta64[D]").astype(int))
    if not (5 <= d <= 150):
        return None
    return dict(pic=float(v[ip]), jour_pic=pd.Timestamp(at[ip - ip]).dayofyear, duree=d,
                date_pic=pd.Timestamp(at[0]))


def main(reg, station):
    import torch
    import banc_sousbassin as banc
    from meandre.model import HydroModel

    os.environ.setdefault("JOINT_FX_SUFFIX", "-budyko")
    os.environ.setdefault("ETL_SEED", "1234")
    _DG = {}
    _orig = HydroModel.simulate

    def _capte(self, *a, **kw):
        kw["return_diagnostics"] = True
        out = _orig(self, *a, **kw)
        _DG["dg"] = out[2]
        return out

    HydroModel.simulate = _capte
    try:
        temps, q, o = banc.simuler(reg, station, melt_saison=0.5, sol="sauf_ks", annees=6)
    finally:
        HydroModel.simulate = _orig
    s = banc.extraire(reg, station)
    idx = list(s["idx"])
    swe = _DG["dg"].swe.detach().cpu().numpy()

    c = duckdb.connect(f"D:/meandre-data/quebec/{reg}.duckdb", read_only=True)
    sn = c.execute("select swe_station_id,node_idx from snow_sites").fetchdf()
    ob = c.execute("select swe_station_id,date,swe_mm from snow_obs "
                   "where quality_ok and swe_mm is not null").fetchdf()
    c.close()
    ob["date"] = pd.to_datetime(ob.date)
    ob["an"] = ob.date.dt.year + (ob.date.dt.month >= 8).astype(int)
    pos = {n: k for k, n in enumerate(idx)}
    tt = np.asarray(temps.values, dtype="datetime64[D]")
    an_sim = temps.year.values + (temps.month.values >= 8).astype(int)

    lig = []
    for r in sn.itertuples():
        nd = int(r.node_idx)
        if nd not in pos:
            continue
        g0 = ob[ob.swe_station_id == r.swe_station_id]
        for an in sorted(set(g0.an) & set(np.unique(an_sim))):
            g = g0[g0.an == an].sort_values("date")
            if len(g) < 4:
                continue
            co = carac(g.swe_mm.values, np.asarray(g.date.values, dtype="datetime64[D]"))
            m = an_sim == an
            if co is None or m.sum() < 200:
                continue
            cs = carac(swe[m, pos[nd]], tt[m])
            if cs is None:
                continue
            lig.append(dict(station=str(r.swe_station_id), an=int(an),
                            pic_obs=co["pic"], pic_sim=cs["pic"],
                            duree_obs=co["duree"], duree_sim=cs["duree"],
                            jour_pic_obs=co["date_pic"].dayofyear,
                            jour_pic_sim=cs["date_pic"].dayofyear))
    d = pd.DataFrame(lig)
    if d.empty:
        print("aucun couple site-hiver comparable")
        return 1
    d.to_csv(f".reports/quebec/caches/fonte-simulee-{reg}-{station}.csv", index=False)
    print(f"\n{reg.upper()} / {station} : {len(d)} couples site-hiver, {d.station.nunique()} sites nivaux")
    print(f"  masse au maximum   observé {d.pic_obs.median():6.0f} mm | simulé {d.pic_sim.median():6.0f} mm"
          f" | rapport médian {np.median(d.pic_sim / d.pic_obs):.2f}")
    print(f"  date du maximum    observé jour {d.jour_pic_obs.median():3.0f} | simulé {d.jour_pic_sim.median():3.0f}"
          f" | écart médian {np.median(d.jour_pic_sim - d.jour_pic_obs):+.0f} j")
    print(f"  durée de la fonte  observé {d.duree_obs.median():5.0f} j | simulé {d.duree_sim.median():5.0f} j"
          f" | écart médian {np.median(d.duree_sim - d.duree_obs):+.0f} j")
    print(f"  écart absolu médian sur la durée : {np.median(np.abs(d.duree_sim - d.duree_obs)):.0f} j")
    for nom, a, b in (("masse", d.pic_obs, d.pic_sim), ("durée", d.duree_obs, d.duree_sim),
                      ("date du maximum", d.jour_pic_obs, d.jour_pic_sim)):
        if a.std() > 0 and b.std() > 0:
            print(f"  corrélation observé-simulé, {nom:16s} {np.corrcoef(a, b)[0, 1]:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu",
                  sys.argv[2] if len(sys.argv) > 2 else "062101"))
