"""La neige mesuree au sol : quelle grandeur est spatialement representative ?

Objection d'Essi, 2026-09-14 : l'epaisseur du manteau se krige mal. Elle a raison, et la
consequence porte sur la CONCEPTION de la contrainte, non sur son interet. Une station
nivale est un point soumis a l'altitude, au couvert forestier, a l'exposition et a la
redistribution par le vent ; sa masse n'est representative de rien au-dela de quelques
centaines de metres. Sa DATE de disparition, elle, est gouvernee par le forcage radiatif
et thermique, qui varie lentement dans l'espace.

Ce script mesure les deux, sur les couples de stations proches d'une meme region :
l'ecart relatif de masse au maximum hivernal, et l'ecart en jours de la date de
disparition. Si la seconde est coherente quand la premiere ne l'est pas, la contrainte
doit porter sur le calendrier de la fonte et sur la forme saisonniere, non sur la masse.
C'est le meme choix que celui deja fait pour l'evapotranspiration satellitaire, comparee
en tendance et non en valeur absolue.

    .venv/Scripts/python.exe .runs/quebec/coherence_canswe.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

from meandre.utils import paths as _p

REGS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cndb", "cndc"]
PAIRE_KM = 50.0
MINI_SWE = 50.0


def haversine(la1, lo1, la2, lo2):
    r = np.radians
    d = 2 * 6371.0 * np.arcsin(np.sqrt(
        np.sin((r(la2) - r(la1)) / 2) ** 2
        + np.cos(r(la1)) * np.cos(r(la2)) * np.sin((r(lo2) - r(lo1)) / 2) ** 2))
    return d


def saison(d):
    """Annee de fonte : un hiver va d'aout a juillet."""
    return d.dt.year + (d.dt.month >= 8).astype(int)


def main():
    lig = []
    for reg in REGS:
        f = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
        try:
            c = duckdb.connect(f, read_only=True)
            ob = c.execute("select swe_station_id, date, swe_mm from snow_obs "
                           "where quality_ok and swe_mm is not null").fetchdf()
            si = c.execute("select swe_station_id, lat, lon, elevation_m from snow_sites").fetchdf()
            c.close()
        except Exception:
            continue
        if ob.empty or len(si) < 2:
            continue
        ob["date"] = pd.to_datetime(ob.date)
        ob["an"] = saison(ob.date)
        si = si.set_index("swe_station_id")
        ids = [i for i in si.index if i in set(ob.swe_station_id)]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = ids[a], ids[b]
                dk = float(haversine(si.lat[ia], si.lon[ia], si.lat[ib], si.lon[ib]))
                if dk > PAIRE_KM:
                    continue
                oa, obb = ob[ob.swe_station_id == ia], ob[ob.swe_station_id == ib]
                for an in sorted(set(oa.an) & set(obb.an)):
                    ga, gb = oa[oa.an == an], obb[obb.an == an]
                    if len(ga) < 5 or len(gb) < 5:
                        continue
                    ma, mb = ga.swe_mm.max(), gb.swe_mm.max()
                    if min(ma, mb) < MINI_SWE:
                        continue
                    # date de disparition : dernier jour observe au-dessus de 10 mm
                    fa = ga[ga.swe_mm > 10.0].date.max()
                    fb = gb[gb.swe_mm > 10.0].date.max()
                    if pd.isna(fa) or pd.isna(fb):
                        continue
                    lig.append(dict(region=reg, dist_km=dk, an=an,
                                    ecart_masse=abs(np.log(ma / mb)),
                                    ecart_jours=abs((fa - fb).days),
                                    d_elev=abs(float(si.elevation_m[ia] - si.elevation_m[ib]))))
    d = pd.DataFrame(lig)
    if d.empty:
        print("aucun couple de stations nivales exploitable")
        return 1
    d.to_csv(".reports/quebec/caches/coherence-canswe.csv", index=False)
    print(f"{len(d)} couples station-hiver, distance jusqu'à {PAIRE_KM:.0f} km, "
          f"{d.region.nunique()} régions")
    print(f"  écart d'altitude médian entre stations d'un couple : {d.d_elev.median():.0f} m\n")
    print(f"  {'distance':>14s} {'couples':>8s} {'écart de masse':>18s} {'écart de date':>16s}")
    for lo, hi in ((0, 10), (10, 25), (25, 50)):
        g = d[(d.dist_km >= lo) & (d.dist_km < hi)]
        if len(g) < 10:
            continue
        print(f"  {lo:5.0f} à {hi:3.0f} km {len(g):8d} "
              f"{100 * np.median(np.exp(g.ecart_masse) - 1):15.0f} % "
              f"{np.median(g.ecart_jours):13.0f} j")
    print(f"\n  ensemble : masse {100 * np.median(np.exp(d.ecart_masse) - 1):.0f} % d'écart médian, "
          f"date {np.median(d.ecart_jours):.0f} jours")
    print(f"  couples dont la masse diffère de plus de 30 % : {100 * np.mean(np.exp(d.ecart_masse) - 1 > 0.3):.0f} %")
    print(f"  couples dont la date diffère de plus de 10 jours : {100 * np.mean(d.ecart_jours > 10):.0f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
