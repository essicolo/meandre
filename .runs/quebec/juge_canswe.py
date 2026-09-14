"""Comparaison A POSTERIORI de la neige simulee aux releves CanSWE.

IDEE D'ESSI (2026-09-10) : ne PAS mettre la neige mesuree au sol dans la fonction de
perte, mais s'en servir comme piece de comparaison la ou c'est possible. La raison est
physique : l'equivalent en eau du manteau varie enormement selon des conditions tres
locales, exposition, couvert forestier, vent, si bien qu'un releve ponctuel ne represente
pas la moyenne d'un troncon de plusieurs dizaines de kilometres carres. En faire une
cible d'entrainement forcerait le modele a reproduire un point ; en faire un diagnostic
dit seulement si le modele est du bon ordre de grandeur et de la bonne saison.

Ce que le script compare, par site : la CLIMATOLOGIE MENSUELLE de l'equivalent en eau,
simulee au troncon de rattachement contre observee au site, et la date du maximum. Le
niveau absolu est rapporte mais n'est pas un verdict, pour la raison ci-dessus.

Entrees : les caches par troncon produits par le pilote avec ETL_DUMP_REACH (ils portent
`swe_mensuel`), et la table `snow_sites` / `snow_obs` de chaque base regionale, qui
conserve la distance de rattachement et l'ecart d'altitude.

    python .runs/quebec/juge_canswe.py                  # toutes les regions disponibles
    python .runs/quebec/juge_canswe.py outv gasp
    python .runs/quebec/juge_canswe.py --dist-max 15 --alt-max 200
"""
import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
REP = f"{DATA}/quebec/rapport"
REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]


def climato_observee(db, dist_max, alt_max):
    """Climatologie mensuelle de l'equivalent en eau par site, et son troncon."""
    c = duckdb.connect(db, read_only=True)
    tables = [x[0] for x in c.execute("show tables").fetchall()]
    if "snow_sites" not in tables or "snow_obs" not in tables:
        c.close()
        return None
    d = c.execute(f"""
        select s.swe_station_id sid, s.node_idx, s.dist_km, s.elev_diff_m,
               month(o.date) mois, avg(o.swe_mm) swe
        from snow_sites s join snow_obs o using (swe_station_id)
        where o.swe_mm is not null and o.quality_ok
          and s.dist_km <= {dist_max} and abs(s.elev_diff_m) <= {alt_max}
        group by 1, 2, 3, 4, 5""").fetchdf()
    c.close()
    return d


def juger(reg, dist_max, alt_max):
    f = f"{REP}/rap-{reg}-avec.npz"
    if not os.path.exists(f):
        return None, f"cache par troncon absent ({os.path.basename(f)})"
    z = np.load(f, allow_pickle=True)
    if "swe_mensuel" not in z.files:
        return None, "le cache ne porte pas swe_mensuel : relancer le pilote avec ETL_DUMP_REACH"
    sim = z["swe_mensuel"]                      # (12, n_noeuds)
    obs = climato_observee(f"{DATA}/quebec/{reg}.duckdb", dist_max, alt_max)
    if obs is None or not len(obs):
        return None, "aucun site de neige retenu"
    lignes = []
    for sid, g in obs.groupby("sid"):
        i = int(g.node_idx.iloc[0])
        if not (0 <= i < sim.shape[1]):
            continue
        g = g.set_index("mois").reindex(range(1, 13))
        o = g.swe.to_numpy(dtype=float)
        s = sim[:, i].astype(float)
        m = np.isfinite(o) & np.isfinite(s)
        if m.sum() < 6 or np.nanmax(o[m]) <= 0:
            continue
        lignes.append({
            "site": sid, "dist_km": float(g.dist_km.dropna().iloc[0]),
            "corr_saison": float(np.corrcoef(o[m], s[m])[0, 1]) if o[m].std() > 0 and s[m].std() > 0 else np.nan,
            "max_sim_mm": float(np.nanmax(s)), "max_obs_mm": float(np.nanmax(o)),
            "mois_max_sim": int(np.nanargmax(s) + 1), "mois_max_obs": int(np.nanargmax(np.where(m, o, np.nan)) + 1),
        })
    if not lignes:
        return None, "aucun site comparable"
    d = pd.DataFrame(lignes)
    d["rapport_max"] = d.max_sim_mm / d.max_obs_mm.clip(lower=1e-6)
    d["decalage_mois"] = ((d.mois_max_sim - d.mois_max_obs + 6) % 12) - 6
    return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*")
    ap.add_argument("--dist-max", type=float, default=25.0,
                    help="distance maximale site-troncon retenue, en km")
    ap.add_argument("--alt-max", type=float, default=300.0,
                    help="ecart d'altitude maximal retenu, en m")
    a = ap.parse_args()
    rows = []
    for reg in (a.regions or REGIONS):
        d, err = juger(reg, a.dist_max, a.alt_max)
        if d is None:
            print(f"  {reg:<6s} {err}")
            continue
        rows.append({
            "region": reg, "sites": len(d),
            "corr_saison_med": round(float(d.corr_saison.median()), 2),
            "rapport_max_med": round(float(d.rapport_max.median()), 2),
            "decalage_mois_med": int(d.decalage_mois.median()),
            "dist_med_km": round(float(d.dist_km.median()), 1),
        })
    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        print("\ncorr_saison : correlation des climatologies mensuelles simulee et observee")
        print("rapport_max : maximum annuel simule sur observe, INDICATIF (un releve ponctuel")
        print("              ne represente pas la moyenne d'un troncon)")
        print("decalage_mois : mois du maximum simule moins celui de l'observe")


if __name__ == "__main__":
    main()
