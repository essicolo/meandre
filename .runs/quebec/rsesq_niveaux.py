"""Niveaux d'eau souterraine du RSESQ : série journalière par puits, et couverture par région.

La base de diffusion porte 41 millions de mesures aux six heures. On en tire une moyenne
journalière par puits, puis on rattache chaque puits au tronçon le plus proche. Le niveau est
une profondeur sous le repère du tubage : seules ses VARIATIONS sont comparables au stockage
souterrain simulé, jamais sa valeur absolue.

    .venv/Scripts/python.exe .runs/quebec/rsesq_niveaux.py
"""
import os
import sqlite3
import sys

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

SOURCE = f"{_paths.SOURCES_ROOT}/eaux-souterraines-rsesq"
BASE = f"{SOURCE}/rsesq_prod_2023_04_05_diffusion_ext.db"
RESEAU = f"zip://{SOURCE}/reseau_eaux_souterraines_aout2026.gdb.zip"
REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
DISTANCE_MAX_KM = 3.0


def series_journalieres():
    c = sqlite3.connect(f"file:{BASE}?mode=ro", uri=True)
    d = pd.read_sql("""
        select s.sampling_feature_name as puits, date(t.datetime) as date,
               avg(t.value) as niveau_m, count(*) as n
        from timeseries_data t
        join timeseries_channel c on c.channel_id = t.channel_id
        join observed_property p on p.obs_property_id = c.obs_property_id
        join observation o on o.observation_id = c.observation_id
        join sampling_feature s on s.sampling_feature_uuid = o.sampling_feature_uuid
        where p.observed_property = 'NIV_EAU'
        group by 1, 2""", c)
    c.close()
    d["date"] = pd.to_datetime(d.date)
    return d


def rattacher():
    """Tronçon le plus proche de chaque puits, par région."""
    g = gpd.read_file(RESEAU, layer="Suivi_eaux_souterraines").to_crs(4326)
    lignes = []
    for reg in REGIONS:
        base = f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb"
        if not os.path.exists(base):
            continue
        nd = duckdb.connect(base, read_only=True).sql("select node_idx, node_id, lon, lat from nodes").fetchdf()
        d = np.hypot(g.geometry.x.values[:, None] - nd.lon.values[None, :], g.geometry.y.values[:, None] - nd.lat.values[None, :]) * 111.0
        j = d.argmin(axis=1)
        lignes.append(pd.DataFrame({"puits": g.ID_PUITS.values, "region": reg, "node_idx": nd.node_idx.values[j],
                                    "troncon": nd.node_id.values[j], "distance_km": d[np.arange(len(g)), j],
                                    "aquifere": g.AQUIFERE.values, "confinement": g.CONFINEMENT.values,
                                    "influence": g.INFLUENCE.values, "lat": g.geometry.y.values, "lon": g.geometry.x.values}))
    t = pd.concat(lignes, ignore_index=True)
    return t.loc[t.groupby("puits").distance_km.idxmin()].query("distance_km <= @DISTANCE_MAX_KM").reset_index(drop=True)


def main():
    d = series_journalieres()
    print(f"{d.puits.nunique()} puits, {len(d)} jours-puits, {d.date.min().date()} à {d.date.max().date()}")
    r = rattacher()
    print(f"{len(r)} puits à moins de {DISTANCE_MAX_KM} km d'un tronçon")
    d = d.merge(r, on="puits", how="inner")
    sortie = f"{_paths.DERIVED_ROOT}/auxiliaires"
    os.makedirs(sortie, exist_ok=True)
    d.to_parquet(f"{sortie}/rsesq-niveaux-journaliers.parquet", index=False)
    r.to_parquet(f"{sortie}/rsesq-puits.parquet", index=False)
    for periode, debut, fin in (("2001-2024", "2001-01-01", "2024-12-31"), ("2022-2024", "2022-01-01", "2024-12-31")):
        s = d[(d.date >= debut) & (d.date <= fin)]
        p = s.groupby(["region", "puits"]).date.agg(["min", "max", "count"]).reset_index()
        g = p.groupby("region").agg(puits=("puits", "nunique"), jours_medians=("count", "median")).reset_index()
        print(f"\n{periode} : {p.puits.nunique()} puits, {len(s)} jours-puits")
        print(g.to_string(index=False))
    print(f"\n{sortie}/rsesq-niveaux-journaliers.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
