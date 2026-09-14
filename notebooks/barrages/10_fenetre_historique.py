"""Les stations en aval des grands réservoirs existent, mais avant 1995.

Constat de 09_gain_hydat.py : sur 2001-2024, presque aucune grande retenue n'a
de jaugeage en aval, ni au CEHQ ni dans HYDAT. En regardant les dates, le motif
est systématique : les stations exploitées par Hydro-Québec (Gouin aval,
La Tuque, Grande-Mère, Manic-2, Manic-5, Outardes-3 et 4, Paugan, Rapides
Farmers, Dozois, Rapide-Sept, Rapides-2, Rapides-des-Îles) s'arrêtent toutes en
1993 ou 1994. Elles ne sont pas absentes, elles sont anciennes.

Ce script mesure la fenêtre exploitable : combien de stations HYDAT en aval
d'une retenue candidate couvrent une période où le forçage existe aussi
(CaSR démarre en 1980).

Sortie : barrages/data/fenetre-historique.csv
"""
import sqlite3
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
BASES = Path("D:/meandre-data/quebec")
HYDAT = Path("D:/meandre-data/hydat/Hydat.sqlite3")
CRS = "EPSG:32198"
RAYON_M = 2000.0
FENETRE = (1980, 1994)      # CaSR commence en 1980, les stations HQ s'arretent en 1994


def stations(fenetre):
    c = sqlite3.connect(HYDAT)
    d = pd.read_sql("""
        select s.STATION_NUMBER station_id, s.STATION_NAME nom,
               s.LATITUDE lat, s.LONGITUDE lon, s.DRAINAGE_AREA_GROSS aire_km2,
               min(f.YEAR) an0, max(f.YEAR) an1,
               (select REGULATED from STN_REGULATION r
                 where r.STATION_NUMBER = s.STATION_NUMBER) regule
        from STATIONS s join DLY_FLOWS f on f.STATION_NUMBER = s.STATION_NUMBER
        where s.PROV_TERR_STATE_LOC = 'QC'
        group by s.STATION_NUMBER""", c)
    c.close()
    a, b = fenetre
    return d[(d.an0 <= a) & (d.an1 >= b) & d.lat.notna()].reset_index(drop=True)


def main():
    hy = stations(FENETRE)
    print(f"{len(hy)} stations HYDAT couvrant {FENETRE[0]}-{FENETRE[1]} en continu "
          f"({int((hy.regule == 1).sum())} régularisées)")

    cand = (pl.read_csv(DATA / "candidats-reservoirs.csv")
            .with_columns(region=pl.col("IDTRONCON").str.slice(0, 4).str.to_lowercase(),
                          node_idx=pl.col("IDTRONCON").str.slice(4).cast(pl.Int64) - 1))

    lignes = []
    for region in sorted(cand["region"].unique()):
        db = BASES / f"{region}.duckdb"
        if not db.exists():
            continue
        c = duckdb.connect(str(db), read_only=True)
        nodes = c.execute("select node_idx, lon, lat from nodes").df()
        aval = dict(c.execute("select src, dst from edges").fetchall())
        c.close()

        g_n = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
                               crs="EPSG:4326").to_crs(CRS)
        g_s = gpd.GeoDataFrame(hy, geometry=gpd.points_from_xy(hy.lon, hy.lat),
                               crs="EPSG:4326").to_crs(CRS)
        j = gpd.sjoin_nearest(g_s, g_n[["node_idx", "geometry"]], distance_col="d_m")
        j = j[j.d_m <= RAYON_M].drop_duplicates("station_id")
        st = {int(r.node_idx): (r.station_id, r.nom, r.an0, r.an1) for r in j.itertuples()}

        for r in cand.filter(pl.col("region") == region).iter_rows(named=True):
            cur, k, vus, trouve = r["node_idx"], 0, set(), None
            while cur is not None and cur not in vus:
                if cur in st and k > 0:
                    trouve = st[cur]
                    break
                vus.add(cur)
                cur = aval.get(cur)
                k += 1
            lignes.append({
                "IDTRONCON": r["IDTRONCON"], "toponyme_grhq": r["toponyme_grhq"],
                "stockage_hm3": r["stockage_hm3"],
                "station": trouve[0] if trouve else None,
                "nom_station": trouve[1] if trouve else None,
                "an0": trouve[2] if trouve else None,
                "an1": trouve[3] if trouve else None,
                "troncons": k if trouve else None})

    d = pl.DataFrame(lignes)
    d.write_csv(DATA / "fenetre-historique.csv")
    ok = d.filter(pl.col("station").is_not_null())
    tot = d["stockage_hm3"].sum()
    print(f"\nsur la fenêtre {FENETRE[0]}-{FENETRE[1]} : {ok.height}/{d.height} tronçons "
          f"candidats ont un jaugeage en aval, {ok['stockage_hm3'].sum():,.0f} hm³ sur "
          f"{tot:,.0f} ({100 * ok['stockage_hm3'].sum() / tot:.0f} %)")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(180)
    pl.Config.set_fmt_str_lengths(46)
    print(ok.sort("stockage_hm3", descending=True)
          .select("IDTRONCON", "toponyme_grhq", "stockage_hm3", "station", "nom_station",
                  "an0", "an1").head(20))


if __name__ == "__main__":
    main()
