"""Ce que HYDAT changerait au verrou des réservoirs.

Le réseau de stations actuellement injecté dans meandre est celui du CEHQ, soit
178 stations. HYDAT en compte 755 au Québec avec des débits journaliers. La
question n'est pas leur nombre mais leur position : une station de plus en
amont d'un réservoir déjà jaugé n'apporte rien, une station en aval d'une
retenue jusqu'ici muette rend sa politique de lâcher observable.

On rattache donc les stations HYDAT aux nœuds meandre, puis on recompte les
tronçons candidats qui acquièrent un jaugeage en aval.

Sortie : barrages/data/gain-hydat.csv
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
# au-dela, la station ne tombe pas sur le reseau modelise
RAYON_M = 2000.0
# on ne retient que les stations utiles a la periode d'entrainement
AN_MIN, AN_MAX = 2001, 2024


def stations_hydat():
    c = sqlite3.connect(HYDAT)
    d = pd.read_sql("""
        select s.STATION_NUMBER station_id, s.STATION_NAME nom,
               s.LATITUDE lat, s.LONGITUDE lon, s.DRAINAGE_AREA_GROSS aire_km2,
               min(f.YEAR) an0, max(f.YEAR) an1, count(*) n_annees,
               (select REGULATED from STN_REGULATION r
                 where r.STATION_NUMBER = s.STATION_NUMBER) regule
        from STATIONS s join DLY_FLOWS f on f.STATION_NUMBER = s.STATION_NUMBER
        where s.PROV_TERR_STATE_LOC = 'QC'
        group by s.STATION_NUMBER""", c)
    c.close()
    return d[(d.an1 >= AN_MIN) & (d.an0 <= AN_MAX) & d.lat.notna()].reset_index(drop=True)


def rattacher(hy, db):
    """Nœud meandre le plus proche de chaque station, dans une région donnée."""
    c = duckdb.connect(str(db), read_only=True)
    nodes = c.execute("select node_idx, lon, lat from nodes").df()
    aval = dict(c.execute("select src, dst from edges").fetchall())
    cehq = c.execute("select node_idx from stations where node_idx is not null").fetchall()
    c.close()

    g_n = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
                           crs="EPSG:4326").to_crs(CRS)
    g_s = gpd.GeoDataFrame(hy, geometry=gpd.points_from_xy(hy.lon, hy.lat),
                           crs="EPSG:4326").to_crs(CRS)
    j = gpd.sjoin_nearest(g_s, g_n[["node_idx", "geometry"]], distance_col="d_m")
    j = j[j.d_m <= RAYON_M].drop_duplicates("station_id")
    return j, aval, {x[0] for x in cehq}


def premiere_station(n, aval, stations):
    """Première station rencontrée en descendant, et le nombre de tronçons."""
    cur, k, vus = n, 0, set()
    while cur is not None and cur not in vus:
        if cur in stations and k > 0:
            return stations[cur], k
        vus.add(cur)
        cur = aval.get(cur)
        k += 1
    return None, None


def main():
    hy = stations_hydat()
    print(f"{len(hy)} stations HYDAT au Québec couvrant {AN_MIN}-{AN_MAX}")

    cand = (pl.read_csv(DATA / "candidats-reservoirs.csv")
            .with_columns(region=pl.col("IDTRONCON").str.slice(0, 4).str.to_lowercase(),
                          node_idx=pl.col("IDTRONCON").str.slice(4).cast(pl.Int64) - 1))

    lignes, n_hydat_rattachees = [], 0
    for region in sorted(cand["region"].unique()):
        db = BASES / f"{region}.duckdb"
        if not db.exists():
            continue
        j, aval, cehq_nodes = rattacher(hy, db)
        n_hydat_rattachees += len(j)
        hydat_nodes = {int(r.node_idx): r.station_id for r in j.itertuples()}
        cehq_only = {n: "CEHQ" for n in cehq_nodes}
        union = {**hydat_nodes, **cehq_only}

        for r in cand.filter(pl.col("region") == region).iter_rows(named=True):
            s_c, d_c = premiere_station(r["node_idx"], aval, cehq_only)
            s_u, d_u = premiere_station(r["node_idx"], aval, union)
            lignes.append({
                "IDTRONCON": r["IDTRONCON"], "toponyme_grhq": r["toponyme_grhq"],
                "stockage_hm3": r["stockage_hm3"],
                "contraint_cehq": s_c is not None,
                "station_hydat": s_u if s_u not in (None, "CEHQ") else None,
                "troncons_jusqu_station": d_u,
                "contraint_union": s_u is not None})
        print(f"  {region}: {len(j):3d} stations HYDAT rattachées au réseau")

    d = pl.DataFrame(lignes)
    d.write_csv(DATA / "gain-hydat.csv")

    tot = d["stockage_hm3"].sum()
    for lib, col in (("CEHQ seul", "contraint_cehq"), ("CEHQ + HYDAT", "contraint_union")):
        s = d.filter(pl.col(col))
        print(f"\n{lib} : {s.height}/{d.height} tronçons candidats contraints, "
              f"{s['stockage_hm3'].sum():,.0f} hm³ sur {tot:,.0f} "
              f"({100 * s['stockage_hm3'].sum() / tot:.0f} %)")
    gagnes = d.filter(~pl.col("contraint_cehq") & pl.col("contraint_union"))
    print(f"\ngagnés par HYDAT : {gagnes.height} tronçons, "
          f"{gagnes['stockage_hm3'].sum():,.0f} hm³")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(150)
    pl.Config.set_fmt_str_lengths(32)
    print(gagnes.sort("stockage_hm3", descending=True)
          .select("IDTRONCON", "toponyme_grhq", "stockage_hm3", "station_hydat",
                  "troncons_jusqu_station").head(20))


if __name__ == "__main__":
    main()
