"""Une retenue n'est modélisable que si elle draine vers une station jaugée.

Sans station en aval, les paramètres d'une politique de lâcher ne sont
contraints par rien : le modèle peut leur donner n'importe quelle valeur sans
que la fonction de coût s'en aperçoive. C'est le filtre qui décide si le
chantier « réservoirs » a un sens, avant toute question de formulation.

Pour chaque tronçon candidat, on descend le réseau de meandre (table `edges`
de la base régionale) jusqu'à rencontrer une station ou l'exutoire, et on note
la première station atteinte, sa distance en tronçons et son bassin versant.

Le lien entre les deux mondes : IDTRONCON porte le numéro de nœud à un près
(OUTV01314 -> node_id 1314 -> node_idx 1313).

Sortie : barrages/data/candidats-reservoirs-stations.csv
"""
from pathlib import Path

import duckdb
import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
BASES = Path("D:/meandre-data/quebec")


def descendre(db):
    """{node_idx: (station_id, n_troncons, aire_station_km2)} pour tout nœud."""
    c = duckdb.connect(str(db), read_only=True)
    aval = dict(c.execute("select src, dst from edges").fetchall())
    st = {r[0]: (r[1], r[2]) for r in
          c.execute("select node_idx, station_id, drainage_area_km2 from stations "
                    "where node_idx is not null").fetchall()}
    n_nodes = c.execute("select count(*) from nodes").fetchone()[0]
    c.close()

    memo = {}

    def resoudre(n):
        chemin = []
        cur = n
        while True:
            if cur in memo:
                trouve = memo[cur]
                break
            if cur in st and cur != n:
                sid, aire = st[cur]
                trouve = (sid, 0, aire)
                break
            chemin.append(cur)
            suivant = aval.get(cur)
            if suivant is None or suivant in chemin:
                trouve = (None, None, None)
                break
            cur = suivant
        for k, nd in enumerate(reversed(chemin)):
            memo[nd] = (trouve[0], (trouve[1] + k + 1) if trouve[1] is not None else None,
                        trouve[2])
        return memo.get(n, trouve)

    # un nœud qui PORTE une station est evidemment contraint : distance 0
    for nd, (sid, aire) in st.items():
        memo[nd] = (sid, 0, aire)
    for nd in range(n_nodes):
        if nd not in memo:
            resoudre(nd)
    return memo, len(st)


def main():
    cand = pl.read_csv(DATA / "candidats-reservoirs.csv")
    cand = cand.with_columns(
        region=pl.col("IDTRONCON").str.slice(0, 4).str.to_lowercase(),
        node_idx=pl.col("IDTRONCON").str.slice(4).cast(pl.Int64) - 1)

    lignes, n_st_total = [], 0
    for region in sorted(cand["region"].unique()):
        db = BASES / f"{region}.duckdb"
        if not db.exists():
            print(f"  base absente : {region}")
            continue
        memo, n_st = descendre(db)
        n_st_total += n_st
        sous = cand.filter(pl.col("region") == region)
        for r in sous.iter_rows(named=True):
            sid, dist, aire = memo.get(r["node_idx"], (None, None, None))
            lignes.append({**r, "station_aval": sid, "troncons_jusqu_station": dist,
                           "aire_station_km2": aire})
        print(f"  {region}: {sous.height:3d} candidats, {n_st:2d} stations")

    out = pl.DataFrame(lignes).drop("region", "node_idx")
    out = out.with_columns(
        contraint=pl.col("station_aval").is_not_null(),
        stockage_mm_sur_station=(pl.col("stockage_hm3") * 1e6
                                 / (pl.col("aire_station_km2") * 1e6) * 1000).round(0))
    out.write_csv(DATA / "candidats-reservoirs-stations.csv")

    tot = out["stockage_hm3"].sum()
    ok = out.filter(pl.col("contraint"))
    print(f"\n{n_st_total} stations sur l'ensemble des régions")
    print(f"{out.height} tronçons candidats, {tot:,.0f} hm³")
    print(f"  en amont d'une station : {ok.height} tronçons, "
          f"{ok['stockage_hm3'].sum():,.0f} hm³ ({100 * ok['stockage_hm3'].sum() / tot:.0f} %)")
    print(f"  aucun jaugeage en aval : {out.height - ok.height} tronçons, "
          f"{tot - ok['stockage_hm3'].sum():,.0f} hm³")
    print()
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(170)
    pl.Config.set_fmt_str_lengths(30)
    print(out.sort("stockage_hm3", descending=True)
          .select("IDTRONCON", "toponyme_grhq", "stockage_hm3", "station_aval",
                  "troncons_jusqu_station", "aire_station_km2", "stockage_mm_sur_station")
          .head(25))


if __name__ == "__main__":
    main()
