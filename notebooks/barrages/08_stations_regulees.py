"""Quelles stations d'entraînement portent une régularisation non modélisée ?

C'est la question décisive, et elle se pose dans ce sens-là. Un réservoir sans
station en aval ne peut pas être calé, mais il ne fait de mal à personne : il
n'entre pas dans la fonction de coût. Un réservoir EN AMONT d'une station, lui,
impose son signal à un hydrogramme contre lequel meandre s'ajuste, et comme le
modèle n'a aucun organe pour le porter, il l'absorbe dans les paramètres de sol
et de routage. C'est là que la régularisation coûte cher.

Pour chaque station : on remonte le réseau, on somme le stockage des retenues
en amont (une valeur par tronçon, la capacité maximale déclarée, cf.
06_candidats_reservoirs.py), et on le rapporte au volume écoulé moyen observé.

Sortie : barrages/data/stations-regulees.csv
"""
from collections import defaultdict, deque
from pathlib import Path

import duckdb
import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
BASES = Path("D:/meandre-data/quebec")


def amont(db):
    """{node_idx station: set des nœuds en amont}, station incluse."""
    c = duckdb.connect(str(db), read_only=True)
    enfants = defaultdict(list)
    for src, dst in c.execute("select src, dst from edges").fetchall():
        enfants[dst].append(src)
    stations = c.execute(
        "select station_id, node_idx, drainage_area_km2 from stations "
        "where node_idx is not null").fetchall()
    obs = {r[0]: r[1:] for r in c.execute(
        "select station_id, avg(discharge), min(date), max(date) from observations "
        "where discharge is not null group by station_id").fetchall()}
    c.close()

    res = {}
    for sid, nd, aire in stations:
        vus, file = {nd}, deque([nd])
        while file:
            x = file.popleft()
            for e in enfants.get(x, ()):
                if e not in vus:
                    vus.add(e)
                    file.append(e)
        q, d0, d1 = obs.get(sid, (None, None, None))
        res[sid] = (nd, aire, q, d0, d1, vus)
    return res


def main():
    # Une retenue ne pollue une station que si elle existait pendant la periode
    # observee. Contre-exemple qui impose le filtre : la station 073801 sur la
    # Romaine porte 10 691 hm3 en amont, mais elle s'arrete en juin 2014, a la
    # mise en eau de Romaine-2. Son enregistrement est naturel de bout en bout.
    bar = pl.read_parquet(DATA / "barrages.parquet").select(
        "no_barrage", "annee_construction")
    m = (pl.read_csv(DATA / "mapping-barrages-troncons.csv")
         .join(bar, on="no_barrage", how="left")
         .filter(pl.col("dans_region") & (pl.col("capacite_retenue_m3") > 0)))
    par_troncon = (m.group_by("IDTRONCON")
                   .agg(hm3=(pl.col("capacite_retenue_m3").max() / 1e6),
                        n=pl.len(),
                        nom=pl.col("toponyme_grhq").first(),
                        annee=pl.col("annee_construction").min()))

    lignes = []
    for db in sorted(BASES.glob("*.duckdb")):
        region = db.stem.upper()
        sous = par_troncon.filter(pl.col("IDTRONCON").str.starts_with(region))
        stock = {int(r["IDTRONCON"][4:]) - 1: (r["hm3"], r["nom"], r["annee"])
                 for r in sous.iter_rows(named=True)}
        for sid, (nd, aire, q_moy, d0, d1, vus) in amont(db).items():
            # la retenue doit avoir existe pendant l'essentiel de la periode
            # observee : on exige une mise en eau avant le debut du suivi
            an_debut = d0.year if d0 is not None else 1900
            dedans = [(k, v) for k, v in stock.items()
                      if k in vus and (v[2] is None or v[2] <= an_debut)]
            hm3 = sum(v[0] for _, v in dedans)
            lignes.append({
                "station_id": sid, "region": region, "aire_km2": aire,
                "obs_debut": str(d0), "obs_fin": str(d1),
                "Q_moy_obs_m3s": round(q_moy, 2) if q_moy else None,
                "n_retenues_amont": len(dedans),
                "stockage_amont_hm3": round(hm3, 1),
                "plus_grosse": max(dedans, key=lambda t: t[1][0])[1][1] if dedans else None,
                "plus_grosse_hm3": round(max(v[0] for _, v in dedans), 1) if dedans else 0.0,
            })

    d = pl.DataFrame(lignes).with_columns(
        stockage_mm=(pl.col("stockage_amont_hm3") * 1e3 / pl.col("aire_km2")).round(0),
        jours_de_debit=(pl.col("stockage_amont_hm3") * 1e6
                        / (pl.col("Q_moy_obs_m3s") * 86400)).round(1))
    d = d.sort("jours_de_debit", descending=True, nulls_last=True)
    d.write_csv(DATA / "stations-regulees.csv")

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(180)
    pl.Config.set_fmt_str_lengths(28)
    print(f"{d.height} stations")
    for s in (1, 5, 15, 30, 60):
        print(f"  stockage amont > {s:3d} jours de débit moyen : "
              f"{d.filter(pl.col('jours_de_debit') > s).height}")
    print()
    print(d.select("station_id", "region", "aire_km2", "Q_moy_obs_m3s", "n_retenues_amont",
                   "stockage_amont_hm3", "jours_de_debit", "plus_grosse").head(25))


if __name__ == "__main__":
    main()
