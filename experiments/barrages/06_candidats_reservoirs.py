"""Dresse la liste des tronçons qui méritent un nœud de stockage.

Un barrage n'intéresse la modélisation que s'il retient assez d'eau pour
moduler le tronçon. Le tri se fait donc sur la capacité de retenue, et sur le
temps de résidence qu'elle impose quand le débit simulé est disponible.

ATTENTION à l'agrégation. Une retenue est fermée par plusieurs ouvrages
(Baskatong en compte onze : un barrage principal, deux barrages secondaires et
huit digues) et chacun déclare la capacité de la retenue qu'il ferme. Sommer
les ouvrages d'un même tronçon compterait la retenue autant de fois qu'elle a
d'ouvrages. On prend donc le MAXIMUM, qui redonne les valeurs publiées :
Baskatong 3 664 hm³, Manicouagan 141 851 hm³, Romaine-2 3 720 hm³.

Rappel : la capacité de retenue du CEHQ est le volume total (hauteur de la
retenue × superficie du réservoir), pas le volume utile. C'est une borne
supérieure du marnage exploitable.

Sortie : barrages/data/candidats-reservoirs.csv
"""
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
N_GARDES = 120


def main():
    m = (pl.read_csv(DATA / "mapping-barrages-troncons.csv")
         .filter(pl.col("dans_region") & (pl.col("capacite_retenue_m3") > 0)))

    g = (m.group_by("IDTRONCON")
         .agg(toponyme_grhq=pl.col("toponyme_grhq").first(),
              region_administrative=pl.col("region_administrative").first(),
              n_ouvrages=pl.len(),
              stockage_hm3=(pl.col("capacite_retenue_m3").max() / 1e6).round(1),
              Q_moy_m3s=pl.col("Q_moy_m3s").max(),
              hauteur_max_m=pl.col("hauteur_m").max(),
              annee_construction=pl.col("no_barrage").count(),
              usages=pl.col("utilisations").drop_nulls().unique().str.join(" ; "),
              qualite_pire=pl.col("qualite").unique().str.join(" ; "),
              barrages=pl.col("no_barrage").str.join(" "),
              noms=pl.col("nom_barrage").drop_nulls().unique().str.join(" | "))
         .drop("annee_construction")
         .with_columns(
             jours_de_debit=(pl.col("stockage_hm3") * 1e6
                             / (pl.col("Q_moy_m3s") * 86400)).round(1))
         .sort("stockage_hm3", descending=True))

    tot = g["stockage_hm3"].sum()
    print(f"{g.height} tronçons portent au moins un ouvrage, {tot:,.0f} hm³ au total")
    for k in (10, 20, 50, 100, 200):
        print(f"  les {k:3d} premiers : {100 * g.head(k)['stockage_hm3'].sum() / tot:5.1f} %")

    dest = DATA / "candidats-reservoirs.csv"
    g.head(N_GARDES).write_csv(dest)
    print(f"\n{N_GARDES} tronçons -> {dest}")


if __name__ == "__main__":
    main()
