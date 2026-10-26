"""Rattache chaque barrage du répertoire CEHQ à un tronçon meandre.

Même recette que 01_mapping.qmd pour les sites de prélèvement : projection en
EPSG:32198 puis `sjoin_nearest`. Deux particularités des barrages :

- un barrage retient une masse d'eau ; quand la fiche déclare un lac (ou un nom
  de réservoir), on privilégie le tronçon-lac le plus proche (`est_lac`) ;
- la fiche donne la superficie du bassin versant amont, ce qui permet de
  vérifier le rattachement autrement que par la distance (voir 05_verification).

Sortie : barrages/data/mapping-barrages-troncons.csv
"""
import unicodedata
import re
from pathlib import Path

import geopandas as gpd
import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
# Le reseau de troncons, les toponymes GRHQ et la climatologie de debit
# viennent du depot io-eau, qui reste leur unique responsable.
IOEAU = Path("C:/Users/parse01/documents-locaux/GitHub/io-eau")
CRS = "EPSG:32198"
# au-dela de ce rayon, on ne cherche plus le tronçon-lac : le lac declare
# n'est pas represente dans le reseau
RAYON_LAC_M = 2000.0


def sans_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


GENERIQUES = {"riviere", "lac", "ruisseau", "petit", "petite", "grand", "grande",
              "du", "de", "des", "la", "le", "les", "d", "l", "a", "au", "aux",
              "reservoir", "etang", "branche", "bras", "riv", "rive", "nord",
              "sud", "est", "ouest"}


def mots_cles(nom):
    """Jeu de mots discriminants d'un toponyme, insensible à la forme inversée.

    Le CEHQ écrit « Noire, Rivière », le GRHQ « Rivière Noire » : en retirant
    les génériques et la ponctuation, les deux donnent {noire}.
    """
    if not nom:
        return frozenset()
    s = sans_accents(str(nom)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return frozenset(m for m in s.split() if m and m not in GENERIQUES)


def main():
    bar = pl.read_parquet(DATA / "barrages.parquet")
    trc = gpd.read_parquet(IOEAU / "data" / "source" / "troncons.parquet")
    topo = pl.read_csv(IOEAU / "data" / "derived" / "troncons-toponymes.csv")

    b = bar.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
    print(f"{bar.height} barrages, {b.height} géolocalisés")

    g = gpd.GeoDataFrame(
        b.select("no_barrage").to_pandas(),
        geometry=gpd.points_from_xy(b["longitude"], b["latitude"]),
        crs="EPSG:4326",
    ).to_crs(CRS)

    trc_p = trc[["IDTRONCON", "est_lac", "geometry"]].to_crs(CRS)

    def plus_proche(pts, cible, suffixe):
        j = gpd.sjoin_nearest(pts, cible, how="left", distance_col=f"distance_m{suffixe}")
        j = j.drop(columns=["index_right"]).drop_duplicates("no_barrage")
        return j.rename(columns={"IDTRONCON": f"IDTRONCON{suffixe}",
                                 "est_lac": f"est_lac{suffixe}"})

    tout = plus_proche(g, trc_p, "")
    lacs = plus_proche(g, trc_p[trc_p["est_lac"]], "_lac")

    m = (pl.from_pandas(tout.drop(columns="geometry"))
         .join(pl.from_pandas(lacs.drop(columns="geometry")), on="no_barrage", how="left")
         .join(b, on="no_barrage", how="left"))

    # un barrage qui retient un lac ou un reservoir nomme est rattache au
    # troncon-lac s'il en existe un a portee
    declare_lac = (pl.col("lac_numero").is_not_null() | pl.col("lac_nom").is_not_null()
                   | pl.col("nom_reservoir").is_not_null())
    m = m.with_columns(
        relocalise_lac=(declare_lac & ~pl.col("est_lac")
                        & (pl.col("distance_m_lac") <= RAYON_LAC_M)).fill_null(False))
    m = m.with_columns(
        IDTRONCON_final=pl.when(pl.col("relocalise_lac")).then(pl.col("IDTRONCON_lac"))
                          .otherwise(pl.col("IDTRONCON")),
        distance_m_final=pl.when(pl.col("relocalise_lac")).then(pl.col("distance_m_lac"))
                           .otherwise(pl.col("distance_m")),
    )

    # accord de nom : le toponyme GRHQ du tronçon retenu contre les noms
    # hydrographiques declares par la fiche
    m = m.join(topo.select("IDTRONCON", "toponyme_grhq"),
               left_on="IDTRONCON_final", right_on="IDTRONCON", how="left")
    noms_cehq = [
        [x for x in (r["lac_nom"], r["cours_eau_nom"], r["nom_reservoir"], r["bassin_nom"])
         if x] for r in m.iter_rows(named=True)
    ]
    accord, cles = [], []
    for r, noms in zip(m.iter_rows(named=True), noms_cehq):
        cible = mots_cles(r["toponyme_grhq"])
        cle_cehq = [mots_cles(n) for n in noms]
        cles.append(" ; ".join(sorted(set().union(*cle_cehq))) if cle_cehq else "")
        if not cible or not any(cle_cehq):
            accord.append(None)                    # indecidable : un nom manque
        else:
            accord.append(any(k & cible for k in cle_cehq if k))
    m = m.with_columns(accord_nom=pl.Series(accord, dtype=pl.Boolean),
                       mots_cles_cehq=pl.Series(cles))

    # --- hors couverture : le point ne tombe dans aucune region hydrographique
    regions = gpd.read_parquet(IOEAU / "data" / "source" / "regions.parquet").to_crs(CRS)
    union = regions.geometry.union_all()
    dedans = g.set_index("no_barrage").geometry.within(union)
    m = m.join(pl.DataFrame({"no_barrage": dedans.index.tolist(),
                             "dans_region": dedans.to_numpy()}),
               on="no_barrage", how="left")

    # --- superficie implicite du tronçon, deduite du debit climatologique
    q = (pl.read_parquet(IOEAU / "data" / "derived" / "q_climato_2001-2025.parquet")
         .group_by("IDTRONCON").agg(Q_moy_m3s=pl.col("Q_p50_m3s").mean()))
    m = m.join(q, left_on="IDTRONCON_final", right_on="IDTRONCON", how="left")

    # etalon : debit specifique median sur les rattachements les mieux assures
    ref = m.filter((pl.col("distance_m_final") < 100) & (pl.col("accord_nom") == True)  # noqa: E712
                   & (pl.col("superficie_bv_km2") > 0) & (pl.col("Q_moy_m3s") > 0))
    q_spec = (ref["Q_moy_m3s"] * 1000 / ref["superficie_bv_km2"]).median()
    print(f"débit spécifique de référence : {q_spec:.1f} L/s/km² "
          f"(sur {ref.height} rattachements sûrs)")
    m = m.with_columns(
        superficie_implicite_km2=(pl.col("Q_moy_m3s") * 1000 / q_spec).round(1))
    m = m.with_columns(
        ratio_superficie=(pl.col("superficie_implicite_km2") / pl.col("superficie_bv_km2")))

    # --- verdict par barrage
    # « sous-maille » : le tronçon draine plus de cinq fois le bassin du barrage,
    # autrement dit le cours d'eau barré n'est pas represente dans le reseau.
    # Le debit climatologique ne couvre qu'une partie des tronçons ; a defaut,
    # on tranche sur la superficie declaree, calibree sur le sous-ensemble ou le
    # ratio est calculable (98 % des barrages de moins de 1 km2 y sont
    # sous-maille, 87 % de ceux de moins de 10 km2).
    SEUIL_SOUS_MAILLE_KM2 = 10.0
    m = m.with_columns(
        qualite=pl.when(~pl.col("dans_region").fill_null(False)).then(pl.lit("hors couverture"))
        .when(pl.col("ratio_superficie") > 5).then(pl.lit("sous-maille"))
        .when(pl.col("ratio_superficie").is_null()
              & (pl.col("superficie_bv_km2") < SEUIL_SOUS_MAILLE_KM2))
        .then(pl.lit("sous-maille"))
        .when((pl.col("distance_m_final") <= 150)
              & ((pl.col("accord_nom") == True)  # noqa: E712
                 | pl.col("ratio_superficie").is_between(0.3, 3)))
        .then(pl.lit("assuré"))
        .otherwise(pl.lit("douteux")))

    sortie = m.select(
        "no_barrage", "nom_barrage", "categorie", "latitude", "longitude",
        "region_administrative", "municipalite", "mrc",
        pl.col("IDTRONCON_final").alias("IDTRONCON"),
        pl.col("distance_m_final").round(1).alias("distance_m"),
        "est_lac", "relocalise_lac", "dans_region", "toponyme_grhq", "accord_nom",
        "lac_nom", "cours_eau_nom", "bassin_nom", "nom_reservoir",
        "superficie_bv_km2", "Q_moy_m3s", "superficie_implicite_km2",
        pl.col("ratio_superficie").round(2), "qualite",
        "capacite_retenue_m3", "hauteur_m",
        "hauteur_retenue_m", "superficie_reservoir_ha", "utilisations",
    )
    dest = DATA / "mapping-barrages-troncons.csv"
    sortie.write_csv(dest)
    print(f"{sortie.height} rattachements -> {dest}")

    # meme contenu en points, pour la cartographie (cf. prelevements.geojson)
    pts = sortie.to_pandas()
    gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
                     crs="EPSG:4326").to_file(DATA / "barrages.geojson",
                                              driver="GeoJSON")
    print(sortie.select(
        n_lac=pl.col("relocalise_lac").sum(),
        d_med=pl.col("distance_m").median(),
        d_p90=pl.col("distance_m").quantile(0.9),
        d_max=pl.col("distance_m").max(),
        accord=pl.col("accord_nom").sum(),
        desaccord=(~pl.col("accord_nom")).sum(),
        indecidable=pl.col("accord_nom").is_null().sum(),
    ))
    print(sortie.group_by("qualite").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
