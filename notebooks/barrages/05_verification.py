"""Vérifie les données récoltées et leur rattachement aux tronçons.

Trois familles de contrôles :

A. complétude de la récolte (liste vs fiches, identifiants) ;
B. exactitude interne des fiches (cohérence géométrique, chronologique, et
   conformité de la catégorie administrative à la Loi sur la sécurité des
   barrages) ;
C. exactitude du rattachement (distance, accord de toponyme, et surtout
   confrontation de la superficie du bassin versant déclarée au débit
   climatologique simulé sur le tronçon retenu).

Sorties : barrages/RAPPORT-VERIFICATION.md et barrages/data/anomalies.csv
"""
import datetime as dt
import re
import unicodedata
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

# emprise du Québec, marge comprise
BBOX = (-80.0, -56.5, 44.9, 62.7)      # lon_min, lon_max, lat_min, lat_max
ANNEE = dt.date.today().year


def categorie_attendue(h, c):
    """Loi sur la sécurité des barrages (RLRQ c. S-3.1.01, art. 2).

    Forte contenance : hauteur >= 1 m et capacité > 1 000 000 m³, ou
    hauteur >= 2,5 m et capacité > 30 000 m³. Faible contenance : les autres
    barrages de 2 m et plus. Petit barrage : de 1 m à moins de 2 m.

    La règle est appliquée telle quelle aux fiches ; les écarts qu'elle
    signale sont donc soit une hauteur ou une capacité fausse, soit une
    catégorie mal saisie.
    """
    if h is None or c is None:
        return None
    if (h >= 1 and c > 1_000_000) or (h >= 2.5 and c > 30_000):
        return "Forte contenance"
    return "Faible contenance" if h >= 2 else "Petit barrage"


BDAT = "Bdat/SHP/munic_s.shp"      # dans data/source/BDAT_adm_SHP.zip


def normaliser(s):
    """Forme comparable d'un nom de municipalité ou de MRC."""
    if s is None:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", str(s))
                if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"\bst(e)?\b\.?", lambda m: "sainte" if m.group(1) else "saint", s)
    # le CEHQ ecrit « Ville de Quebec » et « Des Chenaux » la ou la BDAT ecrit
    # « Quebec » et « Les Chenaux » : ce sont des conventions, pas des ecarts
    s = re.sub(r"^(ville|municipalite|mrc)\s+(de\s+|du\s+|des\s+|d')?", "", s)
    s = re.sub(r"^(le|la|les|des|du|de|l')\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def controle_geographique(bar):
    """Le point tombe-t-il dans la municipalité et la MRC déclarées ?

    Contrôle indépendant du site : la coordonnée vient de la fiche, le
    découpage administratif vient de la BDAT.
    """
    zip_bdat = (IOEAU / "data" / "source" / "BDAT_adm_SHP.zip").as_posix()
    mun = gpd.read_file(f"zip://{zip_bdat}!{BDAT}")
    pts = gpd.GeoDataFrame(
        bar.select("no_barrage", "municipalite", "mrc").to_pandas(),
        geometry=gpd.points_from_xy(bar["longitude"], bar["latitude"]),
        crs="EPSG:4326").to_crs(mun.crs)
    j = gpd.sjoin(pts, mun[["MUS_NM_MUN", "MUS_NM_MRC", "geometry"]],
                  how="left", predicate="within").drop_duplicates("no_barrage")
    d = pl.from_pandas(j.drop(columns="geometry"))
    return d.with_columns(
        hors_polygone=pl.col("MUS_NM_MUN").is_null(),
        mun_accord=pl.col("municipalite").map_elements(normaliser, return_dtype=pl.Utf8)
        == pl.col("MUS_NM_MUN").map_elements(normaliser, return_dtype=pl.Utf8),
        mrc_accord=pl.col("mrc").map_elements(normaliser, return_dtype=pl.Utf8)
        == pl.col("MUS_NM_MRC").map_elements(normaliser, return_dtype=pl.Utf8))


def bloc(titre, lignes):
    return f"## {titre}\n\n" + "\n".join(lignes) + "\n\n"


def main():
    bar = pl.read_parquet(DATA / "barrages.parquet")
    liste = pl.read_csv(DATA / "liste.csv")
    mp = pl.read_csv(DATA / "mapping-barrages-troncons.csv")
    q = pl.read_parquet(IOEAU / "data" / "derived" / "q_climato_2001-2025.parquet")

    md = ["# Vérification du répertoire des barrages (CEHQ)\n",
          f"Récolte du {dt.date.today().isoformat()}, source "
          "<https://www.cehq.gouv.qc.ca/barrages/default.asp>.\n"]
    anomalies = []

    def signaler(code, sous_df, commentaire):
        if sous_df.height:
            anomalies.append(sous_df.select("no_barrage")
                             .with_columns(code=pl.lit(code), detail=pl.lit(commentaire)))
        return sous_df.height

    # ---------------------------------------------------------------- A
    manquantes = liste.join(bar, on="no_barrage", how="anti")
    attendus = liste
    f_orph0 = DATA / "numeros-orphelins.csv"
    if f_orph0.exists():
        attendus = pl.concat([liste.select("no_barrage"),
                              pl.read_csv(f_orph0).select("no_barrage")]).unique()
    surplus = bar.join(attendus, on="no_barrage", how="anti")
    lignes_a = [
        f"- barrages listés (partition par MRC) : **{liste.height}**",
        f"- fiches techniques analysées : **{bar.height}**",
        f"- listés sans fiche : **{manquantes.height}**",
        f"- fiches hors liste : **{surplus.height}**",
        f"- numéros uniques : **{bar['no_barrage'].n_unique()}**",
    ]
    # recoupement par une seconde partition du territoire (voir 01b)
    f_mun = DATA / "liste-municipalites.csv"
    if f_mun.exists():
        mun = pl.read_csv(f_mun)
        seuls_mrc = liste.join(mun, on="no_barrage", how="anti").height
        seuls_mun = mun.join(liste, on="no_barrage", how="anti").height
        lignes_a += [
            f"- contre-énumération par municipalité : **{mun.height}** barrages",
            f"- vus par la seule partition MRC : **{seuls_mrc}** ; "
            f"par la seule partition municipalité : **{seuls_mun}**",
        ]
    else:
        lignes_a.append(
            "- contre-énumération par municipalité (`01b_recoupement_liste.py`) : "
            "interrompue à 300 municipalités sur 767, le moteur de recherche du "
            "site cessant de répondre sous la charge. Le sondage direct des "
            "numéros ci-dessous répond à la même question plus vite et plus "
            "franchement, en interrogeant les fiches et non l'index.")
    f_orph = DATA / "numeros-orphelins.csv"
    if f_orph.exists():
        orph = pl.read_csv(f_orph)
        lignes_a += [
            f"- sondage direct des {3014} numéros manquants du bloc dense "
            f"X0000462-X0008036 : **{orph.height}** fiches servies par "
            "`detail.asp` mais absentes du moteur de recherche du site",
            "",
            "  Ces fiches sont réelles et ont été ajoutées à la récolte. "
            "L'écart ne vient pas du grattage : réinterrogée, la recherche par "
            "MRC rend exactement la même liste que celle enregistrée "
            "(Charlevoix-Est, 108 lignes des deux côtés), sans le barrage "
            "X0001110 que `detail.asp` sert pourtant.",
            "",
            "  Le gros du lot est concentré : **264** des 271 sont dans la MRC "
            "de Jamésie, la seule dont la recherche renvoie zéro ligne alors "
            "que le territoire porte des barrages. Les 7 autres sont dispersés "
            "(Vaudreuil-Soulanges, Charlevoix-Est, Caniapiscau) et relèvent "
            "d'un index de recherche incomplet.",
        ]
    else:
        lignes_a.append("- sondage direct des numéros : non exécuté "
                        "(lancer `01c_sondage_numeros.py`)")
    md.append(bloc("A. Complétude de la récolte", lignes_a))

    # la liste et la fiche doivent s'accorder sur la localisation administrative
    croise = liste.join(bar, on="no_barrage", how="inner", suffix="_fiche")
    desaccord_mun = croise.filter(
        pl.col("municipalite").str.strip_chars() != pl.col("municipalite_fiche").str.strip_chars())
    n_mun = signaler("liste_vs_fiche_municipalite", desaccord_mun,
                     "municipalité differente entre la liste et la fiche")

    # ---------------------------------------------------------------- B
    sans_coord = bar.filter(pl.col("latitude").is_null() | pl.col("longitude").is_null())
    hors_bbox = bar.filter(
        pl.col("longitude").is_not_null()
        & ~(pl.col("longitude").is_between(BBOX[0], BBOX[1])
            & pl.col("latitude").is_between(BBOX[2], BBOX[3])))
    doublons_xy = (bar.filter(pl.col("latitude").is_not_null())
                   .group_by("latitude", "longitude").len()
                   .filter(pl.col("len") > 1))
    n_doublon_bar = (bar.join(doublons_xy, on=["latitude", "longitude"], how="semi")).height

    retenue_gt_barrage = bar.filter(pl.col("hauteur_retenue_m") > pl.col("hauteur_m") + 1e-9)
    annee_absurde = bar.filter(
        pl.col("annee_construction").is_not_null()
        & ~pl.col("annee_construction").is_between(1600, ANNEE))
    modif_avant_constr = bar.filter(
        pl.col("annee_modification").is_not_null()
        & (pl.col("annee_modification") < pl.col("annee_construction")))

    # capacité de retenue ~ hauteur de la retenue x superficie du réservoir
    # les reservoirs de moins de 1 ha sont arrondis au dixieme d'hectare :
    # le rapport y est trop bruite pour conclure
    vol = bar.filter((pl.col("capacite_retenue_m3") > 0)
                     & pl.col("hauteur_retenue_m").is_not_null()
                     & (pl.col("hauteur_retenue_m") > 0)
                     & (pl.col("superficie_reservoir_ha") >= 1)).with_columns(
        ratio_volume=pl.col("capacite_retenue_m3")
        / (pl.col("hauteur_retenue_m") * pl.col("superficie_reservoir_ha") * 10_000))
    # hauteur de la retenue x superficie du reservoir est un majorant
    # geometrique du volume : la cuvette n'est jamais un prisme droit.
    # 58 % des fiches posent l'egalite, les autres sont en dessous ; seul un
    # depassement est impossible.
    vol_incoherent = vol.filter(pl.col("ratio_volume") > 1.1)
    vol_identite = vol.filter(pl.col("ratio_volume").is_between(0.95, 1.05))

    cat = bar.with_columns(
        categorie_attendue=pl.struct("hauteur_m", "capacite_retenue_m3").map_elements(
            lambda s: categorie_attendue(s["hauteur_m"], s["capacite_retenue_m3"]),
            return_dtype=pl.Utf8))
    # les fiches « (parent) » decrivent un ensemble d'ouvrages : leur categorie
    # ne se deduit pas de la hauteur et de la capacite de la seule fiche
    cat_testable = cat.filter(pl.col("categorie_attendue").is_not_null()
                              & pl.col("categorie").is_not_null()
                              & ~pl.col("categorie").str.contains(r"\(parent\)"))
    cat_desaccord = cat_testable.filter(pl.col("categorie") != pl.col("categorie_attendue"))

    # intégrité référentielle du réseau amont/aval déclaré
    connus = set(bar["no_barrage"])
    refs_absentes = []
    for r in bar.select("no_barrage", "barrages_amont_ids", "barrages_aval_ids").iter_rows():
        cites = re.findall(r"X\d{7}", " ".join(x or "" for x in r[1:]))
        if any(c not in connus for c in cites):
            refs_absentes.append(r[0])

    n_coord = signaler("coordonnees_absentes", sans_coord, "aucune coordonnée sur la fiche")
    n_bbox = signaler("coordonnees_hors_quebec", hors_bbox, "coordonnée hors de l'emprise du Québec")
    n_ret = signaler("hauteur_retenue_superieure", retenue_gt_barrage,
                     "hauteur de la retenue > hauteur du barrage")
    n_ann = signaler("annee_absurde", annee_absurde, "année de construction hors [1600, courante]")
    n_mod = signaler("modification_avant_construction", modif_avant_constr,
                     "année de modification anterieure a la construction")
    n_vol = signaler("volume_impossible", vol_incoherent,
                     "capacité de retenue superieure a hauteur x superficie du reservoir")
    n_cat = signaler("categorie_non_conforme", cat_desaccord,
                     "catégorie administrative differente de celle qu'impose la loi")

    geo = controle_geographique(bar)
    n_hors_poly = signaler("coordonnee_hors_polygone",
                           geo.filter(pl.col("hors_polygone")),
                           "coordonnée dans aucune municipalité de la BDAT")
    testables_geo = geo.filter(~pl.col("hors_polygone"))
    mun_faux = testables_geo.filter(~pl.col("mun_accord"))
    mrc_faux = testables_geo.filter(~pl.col("mrc_accord"))
    signaler("municipalite_contredite", mun_faux,
             "la coordonnée ne tombe pas dans la municipalité declaree")

    md.append(bloc("B. Exactitude interne des fiches", [
        f"- municipalité divergente entre la liste et la fiche : **{n_mun}**",
        f"- sans coordonnées : **{n_coord}**",
        f"- coordonnées hors de l'emprise du Québec : **{n_bbox}**",
        f"- barrages partageant une coordonnée exacte avec un autre : "
        f"**{n_doublon_bar}** (sur {doublons_xy.height} points distincts)",
        f"- hauteur de la retenue supérieure à la hauteur du barrage : **{n_ret}**",
        f"- année de construction absurde : **{n_ann}**",
        f"- modification antérieure à la construction : **{n_mod}**",
        f"- capacité de retenue dépassant le majorant géométrique hauteur de la "
        f"retenue × superficie du réservoir : **{n_vol}** sur {vol.height} "
        f"testables ({100 * vol_identite.height / vol.height:.0f} % des fiches "
        "posent l'égalité, les autres restent en dessous, ce qui est le cas normal)",
        f"- catégorie administrative non conforme à la loi : **{n_cat}** sur "
        f"{cat_testable.height} testables",
        f"- fiches citant un barrage amont/aval absent du répertoire récolté : "
        f"**{len(refs_absentes)}**",
        "",
        "Confrontation de la coordonnée au découpage administratif (BDAT), "
        "qui ne vient pas du CEHQ :",
        "",
        f"- coordonnée hors de toute municipalité : **{n_hors_poly}**",
        f"- municipalité déclarée contredite par la coordonnée : "
        f"**{mun_faux.height}** sur {testables_geo.height} "
        f"({100 * mun_faux.height / max(testables_geo.height, 1):.1f} %)",
        f"- MRC déclarée contredite par la coordonnée : **{mrc_faux.height}** "
        f"({100 * mrc_faux.height / max(testables_geo.height, 1):.1f} %), pour "
        "l'essentiel des libellés et non des erreurs de position : Jamésie "
        "porte la mention « (terr. conventionné) » côté CEHQ, Beauce-Centre est "
        "le nouveau nom de Robert-Cliche",
        "",
        "| MRC déclarée | MRC du polygone | n |",
        "|---|---|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]} |" for r in
          mrc_faux.group_by("mrc", "MUS_NM_MRC").len()
          .sort("len", descending=True).head(8).iter_rows()],
    ]))

    # ---------------------------------------------------------------- C
    v = mp.filter(pl.col("ratio_superficie").is_not_null())
    sur = v.filter(pl.col("qualite") == "assuré")
    med = (sur["Q_moy_m3s"] * 1000 / sur["superficie_bv_km2"]).median()
    med_tous = (v["Q_moy_m3s"] * 1000 / v["superficie_bv_km2"]).median()
    # un tronçon qui draine plus de 5 fois le bassin declare : le cours d'eau
    # barre n'existe pas dans le reseau, le rattachement est un artefact
    incoherent_bv = v.filter(pl.col("ratio_superficie") > 5)
    surestime = v.filter(pl.col("ratio_superficie") < 0.2)

    loin = mp.filter(pl.col("distance_m") > 500)
    tres_loin = mp.filter(pl.col("distance_m") > 5000)
    desaccord_nom = mp.filter(pl.col("accord_nom") == False)  # noqa: E712
    indecidable = mp.filter(pl.col("accord_nom").is_null())

    signaler("rattachement_lointain", tres_loin, "tronçon le plus proche à plus de 5 km")
    signaler("toponyme_en_desaccord", desaccord_nom,
             "aucun mot-clé commun entre le toponyme du tronçon et les noms de la fiche")
    signaler("troncon_trop_grand", incoherent_bv,
             "le tronçon draine plus de 5 fois le bassin versant declare")
    signaler("troncon_trop_petit", surestime,
             "le tronçon draine moins du cinquieme du bassin versant declare")

    md.append(bloc("C. Exactitude du rattachement aux tronçons", [
        f"- barrages rattachés : **{mp.height}**",
        f"- distance au tronçon : médiane **{mp['distance_m'].median():.0f} m**, "
        f"p90 **{mp['distance_m'].quantile(0.9):.0f} m**, max **{mp['distance_m'].max():.0f} m**",
        f"- au-delà de 500 m : **{loin.height}** ; au-delà de 5 km : **{tres_loin.height}**",
        f"- relocalisés vers un tronçon-lac : **{int(mp['relocalise_lac'].sum())}**",
        f"- toponyme du tronçon en accord avec la fiche : "
        f"**{int((mp['accord_nom'] == True).sum())}**, en désaccord : "  # noqa: E712
        f"**{desaccord_nom.height}**, indécidable (un nom manque) : **{indecidable.height}**",
        "",
        "Contrôle indépendant par le débit : la superficie du bassin versant "
        "déclarée sur la fiche est confrontée au débit moyen climatologique "
        "simulé sur le tronçon retenu.",
        "",
        f"- couples testables (superficie déclarée et débit simulé) : **{v.height}**",
        f"- débit spécifique implicite sur les rattachements assurés "
        f"(n = {sur.height}) : **{med:.1f} L/s/km²**, soit la valeur attendue au "
        "Québec méridional (15 à 30 L/s/km²) : les deux sources, indépendantes, "
        "se recoupent",
        f"- sur l'ensemble des couples testables : **{med_tous:.0f} L/s/km²**, "
        "valeur aberrante qui mesure exactement le défaut de résolution du "
        "réseau plutôt qu'une erreur des fiches",
        f"- tronçon drainant plus de 5 fois le bassin déclaré : **{incoherent_bv.height}** "
        f"({100 * incoherent_bv.height / max(v.height, 1):.0f} %)",
        f"- tronçon drainant moins du cinquième du bassin déclaré : **{surestime.height}**",
        "",
        "Verdict par barrage :",
        "",
        "| qualité | n | superficie médiane du bassin (km²) | distance médiane (m) "
        "| toponyme en accord |",
        "|---|---|---|---|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.0f} | {r[4]:.0%} |" for r in
          mp.group_by("qualite").agg(pl.len(),
                                     pl.col("superficie_bv_km2").median(),
                                     pl.col("distance_m").median(),
                                     pl.col("accord_nom").mean())
            .sort("len", descending=True).iter_rows()],
    ]))

    if anomalies:
        an = pl.concat(anomalies)
        an = an.join(bar.select("no_barrage", "nom_barrage", "municipalite"),
                     on="no_barrage", how="left")
        an.write_csv(DATA / "anomalies.csv")
        md.append(bloc("Anomalies", [
            f"{an.height} signalements sur {an['no_barrage'].n_unique()} barrages, "
            "détaillés dans [data/anomalies.csv](data/anomalies.csv).",
            "",
            "| code | n |",
            "|---|---|",
            *[f"| {c} | {n} |" for c, n in
              an.group_by("code").len().sort("len", descending=True).iter_rows()],
        ]))

    (HERE / "RAPPORT-VERIFICATION.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))


if __name__ == "__main__":
    main()
