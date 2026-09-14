"""Analyse les fiches techniques mises en cache et écrit une table plate.

Sortie : barrages/data/barrages.parquet (une ligne par barrage)
         barrages/data/barrages-hydrographie.parquet (une ligne par entrée
         hydrographique : lac, cours d'eau, bassin)
"""
import re
import html as _html
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")

# étiquette telle qu'affichée -> nom de colonne
ETIQUETTES = {
    "NOM DU BARRAGE": "nom_barrage",
    "Numéro du barrage": "no_barrage",
    "Région administrative": "region_administrative",
    "Municipalité": "municipalite",
    "MRC": "mrc",
    "Coordonnées NAD83": "_coord",
    "Nom du réservoir": "nom_reservoir",
    "Territoire(s)": "territoires",
    "Aménagement(s)": "amenagements",
    "Catégorie administrative": "categorie",
    "Type(s) d'utilisation": "utilisations",
    "Hauteur du barrage": "hauteur_m",
    "Capacité de retenue": "capacite_retenue_m3",
    "Hauteur de la retenue": "hauteur_retenue_m",
    "Longueur de l'ouvrage": "longueur_ouvrage_m",
    "Type de barrage": "type_barrage",
    "Type de terrain de fondation": "type_fondation",
    "Classe": "classe",
    "Niveau des conséquences": "niveau_consequences",
    "Zone sismique": "zone_sismique",
    "Superficie du réservoir": "superficie_reservoir_ha",
    "Superficie du bassin versant": "superficie_bv_km2",
    "Longueur de refoulement": "longueur_refoulement_m",
    "Année de construction": "annee_construction",
    "Année de modification": "annee_modification",
    "Barrage(s) en aval": "barrages_aval",
    "Barrage(s) en amont": "barrages_amont",
    "Nom": "proprietaire",
    "Adresse": "proprietaire_adresse",
    "Code postal": "proprietaire_code_postal",
}
# le tiret cadratin du site signifie « renseignement absent »
VIDE = {"", "—", "-", "---", "&"}

NUM = {"hauteur_m", "capacite_retenue_m3", "hauteur_retenue_m", "longueur_ouvrage_m",
       "superficie_reservoir_ha", "superficie_bv_km2", "longueur_refoulement_m",
       "annee_construction", "annee_modification", "zone_sismique"}


def nettoyer(h):
    h = re.sub(r"(?s)<!--.*?-->", " ", h)
    h = re.sub(r"(?s)<em>.*?</em>", " ", h)      # infobulles
    h = re.sub(r"(?s)<(script|style).*?</\1>", " ", h)
    # la note de bas de page (« le tiret indique que... ») suit la dernière
    # section et serait aspirée comme valeur du dernier champ
    coupe = h.find("le tiret indique que")
    return h[:coupe] if coupe > 0 else h


def jetons(bloc):
    """Texte des noeuds, dans l'ordre, entités résolues, vides retirés."""
    out = []
    for t in re.split(r"<[^>]+>", bloc):
        t = _html.unescape(t).replace("\xa0", " ")
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.append(t)
    return out


def decouper_sections(h):
    """{titre de section: html} d'après les <div class="separateur">."""
    marques = [(m.start(), m.end(), _html.unescape(m.group(1)).strip())
               for m in re.finditer(r'<div class="separateur">(.*?)</div>', h)]
    res = {}
    for i, (_, fin, titre) in enumerate(marques):
        debut_suivant = marques[i + 1][0] if i + 1 < len(marques) else len(h)
        res[titre] = h[fin:debut_suivant]
    return res


def num(v):
    if v is None:
        return None
    v = str(v).replace("\xa0", "").replace(" ", "").replace(",", ".")
    v = re.sub(r"[^0-9.\-]", "", v)
    try:
        return float(v)
    except ValueError:
        return None


def dms(txt):
    """« 47o 23' 14" » -> 47.3872 ; le signe porte sur les degrés."""
    m = re.search(r"(-?\d+)\D+?(\d+)'\s*(\d+(?:[.,]\d+)?)", txt)
    if not m:
        return None
    d, mi, se = m.group(1), float(m.group(2)), float(m.group(3).replace(",", "."))
    signe = -1.0 if d.startswith("-") else 1.0
    return signe * (abs(float(d)) + mi / 60 + se / 3600)


def paires(bloc):
    """Associe chaque étiquette connue aux jetons qui la suivent."""
    etiq = None
    res = {}
    for t in jetons(bloc):
        cle = t.rstrip(" :").strip()
        if cle in ETIQUETTES:
            etiq = ETIQUETTES[cle]
            res.setdefault(etiq, [])
            continue
        if t == ":" or etiq is None:
            continue
        if t.strip() not in VIDE:
            res[etiq].append(t.rstrip(" :").strip())
    return res


def parse_hydro(bloc):
    """Les lignes du tableau HYDROGRAPHIE : 5 colonnes positionnelles."""
    lignes = []
    blocs = re.split(r'<div class="affichage_section">', bloc)[1:]
    for div in blocs:
        cols = [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                for c in re.findall(r'(?s)<div style="float:left;width:\d+%">(.*?)</div>', div)]
        if len(cols) < 5 or cols[0] in ("Type", ""):
            continue
        lignes.append({"type": cols[0], "numero": cols[1] or None, "nom": cols[2] or None,
                       "numero_bassin_primaire": cols[3] or None,
                       "nom_bassin_primaire": cols[4] or None})
    return lignes


def parse_fiche(path):
    h = nettoyer(path.read_text(encoding="cp1252", errors="replace"))
    secs = decouper_sections(h)
    d = {"no_barrage": path.stem}
    for titre, bloc in secs.items():
        if titre.startswith("HYDROGRAPHIE"):
            continue
        for k, vals in paires(bloc).items():
            if k == "_coord":
                joint = " ".join(vals)
                avant = re.split(r"(?i)longitude", joint)[0]
                apres = re.search(r"(?i)longitude\s*:?(.*)", joint)
                d["latitude"] = dms(avant)
                d["longitude"] = dms(apres.group(1)) if apres else None
                continue
            if not vals:
                d.setdefault(k, None)
            elif k in ("utilisations", "territoires", "amenagements",
                       "barrages_aval", "barrages_amont"):
                d[k] = " | ".join(vals)
            else:
                d[k] = vals[0]
    d.pop("_coord", None)
    for k in NUM:
        d[k] = num(d.get(k))
    hydro = parse_hydro(secs.get("HYDROGRAPHIE", ""))
    for t, pref in (("Lac", "lac"), ("Cours d'eau", "cours_eau"), ("Bassin", "bassin")):
        r = next((x for x in hydro if x["type"] == t), None)
        d[f"{pref}_numero"] = r["numero"] if r else None
        d[f"{pref}_nom"] = r["nom"] if r else None
    for k in ("barrages_amont", "barrages_aval"):
        d[f"{k}_ids"] = " | ".join(re.findall(r"X\d{7}", d.get(k) or "")) or None
    r = next((x for x in hydro if x["type"] == "Bassin"), None) or (hydro[0] if hydro else None)
    d["bassin_primaire_numero"] = r["numero_bassin_primaire"] if r else None
    d["bassin_primaire_nom"] = r["nom_bassin_primaire"] if r else None
    return d, [dict(no_barrage=path.stem, **x) for x in hydro]


COLS = ["no_barrage", "nom_barrage", "region_administrative", "municipalite", "mrc",
        "latitude", "longitude", "nom_reservoir", "territoires", "amenagements",
        "categorie", "utilisations", "hauteur_m", "hauteur_retenue_m",
        "capacite_retenue_m3", "longueur_ouvrage_m", "type_barrage", "type_fondation",
        "classe", "niveau_consequences", "zone_sismique", "superficie_reservoir_ha",
        "superficie_bv_km2", "longueur_refoulement_m", "annee_construction",
        "annee_modification", "barrages_amont", "barrages_amont_ids",
        "barrages_aval", "barrages_aval_ids",
        "lac_numero", "lac_nom", "cours_eau_numero", "cours_eau_nom",
        "bassin_numero", "bassin_nom", "bassin_primaire_numero", "bassin_primaire_nom",
        "proprietaire", "proprietaire_adresse", "proprietaire_code_postal"]


def main():
    fiches, hydros = [], []
    for p in sorted(CACHE.glob("*.html")):
        d, hy = parse_fiche(p)
        fiches.append(d)
        hydros.extend(hy)
    df = pl.DataFrame(fiches, infer_schema_length=None)
    for c in COLS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    df = df.select(COLS)
    df.write_parquet(DATA / "barrages.parquet")
    pl.DataFrame(hydros).write_parquet(DATA / "barrages-hydrographie.parquet")
    print(f"{df.height} fiches -> data/barrages.parquet")
    print("valeurs manquantes par colonne :")
    print(df.null_count().transpose(include_header=True, column_names=["n_null"])
            .sort("n_null", descending=True))


if __name__ == "__main__":
    main()
