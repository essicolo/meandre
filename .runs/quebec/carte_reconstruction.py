"""Cartes feuillage de la reconstruction des débits et des prélèvements ponctuels.

Deux cartes pour la présentation (demande d'Essi, 2026-10-05) :

1. Les prélèvements et rejets ponctuels : un point par site de la table io-eau ingérée par
   meandre, coloré par le signe du débit net (rouge pour un prélèvement, bleu pour un
   rejet), d'aire proportionnelle à son débit moyen.
2. La reconstruction des débits : les tronçons du réseau PHYSITEL, colorés par l'influence
   relative des prélèvements et rejets sur le débit d'étiage. Le clic ouvre l'hydrogramme
   journalier modélisé (avec prélèvements et rejets) et naturalisé (sans), selon les
   couches allumées.

Entrées : les caches du pilote produits par ETL_DUMP_REACH et ETL_DUMP_NATUREL=1, soit
<reg>-journalier.npz et <reg>-sans-journalier.npz dans MEANDRE_CARTE_DUMPS. Les territoires
sans cache sont omis : la carte s'étend à mesure que les simulations arrivent.

Sortie : MEANDRE_CARTE_SORTIE (défaut DATA_ROOT/quebec/carte-reconstruction), un dossier
autonome qui porte une copie de l'application feuillage : le servir et ouvrir index.html.

    MEANDRE_CARTE_DUMPS=D:/meandre-data/carte-essai uv run python .runs/quebec/carte_reconstruction.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import duckdb
import numpy as np
import pandas as pd
import zarr

from meandre.utils import paths as _paths

DUMPS = os.environ.get("MEANDRE_CARTE_DUMPS", f"{_paths.DATA_ROOT}/carte-essai")
SORTIE = os.environ.get("MEANDRE_CARTE_SORTIE", f"{_paths.DATA_ROOT}/quebec/carte-reconstruction")
_DEPOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO_EAU = os.environ.get("MEANDRE_IO_EAU", os.path.join(os.path.dirname(_DEPOT), "io-eau", "data", "derived"))
# Application copiée dans le dossier de sortie : le dossier se sert seul, sans dépendre
# d'un clone de feuillage à côté.
FEUILLAGE = os.environ.get("MEANDRE_FEUILLAGE", os.path.join(os.path.dirname(_DEPOT), "feuillage", "index.html"))
# Période des hydrogrammes et des indicateurs, en années civiles complètes.
DEBUT = os.environ.get("MEANDRE_CARTE_DEBUT", "2015-01-01")
FIN = os.environ.get("MEANDRE_CARTE_FIN", "2024-12-31")
# Plancher du débit naturalisé d'étiage, en m³/s, sous lequel l'influence relative n'est pas
# calculée : un rapport à un débit quasi nul n'a pas de sens physique.
Q_PLANCHER = float(os.environ.get("MEANDRE_CARTE_Q_PLANCHER", "0.01"))
REGIONS = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont", "outm",
           "outv", "sagu", "slno", "slso", "vaud"]
# Rouge pour une perte d'eau, bleu pour un apport, gris neutre au centre : le blanc des
# palettes divergentes usuelles disparaît sur le fond de carte.
DIVERGENTE = ["#b2182b", "#ef8a62", "#bdbdbd", "#67a9cf", "#2166ac"]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def io_eau_dir():
    if os.path.exists(os.path.join(IO_EAU, "io-eau-meandre.parquet")):
        return IO_EAU
    raise FileNotFoundError(f"table io-eau-meandre.parquet introuvable dans {IO_EAU} (MEANDRE_IO_EAU)")


def q7min_annuel(q, dates):
    """Moyenne sur les années du minimum annuel du débit moyen sur sept jours, par tronçon."""
    q7 = pd.DataFrame(q, index=dates).rolling(7, min_periods=7).mean()
    return q7.groupby(q7.index.year).min().mean(axis=0).to_numpy()


def sites_ponctuels(d):
    """Un point par site : débit net moyen et estival sur la période, nom et nature."""
    p = os.path.join(d, "io-eau-meandre.parquet")
    df = duckdb.sql(f"""
        SELECT site_id, any_value(lon) AS lon, any_value(lat) AS lat, any_value(source) AS source,
               any_value(IDTRONCON) AS troncon, date, sum(net_withdrawal) AS q
        FROM '{p}' WHERE date BETWEEN '{DEBUT}' AND '{FIN}'
        GROUP BY site_id, date""").df()
    df["mois"] = pd.to_datetime(df.date).dt.month
    noms = {}
    for f, nature in (("prelevements", "prélèvement"), ("rejets", "rejet")):
        for ft in json.load(open(os.path.join(d, f + ".geojson"), encoding="utf-8"))["features"]:
            pr = ft["properties"]
            noms[pr["id"]] = {"nom": pr.get("nom"), "secteur": pr.get("secteur"),
                              "municipalite": pr.get("municipalite"), "nature_declaree": nature}
    feats = []
    for sid, g in df.groupby("site_id"):
        base = sid.replace("_synth", "")
        info = noms.get(base, {})
        moy = float(g.q.mean())
        ete = float(g[g.mois.isin([7, 8, 9])].q.mean())
        profil = g.groupby("mois").q.mean().reindex(range(1, 13))
        pr = {
            "site": sid,
            "nom": info.get("nom") or sid,
            "municipalite": info.get("municipalite"),
            "secteur": info.get("secteur"),
            "source": g.source.iloc[0],
            "troncon": g.troncon.iloc[0],
            "nature": "rejet" if moy > 0 else ("prélèvement" if moy < 0 else "débit net nul"),
            "origine": "retour estimé" if sid.endswith("_synth") else "déclaré",
            "debit_net_l_s": round(1000.0 * moy, 2),
            "debit_net_ete_l_s": round(1000.0 * ete, 2) if np.isfinite(ete) else None,
            "profil_mensuel": json.dumps({"labels": MOIS,
                                          "values": [round(1000.0 * float(v), 2) if np.isfinite(v) else None
                                                     for v in profil]}, ensure_ascii=False),
        }
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point",
                                   "coordinates": [round(float(g.lon.iloc[0]), 5), round(float(g.lat.iloc[0]), 5)]},
                      "properties": pr})
    # Les petits sites au-dessus des gros : l'ordre des entités fixe l'ordre de dessin.
    feats.sort(key=lambda f: -abs(f["properties"]["debit_net_l_s"]))
    return feats


def charge_territoire(reg):
    # Le débit seul, extrait sur la grappe, ou le fichier journalier complet du pilote.
    fa = os.path.join(DUMPS, f"{reg}-q-journalier.npz")
    if not os.path.exists(fa):
        fa = os.path.join(DUMPS, f"{reg}-journalier.npz")
    fs = os.path.join(DUMPS, f"{reg}-sans-journalier.npz")
    if not (os.path.exists(fa) and os.path.exists(fs)):
        return None
    a, s = np.load(fa), np.load(fs)
    dates = pd.DatetimeIndex(a["dates"])
    garde = (dates >= DEBUT) & (dates <= FIN)
    db = _paths.data_path("quebec", f"{reg}.duckdb")
    con = duckdb.connect(db, read_only=True)
    try:
        nodes = con.execute("SELECT node_idx, node_id, lon, lat, is_lake FROM nodes ORDER BY node_idx").df()
    finally:
        con.close()
    return dict(reg=reg, dates=dates[garde], q=a["q"][garde], qn=s["q"][garde],
                ids=[f"{reg.upper()}{int(n):05d}" for n in nodes.node_id],
                lon=nodes.lon.to_numpy(), lat=nodes.lat.to_numpy(), lac=nodes.is_lake.to_numpy())


def main():
    os.makedirs(SORTIE, exist_ok=True)
    d_io = io_eau_dir()
    if os.path.exists(FEUILLAGE):
        shutil.copyfile(FEUILLAGE, os.path.join(SORTIE, "index.html"))

    feats_sites = sites_ponctuels(d_io)
    with open(os.path.join(SORTIE, "prelevements_rejets.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats_sites}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"prelevements_rejets.geojson : {len(feats_sites)} sites")

    geom = {}
    for ft in json.load(open(os.path.join(d_io, "_gj_troncons.json"), encoding="utf-8"))["features"]:
        geom[ft["properties"]["id"]] = (ft["properties"].get("nom"), ft["geometry"])

    terr = [t for t in (charge_territoire(r) for r in REGIONS) if t is not None]
    if not terr:
        print(f"aucun cache journalier dans {DUMPS} : carte des tronçons non produite")
        return
    dates = terr[0]["dates"]
    for t in terr:
        if not t["dates"].equals(dates):
            raise ValueError(f"{t['reg']} : axe temporel différent de {terr[0]['reg']}")

    feats_tr, series = [], []
    for t in terr:
        q, qn = t["q"], t["qn"]
        q7, q7n = q7min_annuel(q, t["dates"]), q7min_annuel(qn, t["dates"])
        moy, moyn = q.mean(axis=0), qn.mean(axis=0)
        n_sans_geom = 0
        for i, tid in enumerate(t["ids"]):
            if tid in geom:
                nom, g = geom[tid]
            elif t["lac"][i]:
                # Les lacs n'ont pas de tracé de tronçon : un point à leur nœud.
                nom, g = None, {"type": "Point", "coordinates": [round(float(t["lon"][i]), 5), round(float(t["lat"][i]), 5)]}
            else:
                n_sans_geom += 1
                continue
            pr = {"troncon": tid, "zidx": len(series), "nom": nom,
                  "q_moyen_m3s": round(float(moy[i]), 4),
                  "q_nat_moyen_m3s": round(float(moyn[i]), 4),
                  "q7min_m3s": round(float(q7[i]), 4),
                  "q7min_nat_m3s": round(float(q7n[i]), 4)}
            if q7n[i] >= Q_PLANCHER:
                pr["influence_etiage_pct"] = round(100.0 * float((q7[i] - q7n[i]) / q7n[i]), 2)
            if moyn[i] >= Q_PLANCHER:
                pr["influence_annuelle_pct"] = round(100.0 * float((moy[i] - moyn[i]) / moyn[i]), 2)
            feats_tr.append({"type": "Feature", "geometry": g, "properties": pr})
            series.append((q[:, i], qn[:, i]))
        print(f"  {t['reg']} : {len(t['ids']) - n_sans_geom} tronçons"
              + (f", {n_sans_geom} sans géométrie" if n_sans_geom else ""))

    with open(os.path.join(SORTIE, "troncons.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats_tr}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"troncons.geojson : {len(feats_tr)} tronçons, "
          f"{os.path.getsize(os.path.join(SORTIE, 'troncons.geojson')) / 1e6:.1f} Mo")

    # Un chunk par tronçon et les deux séries ensemble : un clic lit un seul fichier.
    n, T = len(series), len(dates)
    arr = np.empty((2, n, T), dtype=np.float32)
    for k, (qa, qb) in enumerate(series):
        arr[0, k], arr[1, k] = qa, qb
    root = zarr.open_group(os.path.join(SORTIE, "debits.zarr"), mode="w")
    root.create_array("debit", data=arr, chunks=(2, 1, T), dimension_names=["serie", "zidx", "time"])
    root.create_array("serie", data=np.array([0, 1], dtype=np.int32), dimension_names=["serie"])
    root.create_array("zidx", data=np.arange(n, dtype=np.int32), dimension_names=["zidx"])
    tz = root.create_array("time", data=((dates - pd.Timestamp("1970-01-01")).days.to_numpy().astype(np.int32)),
                           dimension_names=["time"])
    tz.attrs["units"] = "days since 1970-01-01"
    root["serie"].attrs["description"] = "0 : modélisé avec prélèvements et rejets ; 1 : naturalisé"
    root.attrs["description"] = f"Débits journaliers simulés par meandre, m³/s, {DEBUT[:4]}-{FIN[:4]}"
    zarr.consolidate_metadata(root.store)
    print(f"debits.zarr : {n} tronçons x {T} jours")

    ecrit_config(feats_sites, feats_tr)


def ecrit_config(feats_sites, feats_tr):
    infl = np.array([f["properties"].get("influence_etiage_pct", np.nan) for f in feats_tr], dtype=float)
    borne = float(np.nanpercentile(np.abs(infl), 98)) if np.isfinite(infl).any() else 10.0
    borne = max(1.0, min(50.0, round(borne)))
    nom_mod = "Débit modélisé, avec prélèvements et rejets"
    nom_nat = "Débit naturalisé"
    nom_inf = "Influence des prélèvements et rejets sur l'étiage (%)"
    graphique = {
        "type": "chart", "chart_type": "line",
        "data_source": {"type": "zarr", "store_url": "debits.zarr", "value_array": "debit",
                        "feature_dim": "zidx", "feature_id_property": "zidx", "time_array": "time",
                        "series": [{"name": "modélisé", "role": "main", "layer": nom_mod,
                                    "indexers": {"serie": 0}},
                                   {"name": "naturalisé", "role": "main", "layer": nom_nat,
                                    "indexers": {"serie": 1}}]},
        "options": {"title": f"Débit journalier, {DEBUT[:4]}-{FIN[:4]}, échelle logarithmique", "xlabel": "Date",
                    "ylabel": "m³/s", "ylog": True}}
    fiche_troncon = {"title": "Tronçon {properties.troncon} {properties.nom}", "display": "sidebar",
                     "sections": [graphique,
                                  {"type": "properties",
                                   "fields": ["q_moyen_m3s", "q_nat_moyen_m3s", "q7min_m3s", "q7min_nat_m3s",
                                              "influence_etiage_pct", "influence_annuelle_pct"],
                                   "labels": {"q_moyen_m3s": "Débit moyen modélisé (m³/s)",
                                              "q_nat_moyen_m3s": "Débit moyen naturalisé (m³/s)",
                                              "q7min_m3s": "Étiage modélisé, Q7 min. (m³/s)",
                                              "q7min_nat_m3s": "Étiage naturalisé, Q7 min. (m³/s)",
                                              "influence_etiage_pct": "Écart de l'étiage (%)",
                                              "influence_annuelle_pct": "Écart du débit moyen (%)"}}]}
    largeur = {"property": "q_nat_moyen_m3s", "range": [0.4, 6]}
    taille_lac = {"property": "q_nat_moyen_m3s", "range": [1.5, 6]}
    config = {
        "view": {"name": "meandre : débits reconstruits",
                 "description": "Débits modélisés et naturalisés, prélèvements et rejets ponctuels",
                 "center": [47.0, -72.5], "zoom": 5.5, "basemap": "osm"},
        "layers": [
            {"name": nom_inf, "url": "troncons.geojson", "visible": True, "popup_template": "troncon",
             "color_by": {"property": "influence_etiage_pct", "colors": DIVERGENTE,
                          "domain": [-borne, borne], "label": "écart du débit d'étiage, %"},
             "width_by": largeur, "size_by": taille_lac},
            {"name": nom_mod, "url": "troncons.geojson", "visible": False, "popup_template": "troncon",
             "color": "#2563eb", "width_by": largeur, "size_by": taille_lac},
            {"name": nom_nat, "url": "troncons.geojson", "visible": False, "popup_template": "troncon",
             "color": "#d97706", "width_by": largeur, "size_by": taille_lac},
            {"name": "Prélèvements et rejets ponctuels", "url": "prelevements_rejets.geojson",
             "visible": True, "popup_template": "site",
             "color_by": {"property": "debit_net_l_s", "colors": DIVERGENTE,
                          "domain": [-5, 5], "label": "débit net, L/s (rouge : prélèvement)"},
             "size_by": {"property": "debit_net_l_s", "range": [1.5, 14]}},
        ],
        "popup_templates": {
            "troncon": fiche_troncon,
            "site": {"title": "{properties.nom}",
                     "sections": [{"type": "properties",
                                   "fields": ["nature", "origine", "source", "secteur", "municipalite",
                                              "debit_net_l_s", "debit_net_ete_l_s", "troncon"],
                                   "labels": {"nature": "Nature", "origine": "Origine", "source": "Milieu",
                                              "secteur": "Secteur", "municipalite": "Municipalité",
                                              "debit_net_l_s": "Débit net moyen (L/s)",
                                              "debit_net_ete_l_s": "Débit net, juil. à sept. (L/s)",
                                              "troncon": "Tronçon"}},
                                  {"type": "chart", "chart_type": "bar", "data_field": "profil_mensuel",
                                   "options": {"title": f"Débit net moyen par mois, {DEBUT[:4]}-{FIN[:4]}",
                                               "xlabel": "Mois", "ylabel": "L/s"}}]},
        },
    }
    # Deux cartes de présentation sur les mêmes données : la reconstruction des débits, les
    # sites masqués au départ, et les sites seuls.
    sites = config["layers"][-1]
    prelev = {"view": {**config["view"], "name": "meandre : prélèvements et rejets ponctuels",
                       "description": "Débit net moyen par site, 2015-2024, table io-eau"},
              "layers": [sites], "popup_templates": {"site": config["popup_templates"]["site"]}}
    config["layers"][-1] = {**sites, "visible": False}
    for nom, c in (("config.json", config), ("config-prelevements.json", prelev)):
        with open(os.path.join(SORTIE, nom), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
    print(f"config.json et config-prelevements.json écrits dans {SORTIE} "
          f"(influence en étiage bornée à ±{borne:.0f} %)")


if __name__ == "__main__":
    main()
