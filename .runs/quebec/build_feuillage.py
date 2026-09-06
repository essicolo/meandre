"""Couches feuillage du QUEBEC ENTIER : une carte, pas des regions.

Demande d'Essi (2026-08-24) : une vraie carte des troncons du Quebec (MapLibre via son
application feuillage), la repartition des parametres NeRF, les ZONES ROUGES ou les
prelevements et rejets relatifs sont importants, la repercussion vers l'aval, et le
clic sur une zone rouge qui ouvre son hydrogramme. Le decoupage regional est un
artefact de production (les caches arrivent par base de donnees) : il est FONDU ici et
n'apparait nulle part dans la carte.

Produit dans docs/carte/ :
  reseau.geojson        segments du reseau (src -> dst), proprietes = parametres NeRF
                        + debit annuel + effet des prelevements (% du renaturalise)
  zones_rouges.geojson  troncons au signal anthropique relatif le plus fort, avec les
                        cycles mensuels AVEC/SANS embarques (Chart.js au clic)
  stations.geojson      jauges, KGE de tenue de cote, cycle observe/simule au clic
  config.json           l'instance feuillage complete

Les proprietes de modele viennent des caches nb-<reg>-avec/sans.npz produits par le
PILOTE (ETL_DUMP_REACH, dette #6 : le cache porte le runtime exact). Les troncons sans
cache portent la geometrie seule : la carte s'enrichit a mesure que les dumps arrivent.

    .venv/Scripts/python.exe .runs/quebec/build_feuillage.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import duckdb
import numpy as np

from meandre.utils import paths as _paths

RESULTS = f"{_paths.DATA_ROOT}/quebec/results"
FLOTTE = os.environ.get("MEANDRE_FLOTTE", f"{_paths.DATA_ROOT}/quebec/flotte")
BRAS = os.environ.get("MEANDRE_BRAS", "A-v2lr1e-4")
# SORTIE HORS DU DEPOT (architecture, remarque d'Essi 2026-08-24) : meandre produit
# des DONNEES ; feuillage est une APPLICATION qui vit dans son propre depot
# (github.com/essicolo/feuillage) et se deploie a part. L'instance = feuillage pointe
# sur ce dossier (clone local servi par http.server, ou depot de publication dedie
# type meandre-carte sur GitHub Pages). Une copie de l'app dans le depot du modele
# etait le meme travers que le vendoring de PyGMET.
SORTIE = f"{_paths.DATA_ROOT}/quebec/carte"
# L'Outaouais moyen MANQUAIT de cette liste depuis le 2026-08-24 : ses 2379 troncons
# n'ont jamais paru sur la carte, alors que ses caches existent (2026-09-06).
REGIONS = ["outv", "gasp", "mont", "sagu", "slno", "abit", "slso", "outm",
           "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "vaud"]
PARAMS_CARTE = ["krec", "K_sat_1", "k_gw", "C_f", "T_melt"]   # les 5 les plus parlants
N_ROUGES = 150   # zones rouges provinciales (pas par region)


def _plat(x):
    """Part de jours ou le debit varie de moins de 1 % : la platitude, en pourcentage."""
    if len(x) < 3:
        return float("nan")
    return 100.0 * float((np.abs(np.diff(x)) / np.maximum(x[:-1], 1e-9) < 0.01).mean())


def charge_region(reg):
    db = ".runs/slso/data/slso.duckdb" if reg == "slso" else _paths.data_path("quebec", f"{reg}.duckdb")
    if not os.path.exists(db):
        return None
    con = duckdb.connect(db, read_only=True)
    try:
        nodes = con.execute("SELECT node_idx, lon, lat, is_lake FROM nodes ORDER BY node_idx").df()
        edges = con.execute("SELECT src, dst FROM edges").df()
        stations = con.execute("SELECT station_id, node_idx FROM stations").df()
    finally:
        con.close()
    avec = sans = None
    fa, fs = f"{RESULTS}/nb-{reg}-avec.npz", f"{RESULTS}/nb-{reg}-sans.npz"
    if os.path.exists(fa):
        avec = np.load(fa, allow_pickle=True)
    if os.path.exists(fs):
        sans = np.load(fs, allow_pickle=True)
    # SERIES AUX STATIONS (2026-09-06). Les flottes provinciales deposent leurs propres
    # series sous D:/meandre-data/quebec/flotte/q-<region>-<bras>.npz, au meme format que
    # l'ancien cache nb-<region>-q.npz. MEANDRE_BRAS choisit le bras ; a defaut on retombe
    # sur l'ancien. Un rapport, une carte, doivent dire QUEL modele ils montrent.
    qd, qd_src = None, None
    for _f, _s in ((f"{FLOTTE}/q-{reg}-{BRAS}.npz", BRAS), (f"{RESULTS}/nb-{reg}-q.npz", "cache")):
        if os.path.exists(_f):
            qd, qd_src = np.load(_f, allow_pickle=True), _s
            break
    return dict(reg=reg, nodes=nodes, edges=edges, stations=stations,
                avec=avec, sans=sans, qd=qd, qd_src=qd_src)


def main():
    os.makedirs(SORTIE, exist_ok=True)
    feats_reseau, feats_rouges, feats_stations = [], [], []
    rouges_pool = []

    for reg in REGIONS:
        d = charge_region(reg)
        if d is None:
            continue
        nd = d["nodes"]
        lon = nd.lon.to_numpy(); lat = nd.lat.to_numpy()
        n = len(nd)
        avec, sans = d["avec"], d["sans"]
        props_n = {}
        if avec is not None:
            for p in PARAMS_CARTE:
                k = f"param_{p}"
                if k in avec.files:
                    v = avec[k]
                    props_n[p] = np.log10(np.clip(v, 1e-30, None)) if p in ("krec", "k_gw") else v
            props_n["q_annuel"] = avec["q_annuel"]
            # Recharge et evapotranspiration par troncon, en millimetres par an, quand le
            # pilote les a exportees (ETL_DUMP_REACH).
            for _c, _n in (("recharge_annuel", "recharge_mm_an"), ("etr_annuel", "etr_mm_an")):
                if _c in avec.files:
                    props_n[_n] = avec[_c]
            props_n["prelev_abs"] = avec["prelev_net_abs"]
            if sans is not None:
                qs = np.clip(sans["q_annuel"], 1e-6, None)
                props_n["effet_prelev_pct"] = 100.0 * (avec["q_annuel"] - sans["q_annuel"]) / qs
                # signal anthropique RELATIF : somme |prelevements| rapportee au debit
                props_n["prelev_rel_pct"] = 100.0 * avec["prelev_net_abs"] / np.clip(
                    avec["q_annuel"] * 31_557_600.0, 1e-6, None)
                # SIGNAL SUR BRUIT (definition d'Essi, 2026-09-06) : l'importance des
                # prelevements et rejets sur la VARIANCE du signal. Le signal est la
                # perturbation anthropique, debit avec moins debit renaturalise ; le bruit
                # est la variabilite naturelle du debit renaturalise. Les deux sont
                # calcules sur la serie mensuelle de chaque troncon. Sans dimension,
                # exprime en pourcentage : au-dela de 100 %, la perturbation depasse en
                # amplitude la variabilite naturelle et devient detectable partout.
                if "q_mois_serie" in avec.files and "q_mois_serie" in sans.files:
                    _a, _s = avec["q_mois_serie"], sans["q_mois_serie"]
                    if _a.shape == _s.shape:
                        _pert = (_a - _s).std(axis=0)
                        _nat = np.clip(_s.std(axis=0), 1e-9, None)
                        props_n["signal_bruit_pct"] = 100.0 * _pert / _nat

        for _, e in d["edges"].iterrows():
            s, t = int(e.src), int(e.dst)
            if not (0 <= s < n and 0 <= t < n):
                continue
            # ARETES VIRTUELLES EXCLUES (trouvees par Essi SUR la carte, 2026-08-24) :
            # les bassins cotiers independants (Gaspesie, Cote-Nord) sont raccordes par
            # PHYSITEL a un puits artificiel unique -- 167 aretes de 127 km medians vers
            # le noeud 1 de gasp. Aucun impact sur les resultats (rien d'evalue dessus,
            # aires aux stations verifiees a 1.005x par R18), mais la carte en faisait
            # des eventails d'aiguilles. Seuil 35 km : garde les convergences reelles
            # (reservoir Baskatong, 20-30 km) et coupe les raccords fictifs.
            _dkm = (((lon[t] - lon[s]) * 78.0) ** 2 + ((lat[t] - lat[s]) * 111.0) ** 2) ** 0.5
            if _dkm > 35.0:
                continue
            pr = {"reg_source": reg}
            for k, v in props_n.items():
                x = float(v[s])
                if np.isfinite(x):
                    pr[k] = round(x, 4)
            feats_reseau.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[round(float(lon[s]), 5), round(float(lat[s]), 5)],
                                             [round(float(lon[t]), 5), round(float(lat[t]), 5)]]},
                "properties": pr})

        if avec is not None and sans is not None:
            rel = props_n.get("prelev_rel_pct")
            if rel is not None:
                for i in np.argsort(np.nan_to_num(rel))[::-1][:N_ROUGES]:
                    rouges_pool.append((float(rel[i]), reg, int(i),
                                        float(lon[i]), float(lat[i]),
                                        avec["q_mensuel"][:, i], sans["q_mensuel"][:, i],
                                        float(props_n["effet_prelev_pct"][i])))

        if d["qd"] is not None:
            qd = d["qd"]
            sids = list(qd["station_ids"])
            dates = np.array(qd["dates"])
            mois = np.array([int(x[5:7]) for x in dates])
            for j, sid in enumerate(sids):
                ni = int(d["stations"].set_index("station_id").node_idx.get(str(sid), -1))
                if ni < 0 or ni >= n:
                    continue
                qo, qs_ = qd["q_obs"][:, j], qd["q_sim"][:, j]
                v = np.isfinite(qo) & np.isfinite(qs_)
                if v.sum() < 60:
                    continue
                o, si = qo[v], qs_[v]
                r = float(np.corrcoef(o, si)[0, 1])
                beta = float(si.mean() / o.mean())
                gamma = float((si.std() / si.mean()) / (o.std() / o.mean()))
                kge = 1 - float(np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))
                cyc_o = [round(float(np.nanmean(qo[mois == m])), 2) for m in range(1, 13)]
                cyc_s = [round(float(np.nanmean(qs_[mois == m])), 2) for m in range(1, 13)]
                feats_stations.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [round(float(lon[ni]), 5), round(float(lat[ni]), 5)]},
                    "properties": {"station": str(sid), "kge": round(kge, 3),
                                   "r": round(r, 3), "beta": round(beta, 3),
                                   "gamma": round(gamma, 3),
                                   "modele": d.get("qd_src") or "?",
                                   # Amplitude de l'observe rapportee a celle du residu :
                                   # une mesure de PERFORMANCE, a ne pas confondre avec le
                                   # signal sur bruit anthropique du reseau (2026-09-06).
                                   "obs_sur_residu": round(float(o.std() / max((o - si).std(), 1e-9)), 2),
                                   "plat_sim_pct": round(_plat(si), 1),
                                   "plat_obs_pct": round(_plat(o), 1),
                                   "dates": [str(x) for x in dates[v]],
                                   "q_obs": [round(float(x), 3) for x in o],
                                   "q_sim": [round(float(x), 3) for x in si],
                                   "cycle_obs": cyc_o, "cycle_sim": cyc_s}})
        print(f"  {reg}: {len(d['edges'])} segments"
              + ("" if avec is None else " + parametres/effet")
              + ("" if d["qd"] is None else f" + {len(feats_stations)} stations cumulees"))

    # zones rouges : classement PROVINCIAL, pas par region
    rouges_pool.sort(key=lambda x: -x[0])
    for rel, reg, i, lo, la, qm_a, qm_s, eff in rouges_pool[:N_ROUGES]:
        feats_rouges.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lo, 5), round(la, 5)]},
            "properties": {"prelev_rel_pct": round(rel, 2),
                           "effet_prelev_pct": round(eff, 2),
                           "troncon": i, "source": reg,
                           "q_mensuel_avec": [round(float(x), 3) for x in qm_a],
                           "q_mensuel_renature": [round(float(x), 3) for x in qm_s]}})

    for nom, feats in (("reseau", feats_reseau), ("zones_rouges", feats_rouges),
                       ("stations", feats_stations)):
        p = os.path.join(SORTIE, f"{nom}.geojson")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f,
                      separators=(",", ":"))
        print(f"{nom}.geojson : {len(feats)} entites, {os.path.getsize(p)/1e6:.1f} Mo")

    config = {
        "view": {"name": "meandre 1.0 — Québec",
                 "description": "Réseau modélisé, paramètres appris, effet des prélèvements et rejets",
                 "center": [47.5, -73.5], "zoom": 6, "basemap": "osm"},
        "layers": [
            {"name": "Réseau (effet des prélèvements, %)", "url": "reseau.geojson",
             "visible": True, "color_by": "effet_prelev_pct", "popup_template": "troncon"},
            {"name": "Zones rouges (prélèvement relatif)", "url": "zones_rouges.geojson",
             "visible": True, "color": "#d62728", "popup_template": "zone_rouge"},
            {"name": "Stations (KGE tenu de côté)", "url": "stations.geojson",
             "visible": True, "color_by": "kge", "popup_template": "station"},
            # COUCHES DEMANDEES LE 2026-09-06. Une couche par grandeur, toutes sur la meme
            # geometrie de reseau : l'utilisateur en allume une a la fois. Les couches de
            # flux (recharge, evapotranspiration) n'apparaissent que si le pilote a exporte
            # ces champs ; sinon feuillage colore en gris faute de propriete.
            {"name": "Recharge de la nappe (mm/an)", "url": "reseau.geojson",
             "visible": False, "color_by": "recharge_mm_an", "popup_template": "troncon"},
            {"name": "Évapotranspiration réelle (mm/an)", "url": "reseau.geojson",
             "visible": False, "color_by": "etr_mm_an", "popup_template": "troncon"},
            {"name": "Prélèvements et rejets (% du débit)", "url": "reseau.geojson",
             "visible": False, "color_by": "prelev_rel_pct", "popup_template": "troncon"},
            {"name": "Signal sur bruit anthropique (% de la variabilité naturelle)",
             "url": "reseau.geojson", "visible": False, "color_by": "signal_bruit_pct",
             "popup_template": "troncon"},
            {"name": "Amplitude observée sur résidu (stations)", "url": "stations.geojson",
             "visible": False, "color_by": "obs_sur_residu", "popup_template": "station"},
            {"name": "Platitude simulée aux stations (%)", "url": "stations.geojson",
             "visible": False, "color_by": "plat_sim_pct", "popup_template": "station"},
        ] + [
            {"name": f"Champ appris : {_nom}", "url": "reseau.geojson", "visible": False,
             "color_by": _cle, "popup_template": "troncon"}
            for _cle, _nom in (("K_sat_1", "conductivité de surface (m/j)"),
                               ("krec", "drainage profond, log10 (m/h)"),
                               ("k_gw", "récession de nappe, log10 (1/j)"),
                               ("C_f", "facteur de fonte (mm/°C/j)"),
                               ("T_melt", "seuil de fonte (°C)"))
        ],
        "popup_templates": {
            "troncon": {"title": "Tronçon",
                        "sections": [{"type": "properties",
                                      "fields": ["q_annuel", "recharge_mm_an", "etr_mm_an",
                                                 "effet_prelev_pct", "prelev_rel_pct",
                                                 "signal_bruit_pct",
                                                 "krec", "K_sat_1", "k_gw", "C_f", "T_melt"]}]},
            "zone_rouge": {"title": "Zone rouge — prélèvement {properties.prelev_rel_pct} % du débit",
                           "sections": [
                               {"type": "properties",
                                "fields": ["prelev_rel_pct", "effet_prelev_pct"]},
                               {"type": "chart", "chart_type": "line",
                                "data_field": "q_mensuel_avec",
                                "compare_field": "q_mensuel_renature",
                                "options": {"title": "Débit mensuel : avec prélèvements vs renaturalisé",
                                            "xlabel": "Mois", "ylabel": "m³/s"}}]},
            "station": {"title": "Station {properties.station} — KGE {properties.kge}",
                        "sections": [
                            {"type": "properties",
                             "fields": ["modele", "kge", "r", "beta", "gamma", "obs_sur_residu",
                                        "plat_sim_pct", "plat_obs_pct"]},
                            # Hydrogramme journalier embarque dans la station : simule et
                            # observe sur la periode d'evaluation, sans dependance a un
                            # magasin zarr distant.
                            {"type": "chart", "chart_type": "line",
                             "data_field": "q_sim", "compare_field": "q_obs",
                             "x_field": "dates",
                             "options": {"title": "Débit journalier simulé et observé, "
                                                  "période d'évaluation",
                                         "xlabel": "Date", "ylabel": "m³/s"}},
                            {"type": "chart", "chart_type": "line",
                             "data_source": {"type": "zarr", "store_url": "hydro.zarr",
                                             "value_array": "discharge",
                                             "feature_dim": "station_id",
                                             "feature_id_property": "station",
                                             "indexers": {"percentile": 50},
                                             "time_array": "time"},
                             "options": {"title": "Hydrogramme simulé 2022-2024 (médiane ; "
                                                  "changer l'indexer percentile pour "
                                                  "l'enveloppe : 5-95 tête quantile, "
                                                  "0/100 bornes de forçage)",
                                         "xlabel": "Date", "ylabel": "m³/s"}},
                            {"type": "chart", "chart_type": "line",
                             "data_source": {"type": "zarr", "store_url": "hydro.zarr",
                                             "value_array": "observed",
                                             "feature_dim": "station_id",
                                             "feature_id_property": "station",
                                             "time_array": "time"},
                             "options": {"title": "Observé (drapeaux CEHQ : déc-mars "
                                                  "majoritairement reconstruit)",
                                         "xlabel": "Date", "ylabel": "m³/s"}},
                            {"type": "chart", "chart_type": "line",
                             "data_field": "cycle_sim", "compare_field": "cycle_obs",
                             "options": {"title": "Cycle mensuel simulé vs observé",
                                         "xlabel": "Mois", "ylabel": "m³/s"}}]},
        },
    }
    with open(os.path.join(SORTIE, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"config.json ecrit dans {SORTIE}.")
    print("Instance : servir ce dossier et pointer l'index.html d'un clone de "
          "feuillage dessus (?config=.../config.json), ou publier le dossier dans "
          "un depot meandre-carte a cote de l'app.")


if __name__ == "__main__":
    main()
