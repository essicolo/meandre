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
SORTIE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "docs", "carte")
REGIONS = ["outv", "gasp", "mont", "sagu", "slno", "abit", "slso",
           "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "vaud"]
PARAMS_CARTE = ["krec", "K_sat_1", "k_gw", "C_f", "T_melt"]   # les 5 les plus parlants
N_ROUGES = 150   # zones rouges provinciales (pas par region)


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
    qd = None
    fq = f"{RESULTS}/nb-{reg}-q.npz"
    if os.path.exists(fq):
        qd = np.load(fq, allow_pickle=True)
    return dict(reg=reg, nodes=nodes, edges=edges, stations=stations,
                avec=avec, sans=sans, qd=qd)


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
            props_n["prelev_abs"] = avec["prelev_net_abs"]
            if sans is not None:
                qs = np.clip(sans["q_annuel"], 1e-6, None)
                props_n["effet_prelev_pct"] = 100.0 * (avec["q_annuel"] - sans["q_annuel"]) / qs
                # signal anthropique RELATIF : somme |prelevements| rapportee au debit
                props_n["prelev_rel_pct"] = 100.0 * avec["prelev_net_abs"] / np.clip(
                    avec["q_annuel"] * 31_557_600.0, 1e-6, None)

        for _, e in d["edges"].iterrows():
            s, t = int(e.src), int(e.dst)
            if not (0 <= s < n and 0 <= t < n):
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
        ],
        "popup_templates": {
            "troncon": {"title": "Tronçon",
                        "sections": [{"type": "properties",
                                      "fields": ["q_annuel", "effet_prelev_pct", "prelev_rel_pct",
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
                            {"type": "properties", "fields": ["kge", "r", "beta", "gamma"]},
                            {"type": "chart", "chart_type": "line",
                             "data_field": "cycle_sim", "compare_field": "cycle_obs",
                             "options": {"title": "Cycle mensuel simulé vs observé",
                                         "xlabel": "Mois", "ylabel": "m³/s"}}]},
        },
    }
    with open(os.path.join(SORTIE, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"config.json ecrit. Servir docs/carte/ (ex. python -m http.server) et ouvrir index.html")


if __name__ == "__main__":
    main()
