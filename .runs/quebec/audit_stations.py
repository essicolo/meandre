"""Audit du rattachement station-tronçon, par la géographie du modèle lui-même (2026-10-28).

Le rattachement des stations aux tronçons est hérité des projets Hydrotel, et 9 stations sur 178
sont à plus d'un facteur 1,5 de l'aire de leur tronçon (registre R429). Cet audit le refait de
façon systématique, sans autre réseau que celui de PHYSITEL :

1. coordonnées exactes de la station, lues dans l'en-tête de sa fiche CEHQ (mises en cache) ;
2. unité hydrologique (UHRH) qui contient ce point, dans `uhrh.shp` du projet ;
3. tronçon où cette unité se draine, lu dans `troncon.trl` ;
4. descente vers l'aval tant que l'aire drainée du tronçon reste sous celle de la station
   (moins une tolérance) : une station au bas d'une unité peut se trouver après la confluence ;
5. choix final parmi le tronçon trouvé, son voisin aval et ses voisins amont : celui dont l'aire
   est la plus proche de celle de la station. Une station posée sur le chenal près d'un noeud tombe
   dans l'unité d'un tronçon ou de son voisin selon la grille de PHYSITEL ; l'aire tranche. Le
   rattachement d'Hydrotel est gardé sauf gain net (plus de 5 % sur le rapport) ;
6. statut : `kept`, `changed`, `outside_territory` (coordonnées hors de toute unité du projet : la
   station n'a pas de tronçon dans ce domaine, à exclure), `no_coordinates` (fiche vide,
   rattachement d'Hydrotel gardé) ;
7. rapport aire du tronçon retenu / aire de la station, celui que pose `[stations] area_ratio`.

Sorties : `derives/auxiliaires/station-audit.csv` (une ligne par station) et
`derives/auxiliaires/station-attachment.csv` (territory, station, reach_id, status), lu par le
chargeur quand `[stations] attachment_table` le nomme.
Usage : python .runs/quebec/audit_stations.py [territoire ...]
"""
import os
import re
import sys
import time

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from meandre.adapters.physitel import database_path, project_dir
from meandre.adapters.physitel.cehq_loader import fetch_station
from meandre.utils import paths

HEADER_CACHE = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "cehq-entetes")
AUDIT_TABLE = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "station-audit.csv")
ATTACHMENT_TABLE = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "station-attachment.csv")
# Une station n'est pas descendue sous un tronçon dont l'aire vaut au moins 90 % de la sienne :
# l'aire déclarée par le CEHQ et celle de PHYSITEL diffèrent couramment de quelques pour cent.
AREA_TOLERANCE = 0.10
# Gain minimal, en logarithme du rapport d'aires, pour quitter le rattachement d'Hydrotel.
MIN_GAIN = np.log(1.05)


def station_header(station_id):
    """En-tête de la fiche CEHQ (texte), depuis le cache ou le site."""
    os.makedirs(HEADER_CACHE, exist_ok=True)
    path = os.path.join(HEADER_CACHE, f"{station_id}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    try:
        text = fetch_station(station_id)
    except Exception as e:
        return f"ERREUR {e}"
    header = "\n".join(text.splitlines()[:8])
    open(path, "w", encoding="utf-8").write(header)
    time.sleep(0.3)
    return header


def dms(d, m, s):
    return float(d) + float(m) / 60 + float(s) / 3600


def parse_header(header):
    """(lon, lat, aire km², régime) lus dans l'en-tête ; NaN quand absents."""
    lon = lat = area = np.nan
    regime = ""
    c = re.search(r"Coordonn\w+:\s*\(NAD83\)\s*(-?\d+)°\s*(\d+)'\s*(\d+)\"\s*//\s*(-?\d+)°\s*(\d+)'\s*(\d+)\"", header)
    if c:
        lat = dms(*c.group(1, 2, 3))
        lon = -dms(c.group(4).lstrip("-"), *c.group(5, 6))
    a = re.search(r"Bassin versant:\s*([\d.,]+)\s*km", header)
    if a:
        area = float(a.group(1).replace(",", "."))
    r = re.search(r"R\w+gime:\s*(.+?)\s*$", header, flags=re.M)
    if r:
        regime = r.group(1).strip()
    return lon, lat, area, regime


def read_reaches(project):
    """Depuis troncon.trl : unité -> tronçon, tronçon -> tronçon aval, tronçon -> tronçons amont."""
    lines = open(os.path.join(project, "physitel", "troncon.trl"), encoding="latin-1").read().split("\n")
    first = next(i for i, line in enumerate(lines) if line.strip() == "TRONCONS") + 1
    reach_of_unit = {}
    down_node, up_nodes = {}, {}
    for line in lines[first:]:
        t = line.split()
        if len(t) < 8:
            continue
        reach, kind = int(t[0]), int(t[1])
        if kind == 1:
            # rivière : ident, type, noeud aval, noeud amont, longueur, largeur, manning, n, unités..., ordre
            down_node[reach], up_nodes[reach] = int(t[2]), [int(t[3])]
            n = int(t[7])
            units = t[8:8 + n]
        else:
            # lac : ident, type, noeud aval, k, k noeuds amont, quatre réels, n, unités..., ordre
            k = int(t[3])
            down_node[reach], up_nodes[reach] = int(t[2]), [int(x) for x in t[4:4 + k]]
            n = int(t[4 + k + 4])
            units = t[4 + k + 5:4 + k + 5 + n]
        for u in units:
            reach_of_unit[int(u)] = reach
    # tronçon aval : celui dont un noeud amont est le noeud aval du tronçon courant
    reach_by_up_node = {nd: r for r, nds in up_nodes.items() for nd in nds}
    downstream = {r: reach_by_up_node.get(down_node[r]) for r in down_node}
    upstream = {}
    for r, d in downstream.items():
        if d is not None:
            upstream.setdefault(d, []).append(r)
    return reach_of_unit, downstream, upstream


def audit_territory(territory):
    con = duckdb.connect(str(database_path(territory)), read_only=True)
    stations = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
    nodes = con.execute("select n.node_idx, n.node_id, t.area_km2_physical as area from nodes n join territorial t using (node_idx)").df()
    con.close()
    area_of_reach = dict(zip(nodes.node_id.astype(int), nodes.area.astype(float)))
    reach_of_index = dict(zip(nodes.node_idx.astype(int), nodes.node_id.astype(int)))
    project = project_dir(territory)
    reach_of_unit, downstream, upstream = read_reaches(project)
    units = gpd.read_file(os.path.join(project, "physitel", "uhrh.shp"))
    rows = []
    for _, s in stations.iterrows():
        sid = str(s.station_id)
        lon, lat, cehq_area, regime = parse_header(station_header(sid))
        current = reach_of_index.get(int(s.node_idx))
        st_area = cehq_area if np.isfinite(cehq_area) else float(s.drainage_area_km2)
        unit = located = np.nan
        path = ""
        if np.isfinite(lon):
            point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(units.crs).iloc[0]
            inside = units[units.contains(point)]
            if len(inside):
                unit = int(inside.iloc[0]["ident"])
                r = reach_of_unit.get(unit)
                steps = [r]
                while r is not None and area_of_reach.get(r, np.inf) < st_area * (1 - AREA_TOLERANCE) and downstream.get(r) is not None:
                    r = downstream[r]
                    steps.append(r)
                located = r
                path = ">".join(str(x) for x in steps)
        a_current = area_of_reach.get(current, np.nan)
        a_located = area_of_reach.get(located, np.nan) if np.isfinite(located) else np.nan
        # Choix final : voisinage d'écoulement du tronçon trouvé, aire la plus proche ; Hydrotel
        # gardé sauf gain net.
        if not np.isfinite(lon):
            status, chosen = "no_coordinates", current
        elif not np.isfinite(located):
            status, chosen = "outside_territory", np.nan
        else:
            neighbours = {int(located)} | {x for x in [downstream.get(int(located))] if x is not None} | set(upstream.get(int(located), []))
            gap = {r: abs(np.log(area_of_reach[r] / st_area)) for r in neighbours if r in area_of_reach}
            best = min(gap, key=gap.get)
            chosen = current if abs(np.log(a_current / st_area)) <= gap[best] + MIN_GAIN else best
            status = "kept" if chosen == current else "changed"
        a_chosen = area_of_reach.get(chosen, np.nan) if np.isfinite(chosen) else np.nan
        rows.append(dict(territory=territory, station=sid, regime=regime, lon=lon, lat=lat, station_area=st_area,
                         current_reach=current, current_area=a_current, current_ratio=a_current / st_area,
                         unit=unit, located_reach=located, located_area=a_located, located_ratio=a_located / st_area,
                         descent=path, reach_id=chosen, reach_area=a_chosen, ratio=a_chosen / st_area, status=status))
    return pd.DataFrame(rows)


def main(territories):
    if not territories:
        territories = sorted(os.path.basename(f)[:-7] for f in os.listdir(os.path.dirname(str(database_path("sagu")))) if f.endswith(".duckdb") and len(f) == 11)
    tables = []
    for t in territories:
        try:
            d = audit_territory(t)
        except Exception as e:
            print(f"{t} : non audité ({type(e).__name__} : {e})")
            continue
        if d.empty:
            print(f"{t} : aucune station")
            continue
        tables.append(d)
        print(f"{t} : {len(d)} stations | " + ", ".join(f"{k} {v}" for k, v in d.status.value_counts().items()))
    d = pd.concat(tables)
    d.to_csv(AUDIT_TABLE, index=False)
    d[["territory", "station", "reach_id", "status"]].to_csv(ATTACHMENT_TABLE, index=False)
    print(f"\necrit : {AUDIT_TABLE}\necrit : {ATTACHMENT_TABLE}")
    flagged = d[(d.status != "kept") | (np.abs(np.log(d.ratio.fillna(1))) > np.log(1.5))]
    print(f"\n{len(flagged)} station(s) changee(s), hors territoire, sans coordonnees, ou a plus d'un facteur 1,5 apres choix :")
    cols = ["territory", "station", "status", "station_area", "current_reach", "current_ratio", "reach_id", "ratio"]
    print(flagged[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main([a.lower() for a in sys.argv[1:]])
