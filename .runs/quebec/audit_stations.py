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
6. statut : « garde », « change », « hors territoire » (coordonnées hors de toute unité du projet :
   la station n'a pas de tronçon dans ce domaine, à exclure), « sans coordonnées » (fiche vide,
   rattachement d'Hydrotel gardé) ;
7. rapport aire du tronçon retenu / aire de la station, celui que pose `[stations] area_ratio`.

Sorties : `derives/auxiliaires/audit-stations.csv` (une ligne par station) et
`derives/auxiliaires/rattachement-stations.csv` (territoire, station, tronçon retenu, statut), lu
par le chargeur quand `[stations] attachment_table` le nomme.
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

CACHE = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "cehq-entetes")
SORTIE = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "audit-stations.csv")
# Une station n'est pas descendue sous un tronçon dont l'aire vaut au moins 90 % de la sienne :
# l'aire déclarée par le CEHQ et celle de PHYSITEL diffèrent couramment de quelques pour cent.
TOLERANCE = 0.10
# Gain minimal, en logarithme du rapport d'aires, pour quitter le rattachement d'Hydrotel.
GAIN_MIN = np.log(1.05)
RATTACHEMENT = os.path.join(paths.DERIVED_ROOT, "auxiliaires", "rattachement-stations.csv")


def entete(station_id):
    """En-tête de la fiche CEHQ (texte), depuis le cache ou le site."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{station_id}.txt")
    if os.path.exists(f):
        return open(f, encoding="utf-8").read()
    try:
        texte = fetch_station(station_id)
    except Exception as e:
        return f"ERREUR {e}"
    tete = "\n".join(texte.splitlines()[:8])
    open(f, "w", encoding="utf-8").write(tete)
    time.sleep(0.3)
    return tete


def dms(d, m, s):
    return float(d) + float(m) / 60 + float(s) / 3600


def coordonnees(tete):
    """(lon, lat, aire km², régime) lus dans l'en-tête ; NaN quand absents."""
    lon = lat = aire = np.nan
    regime = ""
    c = re.search(r"Coordonn\w+:\s*\(NAD83\)\s*(-?\d+)°\s*(\d+)'\s*(\d+)\"\s*//\s*(-?\d+)°\s*(\d+)'\s*(\d+)\"", tete)
    if c:
        lat = dms(*c.group(1, 2, 3))
        lon = -dms(c.group(4).lstrip("-"), *c.group(5, 6))
    a = re.search(r"Bassin versant:\s*([\d.,]+)\s*km", tete)
    if a:
        aire = float(a.group(1).replace(",", "."))
    r = re.search(r"R\w+gime:\s*(.+?)\s*$", tete, flags=re.M)
    if r:
        regime = r.group(1).strip()
    return lon, lat, aire, regime


def troncons(projet):
    """Depuis troncon.trl : unité -> tronçon, et tronçon -> (noeud aval, noeuds amont)."""
    lignes = open(os.path.join(projet, "physitel", "troncon.trl"), encoding="latin-1").read().split("\n")
    debut = next(i for i, l in enumerate(lignes) if l.strip() == "TRONCONS") + 1
    uhrh_vers = {}
    aval, amont = {}, {}
    for l in lignes[debut:]:
        t = l.split()
        if len(t) < 8:
            continue
        ident, kind = int(t[0]), int(t[1])
        if kind == 1:
            # rivière : ident, type, noeud aval, noeud amont, longueur, largeur, manning, n, unités..., ordre
            aval[ident], amont[ident] = int(t[2]), [int(t[3])]
            n = int(t[7])
            unites = t[8:8 + n]
        else:
            # lac : ident, type, noeud aval, k, k noeuds amont, quatre réels, n, unités..., ordre
            k = int(t[3])
            aval[ident], amont[ident] = int(t[2]), [int(x) for x in t[4:4 + k]]
            n = int(t[4 + k + 4])
            unites = t[4 + k + 5:4 + k + 5 + n]
        for u in unites:
            uhrh_vers[int(u)] = ident
    # tronçon en aval : celui dont un noeud amont est le noeud aval du tronçon courant
    par_noeud_amont = {nd: r for r, nds in amont.items() for nd in nds}
    suivant = {r: par_noeud_amont.get(aval[r]) for r in aval}
    precedents = {}
    for r, sv in suivant.items():
        if sv is not None:
            precedents.setdefault(sv, []).append(r)
    return uhrh_vers, suivant, precedents


def auditer(territoire):
    con = duckdb.connect(str(database_path(territoire)), read_only=True)
    st = con.execute("select s.station_id, s.node_idx, s.drainage_area_km2 from stations s").df()
    nd = con.execute("select n.node_idx, n.node_id, t.area_km2_physical as aire from nodes n join territorial t using (node_idx)").df()
    con.close()
    aire_par_ident = dict(zip(nd.node_id.astype(int), nd.aire.astype(float)))
    ident_par_idx = dict(zip(nd.node_idx.astype(int), nd.node_id.astype(int)))
    projet = project_dir(territoire)
    uhrh_vers, suivant, precedents = troncons(projet)
    uhrh = gpd.read_file(os.path.join(projet, "physitel", "uhrh.shp"))
    lignes = []
    for _, s in st.iterrows():
        sid = str(s.station_id)
        lon, lat, aire_cehq, regime = coordonnees(entete(sid))
        actuel = ident_par_idx.get(int(s.node_idx))
        aire_st = aire_cehq if np.isfinite(aire_cehq) else float(s.drainage_area_km2)
        unite = propose = np.nan
        chemin = ""
        if np.isfinite(lon):
            p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(uhrh.crs).iloc[0]
            dedans = uhrh[uhrh.contains(p)]
            if len(dedans):
                unite = int(dedans.iloc[0]["ident"])
                r = uhrh_vers.get(unite)
                pas = [r]
                while r is not None and aire_par_ident.get(r, np.inf) < aire_st * (1 - TOLERANCE) and suivant.get(r) is not None:
                    r = suivant[r]
                    pas.append(r)
                propose = r
                chemin = ">".join(str(x) for x in pas)
        a_act = aire_par_ident.get(actuel, np.nan)
        a_pro = aire_par_ident.get(propose, np.nan) if np.isfinite(propose) else np.nan
        # Choix final : voisinage d'écoulement du tronçon trouvé, aire la plus proche ; Hydrotel
        # gardé sauf gain net.
        if not np.isfinite(lon):
            statut, retenu = "sans coordonnées", actuel
        elif not np.isfinite(propose):
            statut, retenu = "hors territoire", np.nan
        else:
            voisins = {int(propose)} | {x for x in [suivant.get(int(propose))] if x is not None} | set(precedents.get(int(propose), []))
            ecart = {r: abs(np.log(aire_par_ident[r] / aire_st)) for r in voisins if r in aire_par_ident}
            meilleur = min(ecart, key=ecart.get)
            retenu = actuel if abs(np.log(a_act / aire_st)) <= ecart[meilleur] + GAIN_MIN else meilleur
            statut = "garde" if retenu == actuel else "change"
        a_ret = aire_par_ident.get(retenu, np.nan) if np.isfinite(retenu) else np.nan
        lignes.append(dict(territoire=territoire, station=sid, regime=regime, lon=lon, lat=lat, aire_station=aire_st,
                           troncon_actuel=actuel, aire_actuel=a_act, rapport_actuel=a_act / aire_st,
                           uhrh=unite, troncon_propose=propose, aire_propose=a_pro, rapport_propose=a_pro / aire_st,
                           descente=chemin, troncon_retenu=retenu, aire_retenu=a_ret, rapport_retenu=a_ret / aire_st, statut=statut))
    return pd.DataFrame(lignes)


def main(territoires):
    if not territoires:
        territoires = sorted(os.path.basename(f)[:-7] for f in os.listdir(os.path.dirname(str(database_path("sagu")))) if f.endswith(".duckdb") and len(f) == 11)
    tables = []
    for t in territoires:
        try:
            d = auditer(t)
        except Exception as e:
            print(f"{t} : non audité ({type(e).__name__} : {e})")
            continue
        if d.empty:
            print(f"{t} : aucune station")
            continue
        tables.append(d)
        print(f"{t} : {len(d)} stations | " + ", ".join(f"{k} {v}" for k, v in d.statut.value_counts().items()))
    d = pd.concat(tables)
    d.to_csv(SORTIE, index=False)
    d[["territoire", "station", "troncon_retenu", "statut"]].to_csv(RATTACHEMENT, index=False)
    print(f"\necrit : {SORTIE}\necrit : {RATTACHEMENT}")
    hors = d[(d.statut != "garde") | (np.abs(np.log(d.rapport_retenu.fillna(1))) > np.log(1.5))]
    print(f"\n{len(hors)} station(s) changee(s), hors territoire, sans coordonnees, ou a plus d'un facteur 1,5 apres choix :")
    cols = ["territoire", "station", "statut", "aire_station", "troncon_actuel", "rapport_actuel", "troncon_retenu", "rapport_retenu"]
    print(hors[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main([a.lower() for a in sys.argv[1:]])
