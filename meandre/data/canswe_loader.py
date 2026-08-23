"""CanSWE : equivalent en eau de la neige MESURE, apparie aux noeuds d'un bassin.

Pourquoi ce jeu plutot qu'un produit de modele (2026-08-20). La crue printaniere
de meandre arrive avec un mois de retard (avril a 0.729 de l'observe, mai a 1.07-1.42) :
il faut savoir si le manteau fond trop tard ou s'il n'a jamais eu la bonne masse.
Aucune reponse ne peut venir du debit seul, qui est deja la variable ajustee.

CanSWE (Vionnet et al. 2021, ESSD, https://doi.org/10.5281/zenodo.4734371) est une
MESURE, pas une reconstruction : releves nivometriques des agences provinciales, des
producteurs d'hydroelectricite et du SMC. Il echappe donc entierement au probleme de
circularite qui ecarte CaSR-Rivieres (1704 jauges de debit assimilees) et les cartes
de recharge PACES/HydroBudget (calees sur le debit de base). Voir le tableau des
lignes rouges au registre.

v8 : 1928-2025, 2963 stations. Sur l'emprise d'OUTV : 153 sites actifs et 65 646
mesures entre 2000 et 2024 ; sur le Quebec, 601 sites et ~212 000 mesures.

Nature de la mesure : 2680 stations sur 2963 sont des releves MULTI-POINTS (transect),
donc deja une moyenne spatiale, ce qui les rend bien plus comparables a une moyenne de
troncon qu'une mesure ponctuelle. Reste un biais de site : un site nivometrique est
souvent en clairiere alors que le troncon porte sa fraction forestiere, et la canopee
intercepte. Usage recommande : contrainte de TENDANCE et de TIMING (date de
disparition du manteau), pas de NIVEAU. Meme regle que MODIS et GRACE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meandre.utils import paths as _paths

DEFAULT_FILE = f"{_paths.DATA_ROOT}/canswe/CanSWE-CanEEN_1928-2025_v8.nc"

# data_flag_snw : '' propre, R revise, B releve hors periode nominale (mesure valide,
# seul son calendrier deroge), T trace. On ECARTE M (manquant), A et C (problemes
# d'echantillonnage), E (estimation), G (site a plus d'un km), P (plaques), N.
_DATA_FLAGS_OK = {"", "R", "B", "T"}
# qc_flag_snw : tout drapeau non vide signale une valeur suspecte ou deja mise a nan.
_QC_FLAGS_OK = {""}


def _as_text(a):
    """Normalise en texte un tableau CanSWE. Les drapeaux et identifiants sont stockes
    en OCTETS (dtype |S1) : un str() naif rend "b''" au lieu de "", et tout filtre de
    qualite rejette alors 100 % des mesures (bug du 2026-08-20, 45 037 mesures ecartees
    en silence). On decode explicitement."""
    out = np.empty(a.shape, dtype=object)
    plat = a.ravel()
    res = out.ravel()
    for i, x in enumerate(plat):
        if isinstance(x, bytes):
            res[i] = x.decode("utf-8", "ignore").strip()
        elif x is None:
            res[i] = ""
        else:
            res[i] = str(x).strip()
    return res.reshape(a.shape).astype(str)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique (km), vectorisee sur le second argument."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def read_canswe(lat_noeuds, lon_noeuds, elev_noeuds=None, chemin: str | None = None,
                   marge_deg: float = 0.25, dist_max_km: float = 25.0,
                   date_debut: str = "1980-01-01", date_fin: str = "2025-07-31"):
    """Selectionne les sites CanSWE d'une region et les apparie au noeud le plus proche.

    Retourne (sites, mesures) :
      sites   : swe_station_id, node_idx, lat, lon, elevation_m, source, type_mes,
                dist_km, elev_diff_m  -- une ligne par site retenu
      mesures : swe_station_id, date, swe_mm, snow_depth_m, quality_ok

    dist_max_km ecarte les sites trop loin de tout noeud. elev_diff_m est CONSERVE
    plutot que filtre : la neige depend fortement de l'altitude, et cet ecart est
    l'information qui permettra plus tard de ponderer ou de corriger, ou simplement
    d'expliquer un desaccord.
    """
    import xarray as xr

    d = xr.open_dataset(chemin or DEFAULT_FILE)
    lat_s = d["lat"].values.astype(float)
    lon_s = d["lon"].values.astype(float)

    lat_n = np.asarray(lat_noeuds, dtype=float)
    lon_n = np.asarray(lon_noeuds, dtype=float)
    dans = ((lat_s >= lat_n.min() - marge_deg) & (lat_s <= lat_n.max() + marge_deg)
            & (lon_s >= lon_n.min() - marge_deg) & (lon_s <= lon_n.max() + marge_deg))
    idx_s = np.flatnonzero(dans)
    if idx_s.size == 0:
        d.close()
        return pd.DataFrame(), pd.DataFrame()

    # appariement au noeud le plus proche
    node_idx = np.empty(idx_s.size, dtype=int)
    dist_km = np.empty(idx_s.size, dtype=float)
    for k, i in enumerate(idx_s):
        dd = _haversine_km(lat_s[i], lon_s[i], lat_n, lon_n)
        j = int(np.argmin(dd))
        node_idx[k], dist_km[k] = j, float(dd[j])

    garde = dist_km <= dist_max_km
    idx_s, node_idx, dist_km = idx_s[garde], node_idx[garde], dist_km[garde]
    if idx_s.size == 0:
        d.close()
        return pd.DataFrame(), pd.DataFrame()

    def _txt(nom):
        return _as_text(d[nom].values[idx_s])

    elev_s = d["elevation"].values[idx_s].astype(float)
    ediff = (elev_s - np.asarray(elev_noeuds, dtype=float)[node_idx]
             if elev_noeuds is not None else np.full(idx_s.size, np.nan))
    ids = _txt("station_id")

    sites = pd.DataFrame({
        "swe_station_id": ids,
        "node_idx": node_idx.astype("int32"),
        "lat": lat_s[idx_s], "lon": lon_s[idx_s],
        "elevation_m": elev_s,
        "source": _txt("source"),
        "type_mes": _txt("type_mes"),
        "dist_km": dist_km.astype("float32"),
        "elev_diff_m": ediff.astype("float32"),
    })

    # mesures, restreintes a la periode demandee
    t = pd.DatetimeIndex(d["time"].values)
    per = np.flatnonzero((t >= pd.Timestamp(date_debut)) & (t <= pd.Timestamp(date_fin)))
    snw = d["snw"].values[idx_s][:, per].astype("float32")
    snd = d["snd"].values[idx_s][:, per].astype("float32")
    dfl = d["data_flag_snw"].values[idx_s][:, per]
    qfl = d["qc_flag_snw"].values[idx_s][:, per]
    d.close()

    fini = np.isfinite(snw)
    si, ti = np.nonzero(fini)
    if si.size == 0:
        return sites, pd.DataFrame()

    def _plat(a):
        return _as_text(a[si, ti])

    ok = np.isin(_plat(dfl), list(_DATA_FLAGS_OK)) & np.isin(_plat(qfl), list(_QC_FLAGS_OK))
    mesures = pd.DataFrame({
        "swe_station_id": ids[si],
        "date": t[per][ti],
        "swe_mm": snw[si, ti],              # kg/m2 == mm d'eau
        "snow_depth_m": snd[si, ti],
        "quality_ok": ok,
    })
    return sites, mesures


def build_swe_targets(mesures, sites, times, max_dist_km: float = 15.0,
                      max_elev_diff_m: float = 150.0):
    """Cible d'entrainement sur la MASSE du manteau, a partir des releves CanSWE.

    Pourquoi la masse et pas la couverture (R24, 2026-08-21). La cible existante est
    la fraction de couverture MOD10, qui sature des qu'il y a un peu de neige
    (``SCF = 1-exp(-SWE/15)``) et ne porte donc presque aucune information sur la
    quantite d'eau stockee -- justement ce qui manque au modele. Pire, MODIS mesure
    une reflectance et sous-estime la neige sous couvert forestier : sur OUTV, boise
    a 74 %, la contrainte demandait au modele de FONDRE PLUS TOT (+0.47 de fraction
    en mars, 38 ecarts-types), contre GRACE et contre CanSWE. Les releves CanSWE sont
    des mesures de masse au sol, insensibles au couvert.

    Representativite. Un releve est PONCTUEL, un troncon est surfacique. On ne peut
    pas la corriger sans inventer un parametre par site, ce qui detruirait
    l'identifiabilite qu'on cherche ; on la BORNE en ecartant les sites trop loin du
    noeud ou a une altitude trop differente, et on laisse le reste au nombre de
    sites. Les deux seuils sont donc des choix explicites, pas des reglages.

    Retourne ``(valeurs, node_idx, sites_gardes)`` :
      - ``valeurs`` (T, n_sites) en mm, NaN partout ou il n'y a pas de releve ;
      - ``node_idx`` (n_sites,) le noeud de chaque site ;
      - ``sites_gardes`` la table des sites retenus (pour le journal).
    Retourne ``(None, None, None)`` si rien ne survit aux filtres.
    """
    import numpy as np
    import pandas as pd
    import torch

    if mesures is None or sites is None or len(mesures) == 0:
        return None, None, None
    gardes = sites[(sites["dist_km"] <= max_dist_km)
                   & (sites["elev_diff_m"].abs() <= max_elev_diff_m)
                   & sites["node_idx"].notna()].copy()
    if len(gardes) == 0:
        return None, None, None
    gardes = gardes.reset_index(drop=True)
    rang = {s: i for i, s in enumerate(gardes["swe_station_id"])}

    m = mesures[mesures["swe_station_id"].isin(rang)].copy()
    m = m[m["swe_mm"].notna()]
    if len(m) == 0:
        return None, None, None

    axe = pd.DatetimeIndex(times)
    pos = pd.Series(np.arange(len(axe)), index=axe)
    it = pos.reindex(pd.DatetimeIndex(m["date"])).to_numpy()
    ok = np.isfinite(it)
    if not ok.any():
        return None, None, None

    valeurs = np.full((len(axe), len(gardes)), np.nan, dtype=np.float32)
    lignes = it[ok].astype(int)
    colonnes = m["swe_station_id"].map(rang).to_numpy()[ok].astype(int)
    # Plusieurs releves peuvent tomber le meme jour au meme site (doublons de source) :
    # le dernier ecrit gagne, l'ecart entre eux est negligeable devant la
    # representativite ponctuelle qu'on ne sait de toute facon pas corriger.
    valeurs[lignes, colonnes] = m["swe_mm"].to_numpy(dtype=np.float32)[ok]

    return (torch.from_numpy(valeurs),
            torch.tensor(gardes["node_idx"].to_numpy(), dtype=torch.long),
            gardes)
