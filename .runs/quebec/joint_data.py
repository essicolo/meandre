"""Chargement d'UNE région pour l'entraînement conjoint (extrait le prep de slso.py).
Retourne train_data/val_data (TrainingData), loss_fn régionale, et les métadonnées
(n_nodes, node_ids, n_gauges) pour le MultiBasinTrainer. Les données viennent du domaine
produit par l'adaptateur PHYSITEL (meandre.adapters.physitel, 2026-10-26) ; ce module ne
garde que ce qui relève de l'entraînement : sous-bassin, tranches, perte.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from meandre.adapters.physitel import load_domain
from meandre.routing.withdrawals import WithdrawalData
import subbasin as _subbasin
from meandre.training.trainer import TrainingData
from meandre.training.loss import HydroLoss
from meandre.utils import settings as _settings

DATE_START, DATE_END = "2000-01-01", "2024-12-31"
# DÉCOUPAGE TEMPOREL, configurable par JOINT_SPLIT="fin_train,debut_val,fin_val".
# Mesuré le 2026-08-12 : le découpage historique sélectionne sur la période la plus
# SÈCHE en été (val 2019-2021 : 1062 mm de pluie JJA) et teste sur la plus HUMIDE
# (2022-2024 : 1384 mm, +30 %), avec en prime +1.0 °C de moyenne annuelle et des pics
# observés PLUS BAS malgré la pluie. Le juge est donc biaisé, et pas dans un sens
# anodin : c'est exactement le régime où méandre est le plus faible (excès d'été de
# 25-40 % contre Hydrotel). Pouvoir déplacer les trois fenêtres est la condition pour
# vérifier ce que le tenu de côté doit à la période plutôt qu'au modèle.
# Le découpage est lu À L'APPEL de load_region et non à l'import (2026-10-09) : le pilote importe
# ce module avant d'appliquer la section [recette] du TOML, si bien qu'un JOINT_SPLIT déclaré
# dans un fichier était ignoré sans bruit. La section [period] du TOML le pose aussi.
TRAIN_END, VAL_START, VAL_END = "2018-12-31", "2019-01-01", "2021-12-31"


def decoupage():
    """(fin d'entraînement, début et fin de validation) en vigueur, JOINT_SPLIT d'abord."""
    _split = os.environ.get("JOINT_SPLIT")
    if _split:
        return tuple(x.strip() for x in _split.split(","))
    return TRAIN_END, VAL_START, VAL_END

def _paths(reg):
    """(base, forçage) du territoire, résolus par l'adaptateur PHYSITEL."""
    from meandre.adapters.physitel import database_path, forcing_path
    return database_path(reg), forcing_path(reg)


def load_region(reg: str, lcfg: dict, device: str = "cuda", drainage_table: str | None = None):
    """lcfg = section [loss] du TOML de base (poids identiques pour toutes les régions) ;
    drainage_table = table des classes de drainage nommée par [soil.ceiling_from_drainage]."""
    reg = reg.lower()
    wanted = {"discharge", "et", "tws"}
    if lcfg.get("w_snow", 0.0) > 0:
        wanted.add("snow")
    if lcfg.get("w_swe_mass", 0.0) > 0:
        wanted.add("swe_mass")
    dom = load_domain(reg, DATE_START, DATE_END, device=device, observations=wanted, drainage_table=drainage_table)
    graph, territorial = dom.graph, dom.attributes
    node_coords, n_nodes, node_ids = dom.node_coords, dom.n_nodes, dom.node_ids
    forcing, times, doy = dom.forcing, dom.times, dom.day_of_year
    _q = dom.observations["discharge"]
    station_node_map = {s: int(n) for s, n in zip(_q.sites, _q.nodes.tolist())}
    station_indices = list(_q.nodes.tolist())
    n_stations = len(station_indices)
    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True
    q_obs = _q.values
    withdrawals = dom.withdrawals
    et_obs = dom.observations["et"].values if "et" in dom.observations else None
    et_weight = dom.observations["et"].weight if "et" in dom.observations else None
    swe_obs = dom.observations["snow"].values if "snow" in dom.observations else None
    swe_mass_obs = swe_mass_node = None
    if "swe_mass" in dom.observations:
        swe_mass_obs, swe_mass_node = dom.observations["swe_mass"].values, dom.observations["swe_mass"].nodes
    elif lcfg.get("w_swe_mass", 0.0) > 0:
        # ABSENCE DE DONNEE, pas erreur de cablage : on eteint le poids POUR CE TERRITOIRE plutot
        # que de laisser le garde-fou du trainer lever, et on le dit.
        lcfg["w_swe_mass"] = 0.0
        print(f"[canswe] {reg}: observation absente -> w_swe_mass mis a 0 POUR CETTE REGION")
    tws_obs = dom.observations["tws"].values if "tws" in dom.observations else None

    # SOUS-BASSIN (2026-10-23) : ETL_SUBBASIN=<territoire>:<station> restreint le territoire aux
    # troncons en amont de la station. Tout ce qui suit, statistiques par station, perte et
    # tranches, se calcule alors sur le sous-bassin. `node_index` garde l'indice d'origine de
    # chaque troncon, pour les tables lues par indice (champ k_gw, attributs bruts).
    node_index = np.arange(n_nodes)
    _station_sb = _subbasin.requested(reg)
    if _station_sb is not None:
        if _station_sb not in station_node_map:
            raise SystemExit(f"{reg}: ETL_SUBBASIN vise la station {_station_sb}, absente des observations retenues")
        keep = _subbasin.upstream(graph, station_node_map[_station_sb])
        rank = {int(v): k for k, v in enumerate(keep)}
        keep_t = torch.as_tensor(keep, device=forcing.device)
        graph = _subbasin.restrict_graph(graph, keep)
        territorial = _subbasin.restrict_territorial(territorial, keep)
        node_coords = node_coords[torch.as_tensor(keep, device=node_coords.device)]
        node_ids = [node_ids[i] for i in keep]
        forcing = forcing[:, keep_t]
        cols = [j for j, ni in enumerate(station_indices) if ni in rank]
        q_obs = q_obs[:, torch.as_tensor(cols, dtype=torch.long, device=q_obs.device)]
        station_indices = [rank[station_indices[j]] for j in cols]
        station_node_map = {s: rank[ni] for s, ni in station_node_map.items() if ni in rank}
        n_stations = len(station_indices)
        station_mask = torch.zeros(len(keep), dtype=torch.bool, device=device)
        station_mask[torch.as_tensor(station_indices, dtype=torch.long, device=device)] = True
        withdrawals = _subbasin.restrict_withdrawals(withdrawals, keep)
        et_obs = _subbasin.restrict_nodes(et_obs, keep)
        swe_obs = _subbasin.restrict_nodes(swe_obs, keep)
        et_weight = _subbasin.restrict_nodes(et_weight, keep)
        if swe_mass_node is not None:
            _sites = [k for k, ni in enumerate(swe_mass_node.tolist()) if int(ni) in rank]
            if _sites:
                _st = torch.as_tensor(_sites, dtype=torch.long, device=swe_mass_obs.device)
                swe_mass_obs = swe_mass_obs[:, _st]
                swe_mass_node = torch.as_tensor([rank[int(swe_mass_node[k])] for k in _sites], dtype=torch.long, device=swe_mass_node.device)
            else:
                swe_mass_obs = swe_mass_node = None
        n_nodes = len(keep)
        node_index = keep
        print(f"[sous-bassin] {reg}: {n_nodes} troncons en amont de la station {_station_sb}, {n_stations} station(s) dedans")

    # slices temporels (mêmes conventions que slso.py)
    def sl(d0, d1):
        i0 = int(np.searchsorted(times.values, np.datetime64(d0)))
        i1 = int(np.searchsorted(times.values, np.datetime64(d1))) + 1
        return slice(i0, i1)
    _te, _vs, _ve = decoupage()
    if (_te, _vs, _ve) != (TRAIN_END, VAL_START, VAL_END):
        print(f"[joint] découpage NON STANDARD : train -> {_te} | val {_vs}..{_ve}")
    train_sl = sl(DATE_START, _te)
    val_sl = sl(_vs, _ve)

    # loss régionale : poids partagés, stats stations locales
    station_var = torch.ones(n_stations, dtype=torch.float32, device=device)
    peak_thr = torch.full((n_stations,), float("inf"), dtype=torch.float32, device=device)
    q_train = q_obs[train_sl]
    for i in range(n_stations):
        m = ~torch.isnan(q_train[:, i])
        if m.sum() > 100:
            station_var[i] = q_train[m, i].var()
            peak_thr[i] = torch.quantile(q_train[m, i], 0.75)
    loss_fn = HydroLoss(
        w_nse=lcfg.get("w_nse", 0.0), w_kge=lcfg.get("w_kge", 0.0), w_pbias=lcfg.get("w_pbias", 0.0),
        w_mse=lcfg.get("w_mse", 0.0), w_nrmse=lcfg.get("w_nrmse", 0.0),
        w_log_nse=lcfg.get("w_log_nse", 0.0), w_log_mse=lcfg.get("w_log_mse", 0.0),
        w_et=lcfg.get("w_et", 0.0), et_mode=lcfg.get("et_mode", "level"),
        w_tws=lcfg.get("w_tws", 0.0),
        w_tws_clim=lcfg.get("w_tws_clim", 0.0),
        w_snow=lcfg.get("w_snow", 0.0),
        w_swe_mass=lcfg.get("w_swe_mass", 0.0),
        w_peak=lcfg.get("w_peak", 0.0),
        # FORME (2026-09-05) : variations journalieres et soutien d'etiage. Mesure au
        # banc de sous-bassin le meme jour : le terme dQ empeche le rabotage des
        # pointes (0.82 du pic observe apres six epoques contre 0.76 sans lui) et
        # gagne 0.008 de KGE et 0.015 de gamma.
        w_dq=lcfg.get("w_dq", 0.0), w_fdc_bas=lcfg.get("w_fdc_bas", 0.0),
        w_etiage=lcfg.get("w_etiage", 0.0),
        w_dq_log=lcfg.get("w_dq_log", 0.0),
        # KGE DECOMPOSE, RAPPORT DES POINTES ET VITESSE DE VIDANGE (2026-09-21). Ces cinq
        # poids etaient poses dans la configuration, imprimes par le pilote, et jamais
        # passes ici : la recette equilibree a tourne avec deux termes sur six.
        w_r=lcfg.get("w_r", 0.0), w_beta=lcfg.get("w_beta", 0.0),
        w_gamma=lcfg.get("w_gamma", 0.0), w_peak_ratio=lcfg.get("w_peak_ratio", 0.0),
        w_recession=lcfg.get("w_recession", 0.0),
        w_plat=lcfg.get("w_plat", 0.0),
        w_physics=lcfg.get("w_physics", 0.0),
        per_station=True, station_weights=None, station_var=station_var,
        peak_threshold=(peak_thr if (lcfg.get("w_peak", 0.0) > 0
                                     or lcfg.get("w_peak_ratio", 0.0) > 0) else None),
    )
    # GARDE CONTRE LA CLASSE DE DEFAUT (2026-09-21). Un poids de la configuration qui
    # n'arrive pas jusqu'a la perte ne se voit nulle part : le pilote l'imprime, le bilan
    # l'ecarte parce que sa valeur est nulle, et l'entrainement tourne sans lui. On compare
    # donc terme a terme ce qui a ete demande et ce qui a ete construit.
    import inspect as _inspect

    _acceptes = set(_inspect.signature(HydroLoss.__init__).parameters)
    _oubliees = [k for k, v in lcfg.items()
                 if k.startswith("w_") and k in _acceptes and float(v or 0.0) > 0
                 and abs(float(getattr(loss_fn, k, 0.0)) - float(v)) > 1e-12]
    if _oubliees:
        raise ValueError("poids demandes et non poses sur la perte : " + ", ".join(sorted(_oubliees)))

    def mk(sl_):
        return TrainingData(
            forcing=forcing, q_obs=q_obs[sl_.start:],
            station_mask=station_mask,
            station_idx=torch.tensor(station_indices, device=device, dtype=torch.long),
            graph=graph, node_coords=node_coords, territorial=territorial,
            withdrawals=withdrawals, day_of_year=doy,
            train_slice=sl_, val_slice=sl_,
            et_obs=et_obs[sl_.start:] if et_obs is not None else None,
            et_weight=et_weight,
            tws_obs=tws_obs[sl_.start:] if tws_obs is not None else None,
            swe_obs=swe_obs[sl_.start:] if swe_obs is not None else None,
            swe_mass_obs=(swe_mass_obs[sl_.start:]
                          if swe_mass_obs is not None else None),
            swe_mass_node=swe_mass_node,
        )
    # Identifiants des stations DANS L'ORDRE DES COLONNES de q_obs, pour pouvoir
    # rejoindre des donnees par station en aval (drapeaux de qualite du CEHQ, R19).
    # `station_indices` est un `sorted(set(...))` : si deux stations visaient le meme
    # noeud, l'une des deux disparaitrait ici. On rend la premiere et on le SIGNALE
    # plutot que de laisser un decalage silencieux s'installer.
    _par_noeud = {}
    for _sid, _ni in station_node_map.items():
        _par_noeud.setdefault(_ni, []).append(str(_sid))
    station_ids = [sorted(_par_noeud[_ni])[0] for _ni in station_indices]
    _double = {n: v for n, v in _par_noeud.items() if len(v) > 1}
    if _double:
        print(f"[stations] {len(_double)} noeud(s) vises par plusieurs stations, "
              f"une seule retenue par noeud : {_double}")

    return dict(name=reg, domain=dom, n_nodes=n_nodes, node_ids=node_ids, node_index=node_index, n_gauges=n_stations,
                train_data=mk(train_sl), val_data=mk(val_sl), loss_fn=loss_fn,
                node_coords=node_coords, territorial=territorial, times=times,
                station_ids=station_ids, station_indices=station_indices)
