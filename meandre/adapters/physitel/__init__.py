"""Adaptateur PHYSITEL : un domaine depuis les bases québécoises construites sur les plateformes.

Tout ce qui est propre au Québec et à PHYSITEL vit ici : la base DuckDB d'un territoire, son
forçage NetCDF, la normalisation des attributs sur les statistiques provinciales, les cibles
CanSWE, le poids hors neige tiré de NEISIM, la gravimétrie GRACE. Le modèle et le pilote ne
voient que le `Domain` qui en sort, et son manifeste dit d'où vient chaque pièce.

Ce que cet adaptateur assemble reproduit exactement ce que `joint_data.load_region` assemblait
jusqu'au 2026-10-26 ; la passe à poids gelés sert de preuve. Les pièces que le pilote lit encore
lui-même (occupation brute du projet, texture SIIGSOL, classe de drainage, champ de conductivité
souterraine) sont inscrites au manifeste dès maintenant, et migreront ici à l'étape 2.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from meandre.data.basin_cache import BasinCache
from meandre.domain import Domain, Manifest, Observation, register
from meandre.spatial.territorial import TerritorialFeatures
from meandre.utils import paths as _paths
from meandre.utils import settings as _settings

FORCING_VARIABLES = ["P", "Tmin", "Tmax", "Rn", "u2", "ea"]
QC_RAW = f"{_paths.DATA_ROOT}/quebec/territorial-raw-QC.parquet"
QC_STATS = ".runs/quebec/data/territorial_stats_QC.csv"
# Bases et forçages propres à un territoire, hors de la famille québécoise (banc SLSO historique).
LEGACY_FORCINGS = {"slso": f"{_paths.DATA_ROOT}/slso/forcing-casr-corr.nc"}
LEGACY_DBS = {"slso": ".runs/slso/data/slso.duckdb"}


def database_path(name: str) -> str:
    """Base du territoire : la famille québécoise dès qu'elle existe, le banc historique sinon."""
    special = LEGACY_DBS.get(name)
    common = f"{_paths.DATA_ROOT}/quebec/{name}.duckdb"
    if special and os.path.exists(special) and not os.path.exists(common):
        return special
    if special and os.path.exists(common):
        print(f"[base] {name} : {os.path.basename(common)} (famille quebecoise) plutot que le banc historique {special}")
    return common


def forcing_path(name: str) -> str:
    """Forçage selon `forcing.suffix` ; `-none` est CaSR brut sans suffixe. Aucun repli silencieux."""
    sfx = _settings.raw("forcing.suffix")
    if sfx == "-none":
        fx = f"{_paths.DATA_ROOT}/quebec/forcing-{name}.nc"
        if not os.path.exists(fx):
            raise FileNotFoundError(f"{name}: CaSR brut demandé mais {fx} absent")
        return fx
    if sfx:
        fx = f"{_paths.DATA_ROOT}/quebec/forcing-{name}{sfx}.nc"
        if os.path.exists(fx):
            return fx
        if name not in LEGACY_FORCINGS:
            raise FileNotFoundError(f"{name}: forcage '{sfx}' demande mais {fx} absent. Aucun repli : un point de reprise evalue sur un autre forcage que le sien rend un score faux.")
        print(f"[forcage] {name}: suffixe '{sfx}' introuvable, fichier propre du territoire utilise ({os.path.basename(LEGACY_FORCINGS[name])})")
    return LEGACY_FORCINGS.get(name, f"{_paths.DATA_ROOT}/quebec/forcing-{name}-budyko.nc")


def _normalise_on_province(name: str, local: TerritorialFeatures, device, manifest: Manifest) -> TerritorialFeatures:
    """Attributs normalisés sur les statistiques provinciales, non territoriales.

    Sans cela le tronçon médian de chaque territoire arrive au champ avec des attributs proches
    de zéro et le modèle ne distingue pas les territoires (2026-07-31). Retombe sur la
    normalisation locale si le brut est absent."""
    if not (os.path.exists(QC_RAW) and os.path.exists(QC_STATS)):
        return local
    manifest.add("attributes_raw_province", QC_RAW, "PHYSITEL, agrégé par territoire", "attributes")
    manifest.add("attributes_stats_province", QC_STATS, "statistiques de normalisation", "attributes")
    try:
        raw = pd.read_parquet(QC_RAW)
        raw = raw[raw["region"] == name]
        if len(raw) != local.data.shape[0]:
            return local
        stats = pd.read_csv(QC_STATS, index_col=0)
        cols = list(local.columns)
        arr = np.stack([((raw[c].values - stats.loc[c, "mean"]) / (stats.loc[c, "std"] + 1e-9)) for c in cols], axis=1).astype(np.float32)
        return TerritorialFeatures(data=torch.tensor(arr, device=device), columns=cols, physical=local.physical)
    except Exception:
        return local


def load_domain(name: str, date_start: str, date_end: str, device="cuda", observations: set[str] | None = None) -> Domain:
    """Domaine d'un territoire PHYSITEL sur une fenêtre de dates.

    `observations` : noms des observations à charger parmi discharge, et, snow, swe_mass, tws ;
    `None` charge discharge, et et tws, comme le pilote. Une observation demandée mais absente
    de la source est omise du domaine, et dite.
    """
    name = name.lower()
    wanted = {"discharge", "et", "tws"} if observations is None else set(observations)
    manifest = Manifest(name)
    db_path, fx_path = database_path(name), forcing_path(name)
    manifest.add("database", db_path, "PHYSITEL, observations CEHQ/HYDAT, MOD16, GRACE, prélèvements io-eau", "network, attributes, observations, state")
    manifest.add("forcing", fx_path, "CaSR" if "casr" in fx_path else os.path.basename(fx_path), "forcing")

    cache = BasinCache(db_path)
    h = cache.load(device=device)
    graph, territorial = h["graph"], h["territorial"]
    attribute_columns = list(territorial.columns) + list(territorial.physical)
    if _settings.raw("spatial.global_normalization", "0") == "1":
        territorial = _normalise_on_province(name, territorial, device, manifest)

    import xarray as xr
    print(f"[forcage] {name}: {os.path.basename(fx_path)}")
    d = xr.open_dataset(fx_path)
    F = d["forcing"].values[:, :, :len(FORCING_VARIABLES)]   # (T, N, 6) ; le banc SLSO a 7 canaux
    times = pd.to_datetime(d["time"].values)
    d.close()
    assert str(times[0])[:10] == date_start and str(times[-1])[:10] == date_end, f"{name}: fenêtre forçage {times[0]}..{times[-1]} pour {date_start}..{date_end}"
    forcing = torch.tensor(F, dtype=torch.float32, device=device)
    n_nodes = h["n_nodes"]

    obs: dict[str, Observation] = {}
    if "discharge" in wanted:
        o = cache.load_observations(date_start=date_start, date_end=date_end, min_valid_days=365)
        station_node_map = o["station_node_map"]
        node_of = {}
        for sid, ni in station_node_map.items():
            node_of.setdefault(int(ni), []).append(str(sid))
        nodes = sorted(node_of)
        doubles = {ni: v for ni, v in node_of.items() if len(v) > 1}
        if doubles:
            print(f"[stations] {len(doubles)} noeud(s) vises par plusieurs stations, une seule retenue par noeud : {doubles}")
        sites = [sorted(node_of[ni])[0] for ni in nodes]
        values = torch.from_numpy(o["discharge"][:, nodes]).to(device) if nodes else torch.full((len(times), 0), float("nan"), device=device)
        obs["discharge"] = Observation("discharge", values, "site", sites, torch.tensor(nodes, dtype=torch.long, device=device), origin="CEHQ, HYDAT")
    if "et" in wanted:
        et = cache.load_modis_et(date_start, date_end, device=device)
        if et is not None:
            weight = None
            tau = _settings.raw("evaporation.mod16_snow_scale_mm")
            if tau:
                fz = f"{_paths.DERIVED_ROOT}/auxiliaires/neisim-{name}.npz"
                if not os.path.exists(fz):
                    raise FileNotFoundError(f"poids hors neige : NEISIM absent, {fz} (build_neisim_targets.py)")
                manifest.add("swe_neisim", fz, "NEISIM", "weight of et")
                zz = np.load(fz)
                vv = torch.tensor(np.nan_to_num(zz["valeurs"], nan=0.0), dtype=torch.float32)
                assert vv.shape[0] == len(times), f"{name}: axe de temps NEISIM {vv.shape[0]} vs {len(times)}"
                # Sur le processeur : 125 Mo sur l'Outaouais ; l'entraîneur n'en extrait que la tranche du bloc.
                weight = torch.ones(len(times), n_nodes, dtype=torch.float32)
                weight[:, torch.tensor(zz["node_idx"], dtype=torch.long)] = torch.exp(-vv / float(tau))
                print(f"[et] {name}: terme MOD16 pondere hors neige, w = exp(-SWE NEISIM / {float(tau):g} mm), poids moyen {float(weight.mean()):.2f}")
            obs["et"] = Observation("et", et, "node", weight=weight, origin="MOD16")
    if "snow" in wanted:
        sn = cache.load_modis_snow(date_start, date_end, device=device)
        if sn is not None:
            obs["snow"] = Observation("snow", sn, "node", origin="MOD10")
    if "swe_mass" in wanted and cache.has_canswe():
        from meandre.data.canswe_loader import build_swe_targets
        mes, sit = cache.load_canswe(date_start, date_end)
        values, nodes, kept = build_swe_targets(mes, sit, times)
        if values is None:
            print(f"[canswe] {name}: aucun site ne passe les filtres de representativite -> observation swe_mass omise")
        else:
            print(f"[canswe] {name}: {len(kept)}/{len(sit)} sites retenus, {int(torch.isfinite(values).sum())} releves de masse")
            obs["swe_mass"] = Observation("swe_mass", values.to(device), "site", [str(s) for s in kept], nodes.to(device), origin="CanSWE")
    if "tws" in wanted:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        if "grace_tws" in [t[0] for t in con.execute("show tables").fetchall()]:
            g = con.execute("select date, tws_mm from grace_tws where quality_ok = true order by date").fetchdf()
            tws = torch.full((len(times),), float("nan"), device=device)
            ad = times.normalize()
            for dt, val in zip(pd.to_datetime(g["date"]), g["tws_mm"].values):
                target = pd.Timestamp(year=dt.year, month=dt.month, day=15)
                dd = np.abs((ad - target).days.values)
                i = int(dd.argmin())
                if dd[i] <= 20:
                    tws[i] = float(val)
            obs["tws"] = Observation("tws", tws, "domain", origin="GRACE")
        con.close()

    withdrawals = cache.load_withdrawals(date_start, date_end, device=device)
    # Pièces encore lues par le pilote lui-même, inscrites pour que leur empreinte soit dans la recette.
    for key, path, origin, role in (
        ("drainage_class", f"{_paths.DERIVED_ROOT}/auxiliaires/slc-drainage-troncons.parquet", "Pédo-paysages du Canada", "percolation ceiling"),
        ("groundwater_conductivity", f"{_paths.DATA_ROOT}/quebec/champ_kgw_QC.parquet", "récessions des jauges", "k_gw field"),
        ("physitel_reaches", f"{_paths.PLATFORMS_ROOT}/LN24HA/{name.upper()}_LN24HA_2020/physitel/troncon.trl", "PHYSITEL", "reach areas, land cover"),
    ):
        if os.path.exists(path):
            manifest.add(key, path, origin, role)

    domain = Domain(name=name, node_ids=h["node_ids"], node_coords=h["node_coords"], graph=graph, attributes=territorial, attribute_columns=attribute_columns, forcing=forcing, forcing_variables=list(FORCING_VARIABLES), times=times, initial_state=h["initial_state"], withdrawals=withdrawals, observations=obs, manifest=manifest)
    issues = domain.validate()
    if issues:
        raise ValueError(f"domaine {name} invalide : " + " ; ".join(issues))
    register(manifest)
    print("[domaine] " + domain.summary())
    return domain
