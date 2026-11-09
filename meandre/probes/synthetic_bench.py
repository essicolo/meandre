"""Banc fictif en un seul processus : le modèle chargé une fois, les variantes enchaînées en mémoire.

Le pilote, lancé avec `forcing.synthetic` (météo fabriquée identique sur tous les tronçons) et
`bench.variants`, appelle `run_bench` au lieu de son évaluation. Chaque variante est simulée deux
fois : avec la météo fabriquée, puis avec la même météo sans les orages (témoin) ; la réponse d'un
orage est la différence entre les deux, ce qui retire la décrue des pluies précédentes. Seules les
grandeurs par tronçon sont gardées : rien de journalier n'est écrit.

Fichier des variantes (TOML) :

    storms = "chemin/fictif-v1-orages.csv"     # date, pluie_mm, apres_pluie
    [[variant]]
    name = "base"
    [[variant]]
    name = "p80"
    multipliers = { porosity_1 = 0.8 }        # facteur sur une sortie du champ spatial
    settings = { "column.saturated_area_exponent" = 16.0 }   # réglages déclarés
    open_water_reservoir = "10,1.5,0.5"       # séjour (j), exposant, hauteur de référence (m)

Grandeurs par tronçon :
- coef_<cas> : débit latéral versé à la rivière des 7 jours suivant l'orage, en sus du témoin,
  rapporté à la pluie de l'orage ; cas = sec_<mm> (après dix jours secs) ou humide_<mm> (après
  trois jours de pluie) ;
- riviere_20j_<cas> : la même chose sur 20 jours ;
- pente_pluie : pente log-log du coefficient sur la pluie, orages après période sèche ;
- rapport_humide_sec_20 : coefficient après pluie sur coefficient après sécheresse, orage de 20 mm ;
- etiage_rapport : débit latéral minimal sur 7 jours de la sécheresse du 15 juillet au
  15 septembre de la deuxième année, rapporté à sa moyenne de juin ;
- etr_mm : évapotranspiration réelle de la deuxième année ;
- part_avril_mai, part_juin_juillet : part de l'écoulement annuel de la deuxième année en avril-mai
  et du 1er juin au 14 juillet (témoin, sans orage) ; ecoulement_annuel_mm : son total.
"""
from __future__ import annotations

import time
import tomllib

import numpy as np
import pandas as pd
import torch

from meandre.utils import settings

COLUMNS = ["coef_sec_10", "coef_sec_60", "pente_pluie", "rapport_humide_sec_20", "part_avril_mai", "part_juin_juillet", "etiage_rapport", "ecoulement_annuel_mm", "etr_mm"]
QUANTILES = [0.1, 0.5, 0.9]


def _series(diag, area_km2):
    """Débit latéral versé à la rivière et ETR par tronçon (T, N), en mm/j, sur l'hôte.

    Le débit latéral est ce que la rivière reçoit du tronçon, après la colonne et tout ce qui
    s'intercale entre elle et le routage (nappe, réservoir de l'eau libre, évaporation de l'eau
    libre) ; les flux de la colonne seuls ne voient pas ces termes."""
    q = diag.q_lateral.detach().float().cpu().numpy()
    lateral = q * 86400.0 / (area_km2[None, :] * 1e6) * 1e3
    return lateral, diag.etr.detach().float().cpu().numpy()


def metrics(dates, storms, run, control):
    """Grandeurs par tronçon (DataFrame) d'une variante et de son témoin, chacun (latéral, etr)."""
    lateral, etr = run
    lateral0, _ = control
    out = {}
    for _, s in storms.iterrows():
        t = int(np.flatnonzero(dates == pd.Timestamp(s.date))[0])
        key = f"{'humide' if s.apres_pluie else 'sec'}_{int(s.pluie_mm)}"
        out[f"coef_{key}"] = (lateral[t:t + 7] - lateral0[t:t + 7]).sum(axis=0) / s.pluie_mm
        out[f"riviere_20j_{key}"] = (lateral[t:t + 20] - lateral0[t:t + 20]).sum(axis=0) / s.pluie_mm
    dry = storms[~storms.apres_pluie]
    x = np.log(dry.pluie_mm.to_numpy())
    y = np.log(np.clip(np.stack([out[f"coef_sec_{int(m)}"] for m in dry.pluie_mm]), 1e-6, None))
    out["pente_pluie"] = ((x - x.mean())[:, None] * (y - y.mean(axis=0))).sum(axis=0) / ((x - x.mean()) ** 2).sum()
    out["rapport_humide_sec_20"] = out["coef_humide_20"] / np.clip(out["coef_sec_20"], 1e-6, None)
    year2 = dates.year.min() + 1
    total = lateral0
    drought = (dates >= f"{year2}-07-15") & (dates <= f"{year2}-09-15")
    june = (dates >= f"{year2}-06-01") & (dates <= f"{year2}-06-30")
    q7 = np.lib.stride_tricks.sliding_window_view(total[drought], 7, axis=0).mean(axis=-1).min(axis=0)
    out["etiage_rapport"] = q7 / np.clip(total[june].mean(axis=0), 1e-6, None)
    out["etr_mm"] = etr[dates.year == year2].sum(axis=0)
    # Calendrier de l'écoulement de la deuxième année, sans orage fabriqué (témoin) : part de
    # l'écoulement annuel en avril-mai (fonte) et du 1er juin au 14 juillet (début d'été).
    year = dates.year == year2
    annual = np.clip(total[year].sum(axis=0), 1e-6, None)
    spring = (dates >= f"{year2}-04-01") & (dates <= f"{year2}-05-31")
    early = (dates >= f"{year2}-06-01") & (dates <= f"{year2}-07-14")
    out["part_avril_mai"] = total[spring].sum(axis=0) / annual
    out["part_juin_juillet"] = total[early].sum(axis=0) / annual
    out["ecoulement_annuel_mm"] = total[year].sum(axis=0)
    return pd.DataFrame(out)


def _control_forcing(forcing, dates, storms):
    """Copie du forçage dont la pluie des jours d'orage est nulle."""
    from meandre.run.province_data import HostSeries
    days = np.flatnonzero(np.isin(dates.strftime("%Y-%m-%d"), storms.date.to_numpy()))
    if isinstance(forcing, HostSeries):
        host = np.array(forcing._host, dtype=np.float32, copy=True)
        host[days, :, 0] = 0.0
        return HostSeries(host, forcing._device)
    out = forcing.clone()
    out[days, :, 0] = 0.0
    return out


def apply_variant(model, v):
    """Pose une variante sur le modèle chargé ; rend la fonction qui la retire.

    Clés : `multipliers` (sorties du champ), `lake_k_factor` (facteur sur la constante de vidange des
    lacs routés), `settings` (réglages déclarés), `open_water_reservoir`
    (« T,beta,h_ref »), `lakes` (« none », ou table de clés de la section [lakes] des lacs hors réseau,
    par exemple `{ share_coefficient = 5.0, residence_days = 60.0 }`), `soil_params`
    (« FORME.cle » = valeur, les clés en _mm converties en mètres, sur les processus déclarés)."""
    from dataclasses import replace

    from meandre.vertical.soil_processes import M_PER_MM
    spatial_encoder = model.spatial_encoder
    fixed = dict(getattr(spatial_encoder, "multipliers", None) or {})
    mult = dict(fixed)
    for k, factor in (v.get("multipliers") or {}).items():
        mult[k] = mult.get(k, 1.0) * torch.tensor(float(factor))
    spatial_encoder.multipliers = mult or None
    lake_k_factor = getattr(spatial_encoder, "lake_k_factor", None)
    if v.get("lake_k_factor") is not None:
        spatial_encoder.lake_k_factor = float(v["lake_k_factor"])
    saved = {k: settings.get(k) for k in (v.get("settings") or {})}
    for k, value in (v.get("settings") or {}).items():
        settings.override(k, value, force=True)
    reservoir = getattr(model, "_open_water_reservoir", None)
    if v.get("open_water_reservoir"):
        T, beta, h_ref = (float(x) for x in v["open_water_reservoir"].split(","))
        model.set_open_water_reservoir(model._open_water_part, T, beta, h_ref, evap_lateral=True)
    lakes = getattr(model, "_distributed_lakes", None)
    lakes_runoff = getattr(model, "_lakes_runoff", None)
    lake_buffers = {n: getattr(model, n) for n in ("lakes_fixed_c", "lakes_fixed_T") if hasattr(model, n)}
    lake_learned = {n: getattr(model, n) for n in ("lakes_learn_c", "lakes_learn_T") if hasattr(model, n)}
    if v.get("lakes") == "none":
        model._distributed_lakes = None
    elif v.get("lakes"):
        # Table de clés de la section [lakes] (share_coefficient, residence_days...) : posées le temps
        # de la variante, le reste vient de la recette ; rien n'est appris au banc.
        for k, value in v["lakes"].items():
            key = f"lakes.{k}"
            if key not in settings.SCHEMA:
                raise ValueError(f"variante {v.get('name')} : clé de lacs inconnue {k}")
            saved.setdefault(key, settings.get(key))
            settings.override(key, settings._from_file(key, value), force=True)
        # Un paramètre appris de la recette primerait sur la valeur fixe de la variante.
        for n in lake_learned:
            delattr(model, n)
        model.set_distributed_lakes_from_settings(model._open_water_f, model._lakes_forcing_P_mean, learn=())
    column = model.vertical_column
    profile = getattr(column, "soil_profile", None)
    if v.get("soil_params") and profile is not None:
        procs = []
        for proc in profile.processes:
            params = dict(proc.params)
            for key, value in v["soil_params"].items():
                form, name = key.split(".")
                if form == proc.form:
                    if name.endswith("_mm"):
                        params[name[:-3]] = float(value) * M_PER_MM
                    else:
                        params[name] = float(value)
            procs.append(replace(proc, params=params))
        column.soil_profile = type(profile)(layers=profile.layers, processes=tuple(procs))

    def undo():
        spatial_encoder.multipliers = fixed or None
        spatial_encoder.lake_k_factor = lake_k_factor
        for k, value in saved.items():
            settings.override(k, value, force=True)
        model._open_water_reservoir = reservoir
        model._distributed_lakes = lakes
        model._lakes_runoff = lakes_runoff
        for n, t in lake_buffers.items():
            setattr(model, n, t)
        for n, t in lake_learned.items():
            setattr(model, n, t)
        column.soil_profile = profile
    return undo


def run_real_variants(simulate_q, model, dates, station_ids, path, output, score=None):
    """Débits réels : chaque variante simulée en mémoire sur le forçage du pilote, débits aux stations
    de la période d'évaluation écrits comme `--dump-q` (q-<nom>.npz), sans recharger le domaine.

    simulate_q() -> (q_sim, q_obs) aux stations sur la période d'évaluation, tableaux (jours, stations)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for v in cfg["variant"]:
        t0 = time.perf_counter()
        undo = apply_variant(model, v)
        try:
            qs, qo = simulate_q()
        finally:
            undo()
        np.savez_compressed(f"{output}/q-{v['name']}.npz", q_sim=qs, q_obs=qo, station_ids=np.array(station_ids, dtype=object), dates=np.array([str(x)[:10] for x in dates]))
        print(f"[banc reel] {v['name']} : {time.perf_counter() - t0:.0f} s" + (f" | {score(qs, qo)}" if score else ""), flush=True)


def run_bench(simulate, model, forcing, dates, area_km2, path, output):
    """Simule chaque variante et son témoin ; écrit un parquet par variante et imprime les quantiles.

    simulate(forcing) -> diagnostics de la simulation ; le champ spatial du modèle porte les
    multiplicateurs.
    """
    spatial_encoder = model.spatial_encoder
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    storms = pd.read_csv(cfg["storms"])
    dates = pd.DatetimeIndex(dates)
    control = _control_forcing(forcing, dates, storms)
    fixed = dict(getattr(spatial_encoder, "multipliers", None) or {})
    results = {}
    for v in cfg["variant"]:
        t0 = time.perf_counter()
        mult = dict(fixed)
        for k, factor in (v.get("multipliers") or {}).items():
            mult[k] = mult.get(k, 1.0) * torch.tensor(float(factor))
        spatial_encoder.multipliers = mult or None
        saved = {k: settings.get(k) for k in (v.get("settings") or {})}
        for k, value in (v.get("settings") or {}).items():
            settings.override(k, value, force=True)
        reservoir = getattr(model, "_open_water_reservoir", None)
        if v.get("open_water_reservoir"):
            T, beta, h_ref = (float(x) for x in v["open_water_reservoir"].split(","))
            model.set_open_water_reservoir(model._open_water_part, T, beta, h_ref, evap_lateral=True)
        m = metrics(dates, storms, _series(simulate(forcing), area_km2), _series(simulate(control), area_km2))
        model._open_water_reservoir = reservoir
        for k, value in saved.items():
            settings.override(k, value, force=True)
        m.to_parquet(f"{output}/banc-{v['name']}.parquet")
        results[v["name"]] = m
        print(f"[banc] {v['name']} : {time.perf_counter() - t0:.0f} s", flush=True)
    spatial_encoder.multipliers = fixed or None
    first = cfg["variant"][0]["name"]
    medians = pd.DataFrame({k: m[COLUMNS].median() for k, m in results.items()}).T
    print(f"[banc] médianes sur {len(results[first])} tronçons")
    print(medians.round(3).to_string())
    effects = pd.DataFrame({k: (m[COLUMNS] - results[first][COLUMNS]).median() for k, m in results.items() if k != first}).T
    print(f"[banc] effet de chaque variante : médiane des différences appariées avec {first}")
    print(effects.round(3).to_string())
    effects.to_csv(f"{output}/effets.csv")
