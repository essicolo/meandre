"""Largeur de l'enveloppe probabiliste sur tous les tronçons du domaine provincial.

La tête de quantiles est indépendante par tronçon : elle lit les paramètres du champ spatial du
tronçon et son débit du jour, et rend six écarts à la médiane. Le pilote ne l'évalue qu'aux
stations, l'enveloppe journalière des 27 834 tronçons ne tenant pas en mémoire. Ici, elle est
appliquée tronçon par tronçon, par blocs, sur les débits journaliers de la passe provinciale, et
seul un résumé par tronçon est gardé : largeur relative médiane des intervalles à 50 et 90 % sur la
période demandée, et la même largeur sur les jours d'étiage (débit sous le 20e centile du tronçon).

Les paramètres du champ sont lus dans le cache par tronçon de la passe (clés param_<nom>), dans
l'ordre des sorties du champ écrit dans le point de reprise (spatial_field_names) : c'est l'ordre
que la tête a vu à l'entraînement. Le contrôle, aux tronçons qui portent une station, compare les
quantiles ainsi obtenus à ceux que le pilote a écrits dans son fichier de débits aux stations.

Usage : python envelope_by_reach.py <point de reprise probabiliste .pt> <cache par tronçon .npz>
            <débits journaliers par tronçon .npz> <sortie .npz> [--check <q aux stations .npz>]
            [--start 2022-01-01 --end 2024-12-31]
"""
import argparse

import numpy as np
import torch

from meandre.utils.quantile_head import QuantileHead

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("reach_cache")
parser.add_argument("daily")
parser.add_argument("output")
parser.add_argument("--check")
parser.add_argument("--start", default="2022-01-01")
parser.add_argument("--end", default="2024-12-31")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
recipe = checkpoint["recipe"]
quantile_settings = recipe.get("settings", {}).get("quantile", {}) if isinstance(recipe, dict) else {}
head = QuantileHead(n_spatial_params=len(checkpoint["spatial_field_names"]), taus=tuple(checkpoint["init_kwargs"]["quantile_taus"]), log_step=float(quantile_settings.get("log_step", 0.05)), log_bound=float(quantile_settings.get("log_bound", 4.0)))
head.load_state_dict({k.removeprefix("quantile_head."): v for k, v in checkpoint["state_dict"].items() if k.startswith("quantile_head.")})
head.eval()

cache = np.load(args.reach_cache)
params = torch.tensor(np.stack([cache[f"param_{name}"] for name in checkpoint["spatial_field_names"]], axis=1), dtype=torch.float32)
daily = np.load(args.daily)
dates = np.array(daily["dates"], dtype="datetime64[D]")
kept = (dates >= np.datetime64(args.start)) & (dates <= np.datetime64(args.end))
q = daily["q"][kept]
n_days, n_reaches = q.shape
print(f"{n_reaches} tronçons, {n_days} jours de {args.start} à {args.end}, {params.shape[1]} paramètres par tronçon")

taus = list(head.taus)
low_90, low_50, high_50, high_90 = taus.index(0.05), taus.index(0.25), taus.index(0.75), taus.index(0.95)
width_90 = np.empty(n_reaches, dtype=np.float32)
width_50 = np.empty(n_reaches, dtype=np.float32)
width_90_low_flow = np.empty(n_reaches, dtype=np.float32)
block = 500
with torch.no_grad():
    for a in range(0, n_reaches, block):
        b = min(a + block, n_reaches)
        median = torch.tensor(q[:, a:b], dtype=torch.float32)
        offsets = head(params[a:b], median).numpy()
        median = median.numpy()
        relative_90 = (offsets[:, :, high_90] - offsets[:, :, low_90]) / np.maximum(median, 1e-6)
        relative_50 = (offsets[:, :, high_50] - offsets[:, :, low_50]) / np.maximum(median, 1e-6)
        width_90[a:b] = np.median(relative_90, axis=0)
        width_50[a:b] = np.median(relative_50, axis=0)
        low_flow_days = median <= np.percentile(median, 20, axis=0, keepdims=True)
        width_90_low_flow[a:b] = np.array([np.median(relative_90[low_flow_days[:, j], j]) for j in range(b - a)])
        if args.check and a == 0:
            # Contrôle aux stations : les écarts du pilote et ceux calculés ici doivent coïncider.
            stations = np.load(args.check, allow_pickle=True)
            station_dates = np.array([str(d)[:10] for d in stations["dates"]], dtype="datetime64[D]")
            common = np.isin(dates[kept], station_dates)
            idx = cache["station_idx"]
            check_offsets = head(params[idx], torch.tensor(q[common][:, idx], dtype=torch.float32)).numpy()
            pilot_offsets = stations["q_quantiles"][np.isin(station_dates, dates[kept])]
            gap = np.nanmax(np.abs(check_offsets - pilot_offsets) / np.maximum(np.abs(pilot_offsets), 1e-6))
            print(f"contrôle aux {len(idx)} stations : écart relatif maximal aux quantiles du pilote {gap:.2e}")

np.savez_compressed(args.output, reach=np.arange(n_reaches, dtype=np.uint32), width_90=width_90, width_50=width_50, width_90_low_flow=width_90_low_flow)
print(f"largeur relative médiane de l'intervalle à 90 % : médiane des tronçons {np.median(width_90):.2f}, à 50 % {np.median(width_50):.2f} ; à l'étiage {np.median(width_90_low_flow):.2f}")
print(f"écrit : {args.output}")
