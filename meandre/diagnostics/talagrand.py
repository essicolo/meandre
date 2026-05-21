"""Compute PIT (Probability Integral Transform) ranks and calibration metrics
for the probabilistic model.

For a Gaussian forecast N(Q_sim, sigma^2), the PIT value of an observation y is
    u = Phi((y - Q_sim) / sigma) in [0, 1].
A well-calibrated forecast produces u ~ Uniform(0, 1), so the histogram of u
binned into K+1 bins is the analog of the Talagrand rank histogram for a
K-member ensemble. Using PIT (analytical) avoids sampling noise.

Usage as a module:
    from meandre.diagnostics.talagrand import compute_pit_data
    data = compute_pit_data(".runs/slso/config/slso-probabilistic.toml")

Usage as a script:
    python -m meandre.diagnostics.talagrand .runs/slso/config/slso-probabilistic.toml
"""
from __future__ import annotations

import sys
from pathlib import Path

import tomllib
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr  # standard normal CDF, vectorised
from scipy.stats import chi2 as chi2_dist


def compute_pit_data(cfg_path: str | Path, device: str | None = None) -> dict:
    """Run a probabilistic simulation and return PIT ranks + raw arrays.

    Returns a dict with:
      - 'pit': (N,) float in [0, 1], PIT values for valid (t, station) pairs
      - 'pit_per_station': dict[str, np.ndarray] PIT values per station_id
      - 'q_sim': (T_val, n_stations) numpy
      - 'q_obs': (T_val, n_stations) numpy with NaN for missing
      - 'sigma': (T_val, n_stations) numpy
      - 'station_ids': list of station identifiers in column order
    """
    from meandre.model import HydroModel
    from meandre.data.basin_cache import BasinCache
    from meandre.data.gridded_forcing import extract_forcing
    from meandre.utils.state import HydroState

    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    date_start_str = cfg["temporal"]["date_start"]
    date_end_str = cfg["temporal"]["date_end"]
    DATE_START = np.datetime64(date_start_str)
    DATE_END = np.datetime64(date_end_str)
    VAL_START = np.datetime64(cfg["temporal"]["val_start"])
    VAL_END = np.datetime64(cfg["temporal"]["val_end"])
    val_sl = slice(int((VAL_START - DATE_START) / np.timedelta64(1, "D")),
                   int((VAL_END - DATE_START) / np.timedelta64(1, "D")))

    ckpt_path = cfg["paths"]["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    init_kw = dict(ckpt["init_kwargs"])
    init_kw["concrete_dropout"] = cfg["model"].get("concrete_dropout", False)
    init_kw["concrete_init_p"] = cfg["model"].get("concrete_init_p", 0.05)

    model = HydroModel(**init_kw)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(dev)
    model.eval()

    cache = BasinCache(cfg["paths"]["basin_db"])
    hydro = cache.load(device=dev)

    forcing_data = extract_forcing(
        zarr_path=cfg["paths"]["weather_grid"],
        node_coords=hydro["node_coords"],
        node_elev=None,
        date_start=date_start_str,
        date_end=date_end_str,
        cache_nc=Path(cfg["paths"]["forcing_cache"]),
        device=dev,
    )

    n_nodes = hydro["n_nodes"]
    obs = cache.load_observations(
        date_start=date_start_str,
        date_end=date_end_str,
        min_valid_days=365,
    )
    station_node_map = obs["station_node_map"]
    station_ids_sorted = sorted(station_node_map.keys(), key=lambda s: station_node_map[s])
    station_indices = [station_node_map[s] for s in station_ids_sorted]

    q_obs_all = torch.from_numpy(obs["discharge"][:, station_indices]).to(dev)
    q_obs_val = q_obs_all[val_sl]

    withdrawals = cache.load_withdrawals(
        date_start=date_start_str,
        date_end=date_end_str,
        device=dev,
    )

    dates = pd.date_range(date_start_str, date_end_str, freq="D")
    doy = torch.tensor([d.day_of_year for d in dates], dtype=torch.long, device=dev)

    with torch.no_grad():
        state0 = HydroState.zeros(n_nodes, device=dev)
        Q_all, _ = model.simulate(
            forcing=forcing_data,
            initial_state=state0,
            graph=hydro["graph"],
            node_coords=hydro["node_coords"],
            territorial=hydro["territorial"],
            withdrawals=withdrawals,
            day_of_year=doy,
        )
        Q_val = Q_all[val_sl.start:val_sl.stop]

        sp = model.spatial_encoder(hydro["node_coords"], hydro["territorial"].to_tensor())
        log_sigma = model.noise_head(sp.to_tensor(), Q_val.detach())
        sigma_full = log_sigma.exp()

    q_sim_val = Q_val[:, station_indices].cpu().numpy()
    sigma_val = sigma_full[:, station_indices].cpu().numpy()
    q_obs_np = q_obs_val.cpu().numpy()

    # PIT: u = Phi((y - mu) / sigma). NaN if obs missing or sigma not finite.
    z = (q_obs_np - q_sim_val) / np.maximum(sigma_val, 1e-9)
    pit_2d = ndtr(z)
    valid = ~np.isnan(q_obs_np) & np.isfinite(pit_2d)
    pit_2d_valid = np.where(valid, pit_2d, np.nan)

    pit_per_station = {
        sid: pit_2d_valid[:, j][~np.isnan(pit_2d_valid[:, j])]
        for j, sid in enumerate(station_ids_sorted)
    }

    return {
        "pit": pit_2d_valid[~np.isnan(pit_2d_valid)],
        "pit_per_station": pit_per_station,
        "q_sim": q_sim_val,
        "q_obs": q_obs_np,
        "sigma": sigma_val,
        "station_ids": station_ids_sorted,
    }


def pit_histogram(pit: np.ndarray, n_bins: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Bin PIT values into n_bins equal-width bins covering [0, 1].

    Returns (counts, edges). With n_bins = K+1, this matches the convention
    of a Talagrand histogram for a K-member ensemble.
    """
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return counts, edges


def flatness_metrics(counts: np.ndarray) -> dict:
    """Compute calibration metrics from a PIT histogram.

    - delta : RMS deviation from uniform, normalised by the uniform value.
              0 = perfect flat, > 0.5 = poor calibration.
    - chi2  : Pearson chi-square statistic vs uniform.
    - p_value : right-tail p-value under H0 (uniform).
    - mean_pit, ideal_mean : sanity-check the central tendency (ideal = 0.5).
    """
    n_bins = len(counts)
    n_total = counts.sum()
    expected = n_total / n_bins
    rel_freq = counts / n_total
    uniform = 1.0 / n_bins
    delta = float(np.sqrt(np.mean((rel_freq - uniform) ** 2)) / uniform)
    chi2_stat = float(((counts - expected) ** 2 / expected).sum())
    dof = n_bins - 1
    p_value = float(1.0 - chi2_dist.cdf(chi2_stat, dof))
    return {
        "delta": delta,
        "chi2": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "expected_per_bin": float(expected),
        "n_total": int(n_total),
        "n_bins": int(n_bins),
    }


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/slso/config/slso-probabilistic.toml"
    data = compute_pit_data(cfg_path)
    counts, edges = pit_histogram(data["pit"], n_bins=21)
    metrics = flatness_metrics(counts)

    print(f"\n{'='*60}")
    print(f"PIT histogram ({metrics['n_total']} valid obs, {metrics['n_bins']} bins)")
    print(f"{'='*60}")
    expected = metrics["expected_per_bin"]
    for i, c in enumerate(counts):
        bin_lo = edges[i]
        bin_hi = edges[i + 1]
        rel = c / expected
        bar = "#" * int(min(rel, 5) * 10)
        print(f"  [{bin_lo:.2f}, {bin_hi:.2f}] : {c:6d}  ({rel:.2f}x)  {bar}")

    print(f"\n{'='*60}")
    print(f"Calibration metrics")
    print(f"{'='*60}")
    print(f"Flatness delta : {metrics['delta']:.4f}   (0 = perfect, >0.5 = poor)")
    print(f"chi2           : {metrics['chi2']:.2f} (dof={metrics['dof']}), p = {metrics['p_value']:.4f}")
    print(f"mean PIT       : {data['pit'].mean():.4f}   (ideal: 0.5)")
    if data["pit"].mean() < 0.45:
        print("  -> Bias: obs systematically below mean forecast (Q_sim trop élevé)")
    elif data["pit"].mean() > 0.55:
        print("  -> Bias: obs systematically above mean forecast (Q_sim trop bas)")

    print(f"\nPer-station flatness delta:")
    rows = []
    for sid, p in data["pit_per_station"].items():
        if len(p) < 30:
            continue
        c, _ = pit_histogram(p, n_bins=21)
        m = flatness_metrics(c)
        rows.append((sid, m["delta"], len(p)))
    rows.sort(key=lambda r: r[1])
    for sid, d, n in rows:
        flag = "✓" if d < 0.3 else ("~" if d < 0.6 else "✗")
        print(f"  {flag} {sid}  delta={d:.3f}  (n={n})")
