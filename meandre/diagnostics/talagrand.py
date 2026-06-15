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
    from meandre.utils.paths import run_dir_from_config, resolve_run_path

    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    run_dir = run_dir_from_config(cfg_path)
    def _p(key: str) -> Path:
        return resolve_run_path(cfg["paths"][key], run_dir)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    date_start_str = cfg["temporal"]["date_start"]
    date_end_str = cfg["temporal"]["date_end"]
    DATE_START = np.datetime64(date_start_str)
    DATE_END = np.datetime64(date_end_str)
    VAL_START = np.datetime64(cfg["temporal"]["val_start"])
    VAL_END = np.datetime64(cfg["temporal"]["val_end"])
    val_sl = slice(int((VAL_START - DATE_START) / np.timedelta64(1, "D")),
                   int((VAL_END - DATE_START) / np.timedelta64(1, "D")))

    ckpt_path = _p("checkpoint")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    init_kw = dict(ckpt["init_kwargs"])
    init_kw["concrete_dropout"] = cfg["model"].get("concrete_dropout", False)
    init_kw["concrete_init_p"] = cfg["model"].get("concrete_init_p", 0.05)

    model = HydroModel(**init_kw)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(dev)
    model.eval()

    cache = BasinCache(_p("basin_db"))
    hydro = cache.load(device=dev)

    forcing_data = extract_forcing(
        zarr_path=_p("weather_grid"),
        node_coords=hydro["node_coords"],
        node_elev=None,
        date_start=date_start_str,
        date_end=date_end_str,
        cache_nc=_p("forcing_cache"),
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

    # Distribution probabiliste utilisée à l'entraînement → espace du PIT.
    lcfg_pit = cfg.get("loss", {})
    dist = str(lcfg_pit.get("nll_distribution", "normal")).lower()
    if dist == "normal":
        nll_lambda = 1.0
    elif dist == "log-normal":
        nll_lambda = 0.0
    elif dist == "box-cox":
        nll_lambda = float(lcfg_pit.get("nll_box_cox_lambda", 0.3))
    else:
        nll_lambda = 1.0

    def _box_cox(x, lam, eps=1e-3):
        x_safe = np.maximum(x, eps)
        if lam == 0.0:
            return np.log(x_safe)
        if lam == 1.0:
            return x_safe - 1.0
        return (x_safe ** lam - 1.0) / lam

    # PIT « training-space » : cohérent avec la NLL utilisée.
    # σ est interprété dans l'espace transformé (sortie directe du noise_head).
    q_obs_t = _box_cox(q_obs_np, nll_lambda)
    q_sim_t = _box_cox(q_sim_val, nll_lambda)
    z = (q_obs_t - q_sim_t) / np.maximum(sigma_val, 1e-9)
    pit_2d = ndtr(z)
    valid = ~np.isnan(q_obs_np) & np.isfinite(pit_2d)
    pit_2d_valid = np.where(valid, pit_2d, np.nan)

    # PIT log-normal what-if (delta-method) — utile seulement si entraîné en
    # « normal ». Sinon redondant, mais on le garde pour comparaison.
    eps_log = 1e-3
    mu_pos = np.maximum(q_sim_val, eps_log)
    var_rel = (sigma_val / mu_pos) ** 2
    sigma_log = np.sqrt(np.log1p(var_rel))
    mu_log = np.log(mu_pos) - 0.5 * sigma_log ** 2
    y_log = np.log(np.maximum(q_obs_np, eps_log))
    z_log = (y_log - mu_log) / np.maximum(sigma_log, 1e-9)
    pit_log_2d = ndtr(z_log)
    valid_log = ~np.isnan(q_obs_np) & np.isfinite(pit_log_2d)
    pit_log_2d_valid = np.where(valid_log, pit_log_2d, np.nan)

    pit_per_station = {
        sid: pit_2d_valid[:, j][~np.isnan(pit_2d_valid[:, j])]
        for j, sid in enumerate(station_ids_sorted)
    }
    pit_log_per_station = {
        sid: pit_log_2d_valid[:, j][~np.isnan(pit_log_2d_valid[:, j])]
        for j, sid in enumerate(station_ids_sorted)
    }

    return {
        "pit": pit_2d_valid[~np.isnan(pit_2d_valid)],
        "pit_per_station": pit_per_station,
        "pit_lognormal": pit_log_2d_valid[~np.isnan(pit_log_2d_valid)],
        "pit_lognormal_per_station": pit_log_per_station,
        "q_sim": q_sim_val,
        "q_obs": q_obs_np,
        "sigma": sigma_val,
        "sigma_log": sigma_log,
        "station_ids": station_ids_sorted,
        "nll_distribution": dist,
        "nll_lambda": nll_lambda,
    }


def pit_histogram(pit: np.ndarray, n_bins: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Bin PIT values into n_bins equal-width bins covering [0, 1].

    Returns (counts, edges). With n_bins = K+1, this matches the convention
    of a Talagrand histogram for a K-member ensemble.
    """
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return counts, edges


def candille_talagrand(
    pit_matrix: np.ndarray,
    n_bins: int = 20,
    block_len: int = 30,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Écart à la platitude normalisé à la Candille-Talagrand (2005).

    Candille & Talagrand définissent Δ = Σ_i (n_i − N/B)² et le rapportent à
    son espérance sous fiabilité parfaite Δ0 = N(B−1)/B ; le rapport
    δ = Δ/Δ0 vaut 1 pour une prévision parfaitement fiable (l'écart observé
    est alors entièrement du bruit d'échantillonnage), et δ >> 1 signale une
    vraie déviation de l'uniformité.

    L'hypothèse i.i.d. derrière Δ0 est fausse pour des débits journaliers
    (autocorrélation temporelle + corrélation inter-stations) : Δ0 sous-estime
    le bruit et δ_iid devient anticonservateur. Correction par bootstrap par
    blocs : les marginales sont d'abord uniformisées par rangs PAR station
    (impose H0 en préservant la dépendance en rangs), puis des blocs temporels
    entiers, communs à toutes les stations (la corrélation spatiale est donc
    conservée), sont rééchantillonnés pour estimer la distribution de δ_iid
    sous H0 avec dépendance.

    Args:
        pit_matrix: (T, S) valeurs PIT, NaN pour les manquants. Un vecteur 1-D
            est accepté (traité comme une seule série temporelle).
        n_bins: nombre de classes (20 = convention pit_metrics).
        block_len: longueur des blocs temporels (jours) ; ~30 j couvre
            l'autocorrélation hydrologique courante.
        n_boot: nombre de rééchantillonnages bootstrap.
        seed: graine du générateur.

    Returns:
        dict : delta_iid (rapport C&T sous i.i.d., cible 1), delta_eff
        (corrigé de la dépendance, cible 1), tau (facteur d'inflation de
        variance = E[δ_iid | H0, dépendance]), p_value (bootstrap, unilatéral
        droit), n, n_bins.
    """
    from scipy.stats import rankdata

    pit = np.asarray(pit_matrix, dtype=float)
    if pit.ndim == 1:
        pit = pit[:, None]
    T, S = pit.shape
    valid = np.isfinite(pit)
    pooled = pit[valid]
    N = pooled.size
    B = int(n_bins)
    if N < 10 * B:
        raise ValueError(f"trop peu de PIT valides ({N}) pour {B} classes")

    counts, _ = np.histogram(pooled, bins=B, range=(0.0, 1.0))
    delta_obs = float(((counts - N / B) ** 2).sum())
    delta0_iid = N * (B - 1) / B
    delta_iid = delta_obs / delta0_iid

    # H0 par rangs : marginales uniformes par station, dépendance préservée.
    U = np.full_like(pit, np.nan)
    for s in range(S):
        m = valid[:, s]
        n_s = int(m.sum())
        if n_s > 0:
            U[m, s] = rankdata(pit[m, s]) / (n_s + 1.0)

    bl = int(max(1, min(block_len, T)))
    n_blocks = int(np.ceil(T / bl))
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, T - bl + 1, size=n_blocks)
        rows = np.concatenate([np.arange(s0, s0 + bl) for s0 in starts])[:T]
        ub = U[rows]
        ub = ub[np.isfinite(ub)]
        Nb = ub.size
        cb, _ = np.histogram(ub, bins=B, range=(0.0, 1.0))
        ratios[b] = float(((cb - Nb / B) ** 2).sum()) / (Nb * (B - 1) / B)

    tau = float(ratios.mean())
    delta_eff = delta_iid / max(tau, 1e-12)
    p_value = float((1 + int((ratios >= delta_iid).sum())) / (n_boot + 1))
    return {
        "delta_iid": float(delta_iid),
        "delta_eff": float(delta_eff),
        "tau": tau,
        "p_value": p_value,
        "n": int(N),
        "n_bins": B,
    }


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

    # Diagnostic log-normal en parallèle
    counts_log, _ = pit_histogram(data["pit_lognormal"], n_bins=21)
    metrics_log = flatness_metrics(counts_log)
    print(f"\n{'='*60}")
    print(f"PIT log-normal (delta-method) — {metrics_log['n_total']} obs")
    print(f"{'='*60}")
    print(f"Flatness delta : {metrics_log['delta']:.4f}   (cf linéaire: {metrics['delta']:.4f})")
    print(f"chi2           : {metrics_log['chi2']:.2f}, p = {metrics_log['p_value']:.4f}")
    print(f"mean PIT       : {data['pit_lognormal'].mean():.4f}")

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
