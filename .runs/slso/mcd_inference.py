"""MC dropout inference on the trained MCD-quantile checkpoint.

Méthode CORRECTE :
  - Chaque forward i produit K quantile POINTS définissant une CDF F_i(y)
  - Distribution prédictive totale : F_total(y) = (1/N) Σ_i F_i(y)
  - PIT(y_obs) = F_total(y_obs), interpolation linéaire par forward
  - Médiane prédictive : médiane sur les N forwards de Q_sim (= q_0.5 par construction)
  - Couvertures : fraction d'obs avec PIT dans [α, 1-α]

Erreur précédente : poolé tous les N×K quantiles dans un échantillon, pris
quantile empirique. Ça mélange points de CDF différents et donne KGE pourri.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import duckdb
import matplotlib.pyplot as plt

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
CK_DEFAULT = ".runs/slso/checkpoints/best-mcd-quantile.pt"
OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Met TOUT en eval(), puis réactive seulement les couches de dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
        if m.__class__.__name__ in ("ConcreteDropout", "ConcreteDropoutLayer"):
            m.train()


def load_obs(start: str, end: str):
    con = duckdb.connect(DB, read_only=True)
    st = con.execute("select node_idx, station_id from stations order by node_idx").fetchdf()
    ob = con.execute(
        f"select date, station_id, discharge as q from observations "
        f"where date between '{start}' and '{end}'"
    ).fetchdf()
    con.close()
    return st, ob


def to_matrix(dt, st, ob, start, end):
    vm = (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))
    vi = np.where(vm)[0]
    d2t = {d: i for i, d in enumerate(dt[vi])}
    s2c = {s: i for i, s in enumerate(st["station_id"].values)}
    qo = np.full((len(vi), len(st)), np.nan, dtype=np.float32)
    for _, r in ob.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in d2t and r["station_id"] in s2c:
            qo[d2t[d], s2c[r["station_id"]]] = float(r["q"])
    return vm, qo


def pit_total_vectorized(
    y: np.ndarray,                 # (M,) observations valides
    quants: np.ndarray,            # (M, N, K) quantiles : par obs, par forward, par tau
    taus: np.ndarray,              # (K,)
) -> np.ndarray:
    """PIT total F_total(y) = (1/N) Σ_i F_i(y), entièrement vectorisé.

    Pour chaque obs y et chaque forward i, on calcule F_i(y) par interpolation
    linéaire des K quantile points (q_i, τ). On moyenne ensuite sur N forwards.
    """
    M, N, K = quants.shape
    # F_i(y) pour chaque (m, i) en parallèle via interpolation broadcasting
    # On veut k tel que quants[m, i, k] <= y[m] < quants[m, i, k+1]
    # Approche : pour chaque forward (boucle i, mais petit), vectoriser sur m
    F = np.zeros((M, N), dtype=np.float32)
    y_col = y[:, None]                              # (M, 1)
    for i in range(N):
        qi = quants[:, i, :]                        # (M, K)
        # searchsorted ligne par ligne (M obs séparées)
        # Truc : utiliser une boucle vectorisée sur k.
        below = y_col <= qi[:, [0]]                 # (M, 1) y sous q_0.05
        above = y_col >= qi[:, [-1]]                # (M, 1) y au-dessus q_0.95
        # Interpolation au milieu via searchsorted vectorisé
        # np.searchsorted ne broadcaste pas, donc on fait via comparaison vectorielle
        # idx_below_y : nombre de quantiles <= y, donne le rang
        idx = (qi < y_col).sum(axis=1)              # (M,) entre 0 et K
        idx_clip = np.clip(idx, 1, K - 1) - 1       # k tel que qi[m, k] <= y < qi[m, k+1]
        k = idx_clip
        m_arr = np.arange(M)
        q_k = qi[m_arr, k]
        q_k1 = qi[m_arr, k + 1]
        tau_k = taus[k]
        tau_k1 = taus[k + 1]
        width = np.maximum(q_k1 - q_k, 1e-9)
        F_mid = tau_k + (tau_k1 - tau_k) * (y - q_k) / width
        # Extrémités
        F_low = np.clip(taus[0] * y / np.maximum(qi[:, 0], 1e-9), 0, taus[0])
        F_high = np.clip(
            taus[-1] + (1 - taus[-1]) * (y - qi[:, -1]) / np.maximum(qi[:, -1], 1e-9),
            taus[-1], 1.0,
        )
        F_i = np.where(below.ravel(), F_low, np.where(above.ravel(), F_high, F_mid))
        F[:, i] = F_i
    return F.mean(axis=1).clip(0, 1)               # (M,)


def main(ckpt_path: str, n_samples: int, val_start: str, val_end: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading {ckpt_path}", flush=True)
    init = torch.load(ckpt_path, map_location="cpu", weights_only=False)["init_kwargs"]
    m = HydroModel(**init).to(device)
    m.load(ckpt_path)
    print(f"use_quantile_head = {m.use_quantile_head}, taus = {m.quantile_head.taus}", flush=True)
    taus = np.array(m.quantile_head.taus)
    K = len(taus)

    cache = BasinCache(DB)
    h = cache.load(device=device)
    # Slice forcing 2000-01-01 → val_end (pas besoin de simuler après val_end)
    ds = xr.open_dataset(FORCING)
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    spinup_start = pd.Timestamp(val_start) - pd.Timedelta(days=365)  # 1 an de spinup
    mask_keep = (dt_all >= spinup_start) & (dt_all <= pd.Timestamp(val_end))
    fc = torch.from_numpy(fc_all[mask_keep]).to(device)
    dt = dt_all[mask_keep]
    print(f"Spinup start (1 an) : {spinup_start}", flush=True)
    doy = torch.tensor(dt.dayofyear.values, dtype=torch.long, device=device)
    print(f"Forcing : {fc.shape} (sliced jusqu'à {val_end})", flush=True)
    withdrawals = cache.load_withdrawals(
        date_start=spinup_start.strftime("%Y-%m-%d"), date_end=val_end, device=device,
    )

    st, ob = load_obs(val_start, val_end)
    vm, qo = to_matrix(dt, st, ob, val_start, val_end)
    sn = st["node_idx"].values.astype(int)
    n_val = int(vm.sum())
    n_st = len(sn)
    print(f"Val period: {val_start} .. {val_end} ({n_val} days, {n_st} stations)")
    print(f"Obs valides : {(~np.isnan(qo)).sum()} / {qo.size}")

    print(f"\nMC inference : N = {n_samples} forwards avec dropout actif...", flush=True)
    enable_mc_dropout(m)
    quant_samples = np.zeros((n_samples, n_val, n_st, K), dtype=np.float32)
    Q_samples = np.zeros((n_samples, n_val, n_st), dtype=np.float32)
    vm_dev = torch.from_numpy(vm).to(device)
    sn_dev = torch.from_numpy(sn).long().to(device)

    import time
    t_start = time.perf_counter()
    for i in range(n_samples):
        torch.manual_seed(i + 1000)
        with torch.no_grad():
            Q, _ = m.simulate(
                forcing=fc,
                initial_state=HydroState.zeros(h["n_nodes"], device=device),
                graph=h["graph"],
                node_coords=h["node_coords"],
                territorial=h["territorial"],
                withdrawals=withdrawals,
                day_of_year=doy,
            )
            sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
            offsets = m.quantile_head(sp.to_tensor(), Q)
            q_pred = Q.unsqueeze(-1) + offsets
        Q_samples[i] = Q[vm_dev][:, sn_dev].cpu().numpy()
        quant_samples[i] = q_pred[vm_dev][:, sn_dev, :].cpu().numpy()
        elapsed = time.perf_counter() - t_start
        rate = (i + 1) / elapsed
        eta = (n_samples - i - 1) / max(rate, 1e-9)
        print(f"  Sample {i+1}/{n_samples}  Q_mean={Q_samples[i].mean():.2f}  "
              f"std_epi={Q_samples[:i+1].std(axis=0).mean():.3f}  "
              f"elapsed={elapsed:.1f}s ETA={eta:.0f}s", flush=True)

    epist_var_Q = Q_samples.std(axis=0)  # std across forwards per (t, n)
    print(f"\nVariance épistémique sur Q (médiane par forward):")
    print(f"  std MOYENNE entre forwards : {epist_var_Q.mean():.3f} m^3/s")
    print(f"  std MAX                    : {epist_var_Q.max():.3f} m^3/s")
    print(f"  Q mean overall             : {Q_samples.mean():.2f} m^3/s")
    print(f"  Rapport std_epi / Q_mean   : {epist_var_Q.mean()/max(Q_samples.mean(),1e-3):.2%}")

    # PIT total vectorisé (moyenne des CDF par forward)
    print(f"\nCalcul du PIT total (vectorisé, {n_samples} forwards)...")
    valid = ~np.isnan(qo) & ~np.isnan(Q_samples).any(axis=0)
    # Indices (t, s) valides : on travaille en flat
    t_idx, s_idx = np.where(valid)                                # (M,) chacun
    M = len(t_idx)
    y_v = qo[t_idx, s_idx]                                        # (M,)
    # quants : (M, N, K) — pour chaque obs valide, les N×K quantiles
    quants_v = quant_samples[:, t_idx, s_idx, :]                   # (N, M, K)
    quants_v = np.transpose(quants_v, (1, 0, 2))                   # (M, N, K)
    pit_flat_arr = pit_total_vectorized(y_v, quants_v, taus)
    pit = np.full((n_val, n_st), np.nan, dtype=np.float32)
    pit[t_idx, s_idx] = pit_flat_arr

    pit_flat = pit[valid]
    print(f"\nPIT total sur {len(pit_flat)} obs valides :")
    print(f"  PIT mean  : {pit_flat.mean():.4f}  (cible 0.5)")
    print(f"  PIT std   : {pit_flat.std():.4f}  (cible 0.289)")
    H, _ = np.histogram(pit_flat, bins=20, range=(0, 1))
    f = H / H.sum()
    d2 = float(((f - 1/20) ** 2).mean() / (1/20) ** 2)
    print(f"  delta^2   : {d2:.4f}")
    print(f"  frac<0.05 : {(pit_flat < 0.05).mean():.4f}")
    print(f"  frac>0.95 : {(pit_flat > 0.95).mean():.4f}")
    print(f"  frac IQR  : {((pit_flat >= 0.25) & (pit_flat <= 0.75)).mean():.4f}  (cible 0.5)")
    print(f"  frac u>0.75 : {(pit_flat > 0.75).mean():.4f}  (cible 0.25)")

    # Couvertures depuis le PIT (la définition propre)
    cov_50 = float(((pit_flat >= 0.25) & (pit_flat <= 0.75)).mean())
    cov_90 = float(((pit_flat >= 0.05) & (pit_flat <= 0.95)).mean())
    print(f"\nCouvertures (depuis PIT) :")
    print(f"  cov_50 = {cov_50:.4f}  (cible 0.50)")
    print(f"  cov_90 = {cov_90:.4f}  (cible 0.90)")

    # KGE déterministe : médiane prédictive = médiane sur forwards de Q_sim
    # (car par construction q_0.5_i = Q_sim_i avec offsets cumsum signé)
    median_pred = np.median(Q_samples, axis=0)
    y_v = qo[valid]; m_v = median_pred[valid]
    from meandre.utils.metrics import kge as _kge
    kge_pooled = float(_kge(torch.from_numpy(y_v), torch.from_numpy(m_v)))
    print(f"\nKGE pooled (médiane MC = médiane sur forwards de Q_sim) : {kge_pooled:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit_flat, bins=20, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
    ax.axhline(len(pit_flat)/20, ls="--", c="k", label=f"Uniforme ({len(pit_flat)/20:.0f})")
    ax.set_xlabel("PIT u (moyenne CDF des forwards)")
    ax.set_ylabel("Effectif")
    ax.set_title(f"MC dropout PIT — val {val_start[:4]}-{val_end[:4]} — d2={d2:.3f} — N={n_samples}")
    ax.legend()
    out_png = OUT / "mcd_pit_histogram.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nPNG : {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CK_DEFAULT)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2021-12-31")
    args = p.parse_args()
    main(args.ckpt, args.n, args.start, args.end)
