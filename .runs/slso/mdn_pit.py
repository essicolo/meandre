"""PIT diagnostic for MDN checkpoint — analytic via mixture CDF.

Un seul forward du backbone (gelé → identique entre runs), un appel au
mixture_head.cdf(), et on a PIT(y_obs | x) en closed-form pour chaque obs.
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
OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)


def main(ckpt_path: str, val_start: str, val_end: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    init = torch.load(ckpt_path, map_location="cpu", weights_only=False)["init_kwargs"]
    m = HydroModel(**init).to(device)
    m.load(ckpt_path)
    m.eval()
    print(f"use_mixture_head={m.use_mixture_head}, K={m.mixture_n_components}", flush=True)

    cache = BasinCache(DB)
    h = cache.load(device=device)
    ds = xr.open_dataset(FORCING)
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    spinup_start = pd.Timestamp(val_start) - pd.Timedelta(days=365)
    mask_keep = (dt_all >= spinup_start) & (dt_all <= pd.Timestamp(val_end))
    fc = torch.from_numpy(fc_all[mask_keep]).to(device)
    dt = dt_all[mask_keep]
    doy = torch.tensor(dt.dayofyear.values, dtype=torch.long, device=device)
    withdrawals = cache.load_withdrawals(
        date_start=spinup_start.strftime("%Y-%m-%d"), date_end=val_end, device=device,
    )

    con = duckdb.connect(DB, read_only=True)
    st = con.execute("select node_idx, station_id from stations order by node_idx").fetchdf()
    ob = con.execute(
        f"select date, station_id, discharge as q from observations "
        f"where date between '{val_start}' and '{val_end}'"
    ).fetchdf()
    con.close()
    vm = (dt >= pd.Timestamp(val_start)) & (dt <= pd.Timestamp(val_end))
    vi = np.where(vm)[0]; d2t = {d: i for i, d in enumerate(dt[vi])}
    sn = st["node_idx"].values.astype(int)
    s2c = {s: i for i, s in enumerate(st["station_id"].values)}
    qo = np.full((len(vi), len(st)), np.nan, dtype=np.float32)
    for _, r in ob.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in d2t and r["station_id"] in s2c:
            qo[d2t[d], s2c[r["station_id"]]] = float(r["q"])
    print(f"Forcing : {fc.shape}, Val : {vm.sum()} jours x {len(sn)} stations", flush=True)
    print(f"Obs valides : {(~np.isnan(qo)).sum()}", flush=True)

    print("Forward (1 pass, backbone gelé)...", flush=True)
    with torch.no_grad():
        Q, _ = m.simulate(
            forcing=fc,
            initial_state=HydroState.zeros(h["n_nodes"], device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=withdrawals, day_of_year=doy,
        )
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
        sp_tensor = sp.to_tensor()                          # (N, F)

    # Extraire val period x stations
    Q_v = Q[torch.from_numpy(vm).to(device)][:, torch.from_numpy(sn).long().to(device)]  # (T, S)
    sp_st = sp_tensor[torch.from_numpy(sn).long().to(device)]                            # (S, F)
    qo_t = torch.from_numpy(qo).to(device)
    valid = ~torch.isnan(qo_t) & ~torch.isnan(Q_v)
    t_idx, s_idx = torch.where(valid)
    y = qo_t[t_idx, s_idx]
    features = sp_st[s_idx]                                  # (M, F)
    q_sim = Q_v[t_idx, s_idx]                                # (M,)

    print(f"Computing PIT analytique via mixture CDF (M={len(y)} obs)...", flush=True)
    with torch.no_grad():
        pit = m.mixture_head.cdf(y, features, q_sim).cpu().numpy()
        log_prob = m.mixture_head.log_prob(y, features, q_sim).cpu().numpy()
        crps = m.mixture_head.crps_gaussian_mixture(y, features, q_sim).cpu().numpy()

    # Diagnostic PIT
    print(f"\n=== PIT analytique (mixture, K={m.mixture_n_components}) ===")
    print(f"  N obs       : {len(pit)}")
    print(f"  PIT mean    : {pit.mean():.4f}  (cible 0.5)")
    print(f"  PIT std     : {pit.std():.4f}   (cible 0.289)")
    H, _ = np.histogram(pit, bins=20, range=(0, 1))
    f = H / H.sum()
    d2 = float(((f - 1/20) ** 2).mean() / (1/20) ** 2)
    print(f"  delta^2     : {d2:.4f}  (cible 0)")
    print(f"  frac<0.05   : {(pit < 0.05).mean():.4f}  (cible 0.05)")
    print(f"  frac<0.10   : {(pit < 0.10).mean():.4f}  (cible 0.10)")
    print(f"  frac<0.25   : {(pit < 0.25).mean():.4f}  (cible 0.25)")
    print(f"  frac IQR    : {((pit >= 0.25) & (pit <= 0.75)).mean():.4f}  (cible 0.5)")
    print(f"  frac>0.75   : {(pit > 0.75).mean():.4f}  (cible 0.25)")
    print(f"  frac>0.90   : {(pit > 0.90).mean():.4f}  (cible 0.10)")
    print(f"  frac>0.95   : {(pit > 0.95).mean():.4f}  (cible 0.05)")
    print(f"\n  NLL moy    : {-log_prob.mean():.4f}")
    print(f"  CRPS moy   : {crps.mean():.4f} m^3/s")
    print(f"  CRPS médian: {np.median(crps):.4f}")

    # Couvertures depuis PIT
    cov_50 = float(((pit >= 0.25) & (pit <= 0.75)).mean())
    cov_90 = float(((pit >= 0.05) & (pit <= 0.95)).mean())
    print(f"\n  cov_50 = {cov_50:.4f}  (cible 0.50)")
    print(f"  cov_90 = {cov_90:.4f}  (cible 0.90)")

    # ── Stat additionnelle : Kolmogorov-Smirnov vs uniforme ──────────────
    from scipy.stats import kstest
    ks_stat, ks_pval = kstest(pit, "uniform")
    print(f"\n  Kolmogorov-Smirnov vs uniforme : D = {ks_stat:.4f}, p = {ks_pval:.2e}")

    # ── PIT stratifié par régime de Q_obs (quantile climato) ─────────────
    q_thirds = np.quantile(y.cpu().numpy(), [1/3, 2/3])
    y_np = y.cpu().numpy()
    regime = np.zeros(len(pit), dtype=np.int8)
    regime[y_np > q_thirds[0]] = 1
    regime[y_np > q_thirds[1]] = 2
    regime_labels = [f"étiage (Q<{q_thirds[0]:.1f})",
                     f"moyen ({q_thirds[0]:.1f}<Q<{q_thirds[1]:.1f})",
                     f"pic (Q>{q_thirds[1]:.1f})"]

    # ── Figure multi-panneaux ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) Histogramme PIT principal
    ax = axes[0, 0]
    ax.hist(pit, bins=20, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
    ax.axhline(len(pit)/20, ls="--", c="k", lw=1.5, label=f"Uniforme ({len(pit)/20:.0f})")
    ax.set_xlabel("PIT u"); ax.set_ylabel("Effectif")
    ax.set_title(f"PIT global — d²={d2:.3f}  KS={ks_stat:.3f}")
    ax.legend(loc="upper center")
    txt = (f"mean={pit.mean():.3f}\nstd={pit.std():.3f}\n"
           f"cov50={cov_50:.3f}\ncov90={cov_90:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # (2) Courbe de fiabilité : CDF empirique du PIT vs idéal uniforme
    ax = axes[0, 1]
    pit_sorted = np.sort(pit)
    ecdf = np.arange(1, len(pit) + 1) / len(pit)
    ax.plot(pit_sorted, ecdf, color="steelblue", lw=1.5, label="Empirique")
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1.5, label="Idéal uniforme")
    # Bandes de confiance Kolmogorov 95% (Dvoretzky-Kiefer-Wolfowitz)
    eps = np.sqrt(np.log(2/0.05) / (2 * len(pit)))
    ax.fill_between(pit_sorted, np.clip(ecdf - eps, 0, 1),
                    np.clip(ecdf + eps, 0, 1), color="gray", alpha=0.2,
                    label=f"95% DKW (±{eps:.3f})")
    ax.set_xlabel("PIT u"); ax.set_ylabel("Fraction cumulée")
    ax.set_title("Diagramme de fiabilité")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # (3) PIT stratifié par régime de Q_obs (3 panneaux superposés en histo)
    ax = axes[1, 0]
    colors = ["#2ca25f", "#feb24c", "#de2d26"]  # vert / orange / rouge
    for r, lbl, col in zip([0, 1, 2], regime_labels, colors):
        sel = regime == r
        ax.hist(pit[sel], bins=20, range=(0, 1), color=col, alpha=0.55,
                edgecolor="k", label=f"{lbl}  n={sel.sum()}")
    ax.axhline(0, ls="--", c="k", lw=0.5)
    # Référence uniforme par tiers de données
    n_third_uniform = len(pit) / 3 / 20
    ax.axhline(n_third_uniform, ls=":", c="k", lw=1.0,
               label=f"Uniforme par tiers ({n_third_uniform:.0f})")
    ax.set_xlabel("PIT u"); ax.set_ylabel("Effectif par régime")
    ax.set_title("PIT par régime de débit")
    ax.legend(fontsize=8, loc="upper center")

    # (4) δ² par régime + recap
    ax = axes[1, 1]
    d2_per_regime = []
    for r in [0, 1, 2]:
        sel = regime == r
        if sel.sum() < 50: d2_per_regime.append(np.nan); continue
        H_r, _ = np.histogram(pit[sel], bins=20, range=(0, 1))
        f_r = H_r / H_r.sum()
        d2_r = float(((f_r - 1/20) ** 2).mean() / (1/20) ** 2)
        d2_per_regime.append(d2_r)
    bars = ax.bar(["étiage", "moyen", "pic"], d2_per_regime, color=colors,
                  edgecolor="k", alpha=0.85)
    ax.axhline(d2, ls="--", c="k", label=f"δ² global = {d2:.3f}")
    for b, v in zip(bars, d2_per_regime):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=10)
    ax.set_ylabel("δ² (0 = uniforme parfait)")
    ax.set_title("Coefficient de non-uniformité par régime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"MDN PIT — val {val_start[:4]}–{val_end[:4]} — K=10 composantes — "
                 f"N={len(pit)} obs", fontsize=12)
    out_png = OUT / "mdn_pit_histogram.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nPNG : {out_png}", flush=True)

    print(f"\nδ² par régime :")
    for lbl, d2_r in zip(["étiage", "moyen", "pic"], d2_per_regime):
        print(f"  {lbl:8s} : {d2_r:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=".runs/slso/checkpoints/best-mdn.pt")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2021-12-31")
    args = p.parse_args()
    main(args.ckpt, args.start, args.end)
