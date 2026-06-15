"""Audit D : et si le PIT pourri venait du biais médiane β, pas de la densité ?

Protocole :
  1) Forward backbone sur train+val (1 forward, backbone gelé)
  2) β_n = mean(Q_sim_train) / mean(Q_obs_train) par station
  3) Q_sim_corrected = Q_sim / β_n appliqué sur val
  4) PIT recalculé avec Q_sim_corrected (via mixture head OU via simple Gaussien hétéro)

Si δ² s'effondre nettement → c'était la médiane, pas la densité.
Si δ² stable → la forme de la prédictive est vraiment le problème.

Sortie : .reports/slso/audit_beta_correction.txt + .png
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
from scipy.stats import kstest

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)


def main(ckpt_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    init = torch.load(ckpt_path, map_location="cpu", weights_only=False)["init_kwargs"]
    m = HydroModel(**init).to(device)
    m.load(ckpt_path)
    m.eval()
    print(f"use_mixture_head={m.use_mixture_head}, use_quantile_head={m.use_quantile_head}", flush=True)

    cache = BasinCache(DB)
    h = cache.load(device=device)
    # Forward sur TOUT le range : train (2001-2018) + val (2019-2021)
    ds = xr.open_dataset(FORCING)
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    # 1 an spinup + train + val
    mask_keep = (dt_all >= pd.Timestamp("2000-01-01")) & (dt_all <= pd.Timestamp("2021-12-31"))
    fc = torch.from_numpy(fc_all[mask_keep]).to(device)
    dt = dt_all[mask_keep]
    doy = torch.tensor(dt.dayofyear.values, dtype=torch.long, device=device)
    withdrawals = cache.load_withdrawals(
        date_start="2000-01-01", date_end="2021-12-31", device=device,
    )
    print(f"Forcing shape: {fc.shape}", flush=True)

    # Obs train (2001-2018) + val (2019-2021)
    con = duckdb.connect(DB, read_only=True)
    st = con.execute("select node_idx, station_id from stations order by node_idx").fetchdf()
    ob_all = con.execute(
        "select date, station_id, discharge as q from observations "
        "where date between '2001-01-01' and '2021-12-31'"
    ).fetchdf()
    con.close()
    sn = st["node_idx"].values.astype(int)
    s2c = {s: i for i, s in enumerate(st["station_id"].values)}
    d2t = {d: i for i, d in enumerate(dt)}
    qo_full = np.full((len(dt), len(st)), np.nan, dtype=np.float32)
    for _, r in ob_all.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in d2t and r["station_id"] in s2c:
            qo_full[d2t[d], s2c[r["station_id"]]] = float(r["q"])

    train_mask = (dt >= pd.Timestamp("2001-01-01")) & (dt <= pd.Timestamp("2018-12-31"))
    val_mask = (dt >= pd.Timestamp("2019-01-01")) & (dt <= pd.Timestamp("2021-12-31"))

    print("Forward backbone (1 pass)...", flush=True)
    with torch.no_grad():
        Q, _ = m.simulate(
            forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=withdrawals, day_of_year=doy,
        )
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
        sp_tensor = sp.to_tensor()

    sn_dev = torch.from_numpy(sn).long().to(device)
    Q_st = Q[:, sn_dev].cpu().numpy()                                   # (T_total, n_st)

    # β par station depuis train
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    qo_tr = qo_full[train_idx]
    qs_tr = Q_st[train_idx]
    beta_per_st = np.full(len(sn), np.nan, dtype=np.float32)
    for i in range(len(sn)):
        v = ~np.isnan(qo_tr[:, i]) & ~np.isnan(qs_tr[:, i])
        if v.sum() >= 30:
            beta_per_st[i] = qs_tr[v, i].mean() / max(qo_tr[v, i].mean(), 1e-6)
    finite = ~np.isnan(beta_per_st)
    print(f"\nβ_per_station depuis train (Q_sim/Q_obs) :")
    print(f"  n stations    : {finite.sum()} / {len(sn)}")
    print(f"  β min / max   : {beta_per_st[finite].min():.3f} / {beta_per_st[finite].max():.3f}")
    print(f"  β median      : {np.median(beta_per_st[finite]):.3f}")
    print(f"  β mean        : {beta_per_st[finite].mean():.3f}")
    print(f"  β std         : {beta_per_st[finite].std():.3f}")
    n_high = (beta_per_st[finite] > 1.1).sum()
    n_low = (beta_per_st[finite] < 0.9).sum()
    print(f"  β > 1.1 (modèle sur-prédit) : {n_high}/{finite.sum()}")
    print(f"  β < 0.9 (modèle sous-prédit) : {n_low}/{finite.sum()}")

    # Q_sim corrigé sur val
    qs_val = Q_st[val_idx]                                              # (T_val, n_st)
    qs_corr = qs_val / beta_per_st[None, :]                             # broadcast
    qo_val = qo_full[val_idx]
    valid = ~np.isnan(qo_val) & ~np.isnan(qs_val) & finite[None, :]

    # KGE pooled avant / après
    from meandre.utils.metrics import kge as _kge
    y = qo_val[valid]
    sp_ = qs_val[valid]
    sc_ = qs_corr[valid]
    kge_before = float(_kge(torch.from_numpy(y), torch.from_numpy(sp_)))
    kge_after = float(_kge(torch.from_numpy(y), torch.from_numpy(sc_)))
    print(f"\nKGE pooled :")
    print(f"  AVANT correction  : {kge_before:.4f}")
    print(f"  APRÈS correction  : {kge_after:.4f}")

    # PIT avec mixture head (si présent) OU avec Gauss σ apprise
    if m.use_mixture_head and hasattr(m, "mixture_head"):
        sp_st = sp_tensor[sn_dev]
        # PIT avant correction (Q_sim normal)
        with torch.no_grad():
            y_t = torch.from_numpy(y).to(device)
            sp_flat = sp_st[torch.from_numpy(np.where(valid)[1]).long().to(device)]
            q_flat = torch.from_numpy(sp_).to(device)
            q_corr_flat = torch.from_numpy(sc_).to(device)
            pit_before = m.mixture_head.cdf(y_t, sp_flat, q_flat).cpu().numpy()
            pit_after = m.mixture_head.cdf(y_t, sp_flat, q_corr_flat).cpu().numpy()
        label = "mixture"
    elif m.use_quantile_head and hasattr(m, "quantile_head"):
        # PIT via interpolation linéaire CDF des 6 quantiles
        from numpy import searchsorted
        taus = np.array(m.quantile_head.taus)
        K = len(taus)

        def quant_for(q_input, sp_st_dev, sn_idx):
            with torch.no_grad():
                q_t = torch.from_numpy(q_input).to(device)
                sp_t = sp_st_dev
                # Need broadcast : (T, n_st, F) + (T, n_st) → (T, n_st, K) offsets
                T_v = q_t.shape[0]
                sp_exp = sp_t.unsqueeze(0).expand(T_v, -1, -1)
                offs = m.quantile_head(sp_exp.reshape(-1, sp_t.shape[-1]),
                                       q_t.reshape(-1)).reshape(T_v, sp_t.shape[0], K)
                q_pred = q_t.unsqueeze(-1) + offs
            return q_pred.cpu().numpy()

        sp_st = sp_tensor[sn_dev]
        q_pred_before = quant_for(qs_val, sp_st, sn)                   # (T, n_st, K)
        q_pred_after = quant_for(qs_corr, sp_st, sn)
        # PIT par interpolation
        def pit_from_quants(qo_v, qpred_v):
            y_v = qo_v[valid]
            qp = qpred_v[valid]
            pit = np.zeros(len(y_v), dtype=np.float32)
            for i in range(len(y_v)):
                q_i = qp[i]
                yi = y_v[i]
                if yi <= q_i[0]:
                    pit[i] = taus[0] * yi / max(q_i[0], 1e-9)
                elif yi >= q_i[-1]:
                    pit[i] = min(1.0, taus[-1] + (1-taus[-1])*(yi-q_i[-1])/max(q_i[-1], 1e-9))
                else:
                    k = searchsorted(q_i, yi) - 1
                    k = max(0, min(K-2, k))
                    pit[i] = taus[k] + (taus[k+1]-taus[k])*(yi-q_i[k])/max(q_i[k+1]-q_i[k], 1e-9)
            return np.clip(pit, 0, 1)
        pit_before = pit_from_quants(qo_val, q_pred_before)
        pit_after = pit_from_quants(qo_val, q_pred_after)
        label = "quantile"
    else:
        print("Aucune tête probabiliste trouvée — abort"); return

    def stats(p, lbl):
        H, _ = np.histogram(p, bins=20, range=(0, 1))
        f = H / H.sum()
        d2 = float(((f - 1/20) ** 2).mean() / (1/20) ** 2)
        ks = float(kstest(p, "uniform").statistic)
        print(f"  [{lbl}] mean={p.mean():.3f} std={p.std():.3f} δ²={d2:.3f} KS={ks:.3f}  "
              f"frac IQR={((p>=0.25)&(p<=0.75)).mean():.3f}  "
              f"cov_90={((p>=0.05)&(p<=0.95)).mean():.3f}")
        return d2, ks

    print(f"\n=== PIT comparaison (tête {label}) ===")
    d2_b, ks_b = stats(pit_before, "AVANT correction β")
    d2_a, ks_a = stats(pit_after, "APRÈS correction β")
    print(f"\nΔ δ² : {d2_a - d2_b:+.4f}  (négatif = mieux)")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, p, ttl in [(axes[0], pit_before, f"AVANT — δ²={d2_b:.3f} KS={ks_b:.3f}"),
                       (axes[1], pit_after,  f"APRÈS — δ²={d2_a:.3f} KS={ks_a:.3f}")]:
        ax.hist(p, bins=20, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
        ax.axhline(len(p)/20, ls="--", c="k")
        ax.set_xlabel("PIT u"); ax.set_ylabel("Effectif")
        ax.set_title(ttl)
    fig.suptitle(f"Audit D : β-correction Q_sim — tête {label}")
    out_png = OUT / "audit_beta_correction.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nPNG : {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=".runs/slso/checkpoints/best-mdn.pt")
    args = p.parse_args()
    main(args.ckpt)
