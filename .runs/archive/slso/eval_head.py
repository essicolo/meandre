"""Eval a trained head on a held-out period (test 2022-2024 par défaut).

Charge cache_backbone + head sauvegardé par fit_head.py, applique sur le
masque demandé (train/val/test), reporte δ², KS, cov, CRPS.

Usage :
  python eval_head.py --head .reports/slso/batch_quantile_K7_doy_gru_head.pt \
                      --cache .runs/slso/data/cache_backbone.npz \
                      --period test
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

# Import head classes from fit_head
sys.path.insert(0, str(Path(__file__).parent))
from fit_head import MixtureGaussHead, QuantileHead, GaussHeteroHead, build_features, pit_metrics

OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True, help="head .pt sauvegardé par fit_head.py")
    p.add_argument("--cache", default=".runs/slso/data/cache_backbone.npz")
    p.add_argument("--period", choices=["train", "val", "test"], default="test")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Charger la tête
    ckpt = torch.load(args.head, map_location=device, weights_only=False)
    head_class = ckpt["head_class"]
    K = ckpt["K"]
    hidden = ckpt["hidden"]
    feat_mode = ckpt["features_mode"]
    print(f"Head : {head_class} K={K} hidden={hidden} features={feat_mode}", flush=True)

    cache = dict(np.load(args.cache, allow_pickle=True))
    features, Q_arr, qo_arr = build_features(cache, mode=feat_mode)
    period_mask = cache[f"{args.period}_mask"]
    n_valid_train = (~np.isnan(qo_arr[cache["train_mask"]])).sum()
    n_valid_period = (~np.isnan(qo_arr[period_mask])).sum()
    print(f"Period '{args.period}' : {period_mask.sum()} days × {qo_arr.shape[1]} stations, "
          f"valid obs = {n_valid_period}", flush=True)

    # Filtrer valides
    m = period_mask[:, None] & ~np.isnan(qo_arr) & ~np.isnan(Q_arr)
    t_idx, s_idx = np.where(m)
    x = torch.from_numpy(features[t_idx, s_idx]).to(device)
    Q = torch.from_numpy(Q_arr[t_idx, s_idx]).to(device)
    y = torch.from_numpy(qo_arr[t_idx, s_idx]).to(device)

    F_in = x.shape[1]

    # Build head and load weights
    if head_class == "mixture-gauss":
        head = MixtureGaussHead(F_in, K=K, hidden=hidden).to(device)
        cdf_fn = lambda y, x, Q: head.cdf(y, x, Q)
    elif head_class == "quantile":
        taus_default = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        head = QuantileHead(F_in, taus=taus_default, hidden=hidden).to(device)
        cdf_fn = lambda y, x, Q: head.cdf_interp(y, x, Q)
    elif head_class == "gauss-hetero":
        head = GaussHeteroHead(F_in, hidden=hidden).to(device)
        cdf_fn = lambda y, x, Q: head.cdf(y, x, Q)
    else:
        raise ValueError(f"unknown head_class {head_class}")

    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    with torch.no_grad():
        pit = cdf_fn(y, x, Q).cpu().numpy()
    metrics = pit_metrics(pit)

    print(f"\n=== Eval {head_class} K={K} {feat_mode} on '{args.period}' ===")
    for k, v in metrics.items():
        print(f"  {k:12s} : {v}")

    # KGE déterministe sur la médiane prédictive
    from meandre.utils.metrics import kge as _kge
    if head_class == "quantile":
        with torch.no_grad():
            q = head._quantiles(x, Q)
            median_idx = head.median_idx
            median = q[:, median_idx]
    elif head_class == "gauss-hetero":
        with torch.no_grad():
            mu, _ = head._params(x, Q)
            median = mu
    else:  # mixture-gauss : pas de médiane closed-form, on prend la moyenne pondérée
        with torch.no_grad():
            log_pi, mu, _ = head._params(x, Q)
            median = (log_pi.exp() * mu).sum(dim=-1)
    kge_median = float(_kge(y.cpu(), median.cpu()))
    print(f"  KGE pooled (médiane prédictive) : {kge_median:.4f}")

    # Plot
    out_prefix = args.out_prefix or f"eval_{Path(args.head).stem}_{args.period}"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit, bins=20, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
    ax.axhline(len(pit)/20, ls="--", c="k", label=f"Uniforme ({len(pit)/20:.0f})")
    ax.set_xlabel("PIT u")
    ax.set_ylabel("Effectif")
    ax.set_title(f"{head_class} K={K} {feat_mode} — {args.period} — "
                 f"d²={metrics['d2']:.4f}  KS={metrics['ks']:.3f}  cov90={metrics['cov_90']:.3f}")
    ax.legend()
    out_png = OUT / f"{out_prefix}_pit.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nPNG : {out_png}", flush=True)


if __name__ == "__main__":
    main()
