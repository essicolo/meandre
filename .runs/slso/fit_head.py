"""Fit a probabilistic head on cached backbone outputs.

Itération rapide : pas de forward backbone, pas de simulate(). Pure tabulaire.
Chaque epoch ≈ 10-100 ms sur GPU pour 250k samples.

Heads supportés :
  - mixture-gauss  : K-Gaussian mixture (par défaut K=10)
  - quantile       : K quantile pinball loss (médiane=Q_sim+offsets)
  - gauss-hetero   : Gaussienne hétéroscédastique (μ=Q_sim+δ, σ=feature)
  - student-t      : Student-t hétéroscédastique avec ν appris

Features par défaut : spatial_params + Q_sim + log(Q_sim+1) + sin/cos(doy)
Options : --extra-features none | doy | doy+context

Usage :
  python fit_head.py --cache cache_backbone.npz --head mixture-gauss --K 10 --epochs 100
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
SQRT_2 = math.sqrt(2.0)
SQRT_2_PI = math.sqrt(2 * math.pi)


# ─── Heads (tabulaires, vectorisés) ──────────────────────────────────────────

class MixtureGaussHead(nn.Module):
    """K-Gaussian mixture conditional on features (M, F)."""
    def __init__(self, F_in: int, K: int = 10, hidden: int = 64) -> None:
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(F_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3 * K),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            b = self.net[-1].bias
            b[:K].zero_(); b[K:2*K].zero_(); b[2*K:].fill_(math.log(3.0))

    def _params(self, x: torch.Tensor, Q: torch.Tensor):
        out = self.net(x)
        log_pi = F.log_softmax(out[..., :self.K], dim=-1)
        mu = Q.unsqueeze(-1) + out[..., self.K:2*self.K]
        log_sigma = out[..., 2*self.K:].clamp(-6, 8)
        return log_pi, mu, log_sigma

    def nll(self, y: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        log_pi, mu, log_sigma = self._params(x, Q)
        sigma = log_sigma.exp()
        z = (y.unsqueeze(-1) - mu) / sigma
        log_normal = -0.5 * z * z - log_sigma - LOG_SQRT_2PI
        return -torch.logsumexp(log_pi + log_normal, dim=-1).mean()

    def cdf(self, y: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        log_pi, mu, log_sigma = self._params(x, Q)
        sigma = log_sigma.exp()
        z = (y.unsqueeze(-1) - mu) / sigma
        Phi_k = 0.5 * (1.0 + torch.erf(z / SQRT_2))
        return (log_pi.exp() * Phi_k).sum(dim=-1).clamp(0, 1)


class QuantileHead(nn.Module):
    """K quantiles, monotones via cumsum exp, médiane = Q_sim + δ_0."""
    def __init__(self, F_in: int, taus: tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
                 hidden: int = 64) -> None:
        super().__init__()
        self.taus = taus
        self.K = len(taus)
        # On apprend K log-largeurs (≥0 via exp) puis intègre
        # Output : 2K values - centers (K) + widths (K)
        self.net = nn.Sequential(
            nn.Linear(F_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * self.K),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
        # médiane index dans taus
        self.median_idx = taus.index(0.50) if 0.50 in taus else self.K // 2

    def _quantiles(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        log_w = out[..., :self.K]                      # log-largeurs
        center = out[..., self.K:].mean(dim=-1, keepdim=True)  # offset central
        widths = log_w.exp()
        # Cumsum signé depuis le centre : q_τ croissant en τ
        # On centre les largeurs sur la médiane
        # Construction : q_0 = -sum_widths_left, q_K = +sum_widths_right, médiane = 0 offset
        # Simple : cumsum(widths) puis recentre sur médiane
        cum = torch.cumsum(widths, dim=-1)
        median_value = cum[..., self.median_idx:self.median_idx+1]
        q_centered = cum - median_value                 # médiane = 0
        return Q.unsqueeze(-1) + center + q_centered

    def pinball(self, y: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        q_pred = self._quantiles(x, Q)
        resid = y.unsqueeze(-1) - q_pred
        taus_t = torch.tensor(self.taus, device=q_pred.device, dtype=q_pred.dtype)
        return torch.maximum(taus_t * resid, (taus_t - 1) * resid).mean()

    def cdf_interp(self, y: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """CDF par interpolation linéaire des K quantile points."""
        q = self._quantiles(x, Q)                       # (M, K)
        taus_t = torch.tensor(self.taus, device=q.device, dtype=q.dtype)
        # Trouver k tel que q[m, k] <= y[m] < q[m, k+1]
        below = y.unsqueeze(-1) < q                     # (M, K)
        idx = below.float().argmax(dim=-1)              # premier True, ou 0 si pas
        # Cas hors-intervalle
        y_c = y.clamp(min=1e-9)
        out = torch.zeros_like(y)
        for k in range(self.K - 1):
            mask = (y >= q[:, k]) & (y < q[:, k+1])
            w = (q[:, k+1] - q[:, k]).clamp(min=1e-9)
            out = torch.where(mask, taus_t[k] + (taus_t[k+1] - taus_t[k]) * (y - q[:, k]) / w, out)
        # Au-delà q_max → vers 1
        out = torch.where(y >= q[:, -1],
                          (taus_t[-1] + (1 - taus_t[-1]) * (y - q[:, -1]).clamp(min=0) /
                           q[:, -1].clamp(min=1e-9)).clamp(max=1.0), out)
        # En-deçà q_min → vers 0
        out = torch.where(y <= q[:, 0],
                          (taus_t[0] * (y / q[:, 0].clamp(min=1e-9))).clamp(min=0.0, max=taus_t[0]), out)
        return out.clamp(0, 1)


class GaussHeteroHead(nn.Module):
    """Gaussien hétéroscédastique : μ = Q_sim + δ(x), σ = exp(s(x))."""
    def __init__(self, F_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([0.0, math.log(3.0)]))

    def _params(self, x, Q):
        out = self.net(x)
        mu = Q + out[..., 0]
        log_sigma = out[..., 1].clamp(-6, 8)
        return mu, log_sigma

    def nll(self, y, x, Q):
        mu, log_sigma = self._params(x, Q)
        sigma = log_sigma.exp()
        z = (y - mu) / sigma
        return (0.5 * z * z + log_sigma + LOG_SQRT_2PI).mean()

    def cdf(self, y, x, Q):
        mu, log_sigma = self._params(x, Q)
        sigma = log_sigma.exp()
        z = (y - mu) / sigma
        return 0.5 * (1 + torch.erf(z / SQRT_2))


# ─── Pipeline ────────────────────────────────────────────────────────────────

def build_features(cache: dict, mode: str = "doy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit la matrice features (T, n_st, F) en NumPy depuis le cache.

    Retourne :
      features : (T, n_st, F)
      Q_sim    : (T, n_st)
      q_obs    : (T, n_st)
    """
    Q = cache["Q_sim"]                                  # (T, n_st)
    sp = cache["spatial_params"]                        # (n_st, F_sp)
    qo = cache["q_obs"]
    doy = cache["day_of_year"]
    T, n_st = Q.shape
    F_sp = sp.shape[1]

    # spatial_params répétés sur T
    feat_sp = np.broadcast_to(sp[None, :, :], (T, n_st, F_sp))
    # Q_sim + log Q_sim
    feat_Q = Q[..., None]
    feat_logQ = np.log(np.maximum(Q, 0) + 1.0)[..., None]
    parts = [feat_sp, feat_Q, feat_logQ]
    if mode in ("doy", "doy+gru", "indices", "indices+gru"):
        doy_rad = 2 * np.pi * doy / 366.0
        feat_doy = np.stack([np.sin(doy_rad), np.cos(doy_rad)], axis=-1)  # (T, 2)
        feat_doy = np.broadcast_to(feat_doy[:, None, :], (T, n_st, 2))
        parts.append(feat_doy)
    if mode in ("doy+gru", "indices+gru"):
        if "gru_context" in cache:
            parts.append(cache["gru_context"])                          # (T, n_st, 16)
        else:
            print("[warn] gru demandé mais cache n'a pas 'gru_context' — skip")
    if mode in ("indices", "indices+gru"):
        # Indices IHI (GDD, API, SPI, frost number, SWE proxy)
        for k in ["gdd_cum", "api_30", "spi_30", "frost_number_90", "swe_proxy"]:
            key = f"idx_{k}"
            if key in cache:
                v = cache[key]                                          # (T, n_st) — par station
                # Normalisation z-score grossière pour stabilité
                v_mean = v.mean(); v_std = v.std() + 1e-6
                v_norm = ((v - v_mean) / v_std).astype(np.float32)
                parts.append(v_norm[..., None])                         # (T, n_st, 1)
            else:
                print(f"[warn] cache manque l'indice {key}")
    features = np.concatenate(parts, axis=-1).astype(np.float32)
    return features, Q.astype(np.float32), qo.astype(np.float32)


def pit_metrics(pit: np.ndarray, pit_matrix: np.ndarray | None = None) -> dict:
    """Métriques de calibration du PIT.

    pit_matrix optionnel : matrice (T, S) avec NaN pour les manquants, pour la
    normalisation Candille-Talagrand corrigée de la dépendance temporelle et
    inter-stations (bootstrap par blocs, cf meandre.diagnostics.talagrand).
    Sans elle, seul delta_ct_iid est calculé (cible 1 si parfaitement fiable).
    """
    pit = np.clip(pit, 0, 1)
    H, _ = np.histogram(pit, bins=20, range=(0, 1))
    N, B = int(H.sum()), 20
    f = H / N
    d2 = float(((f - 1/B) ** 2).mean() / (1/B) ** 2)
    delta_ct_iid = float(((H - N/B) ** 2).sum() / (N * (B - 1) / B))
    ks_stat = float(kstest(pit, "uniform").statistic)
    out = dict(
        n=int(len(pit)),
        mean=float(pit.mean()),
        std=float(pit.std()),
        d2=d2,
        delta_ct_iid=delta_ct_iid,
        ks=ks_stat,
        frac_lt05=float((pit < 0.05).mean()),
        frac_gt95=float((pit > 0.95).mean()),
        frac_iqr=float(((pit >= 0.25) & (pit <= 0.75)).mean()),
        cov_50=float(((pit >= 0.25) & (pit <= 0.75)).mean()),
        cov_90=float(((pit >= 0.05) & (pit <= 0.95)).mean()),
    )
    if pit_matrix is not None:
        from meandre.diagnostics.talagrand import candille_talagrand
        ct = candille_talagrand(pit_matrix, n_bins=B)
        out.update(
            delta_ct_eff=ct["delta_eff"],
            ct_tau=ct["tau"],
            ct_p_value=ct["p_value"],
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=".runs/slso/data/cache_backbone.npz")
    p.add_argument("--head", choices=["mixture-gauss", "quantile", "gauss-hetero"],
                   default="mixture-gauss")
    p.add_argument("--K", type=int, default=10, help="composantes (mixture) ou quantiles")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--features",
                   choices=["base", "doy", "doy+gru", "indices", "indices+gru"],
                   default="doy",
                   help="base = sp+Q+logQ ; doy = +sin/cos(doy) ; doy+gru = +contexte GRU 16D ; "
                        "indices = +GDD+API+SPI+FN+SWE proxy ; indices+gru = tout")
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = full batch (recommandé)")
    p.add_argument("--out", default=None, help="prefix for output files")
    args = p.parse_args()

    out_prefix = args.out or f"head_{args.head}_K{args.K}_{args.features}"
    print(f"out_prefix = {out_prefix}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load cache
    print(f"Loading cache {args.cache}...", flush=True)
    cache = dict(np.load(args.cache, allow_pickle=True))
    print(f"  Q_sim : {cache['Q_sim'].shape}, train_mask : {cache['train_mask'].sum()} days, "
          f"val_mask : {cache['val_mask'].sum()}", flush=True)

    features, Q_arr, qo_arr = build_features(cache, mode=args.features)
    train_m = cache["train_mask"]; val_m = cache["val_mask"]

    # Flatten valid (t, s) couples
    def flatten(mask):
        m = mask[:, None] & ~np.isnan(qo_arr) & ~np.isnan(Q_arr)        # (T, n_st)
        t_idx, s_idx = np.where(m)
        return (
            torch.from_numpy(features[t_idx, s_idx]).to(device),
            torch.from_numpy(Q_arr[t_idx, s_idx]).to(device),
            torch.from_numpy(qo_arr[t_idx, s_idx]).to(device),
        )

    x_tr, Q_tr, y_tr = flatten(train_m)
    x_v, Q_v, y_v = flatten(val_m)
    print(f"Train samples : {len(y_tr)}, val : {len(y_v)}, F = {x_tr.shape[1]}", flush=True)

    # Head
    F_in = x_tr.shape[1]
    if args.head == "mixture-gauss":
        head = MixtureGaussHead(F_in, K=args.K, hidden=args.hidden).to(device)
        loss_fn = lambda y, x, Q: head.nll(y, x, Q)
        cdf_fn = lambda y, x, Q: head.cdf(y, x, Q)
    elif args.head == "quantile":
        # Classe canonique de meandre/utils — utilisée aussi dans le trainer.
        from meandre.utils.contextual_quantile_head import ContextualQuantileHead
        taus_default = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        head = ContextualQuantileHead(F_in, taus=taus_default, hidden=args.hidden).to(device)
        loss_fn = lambda y, x, Q: head.pinball(y, x, Q)
        cdf_fn = lambda y, x, Q: head.cdf_interp(y, x, Q)
    elif args.head == "gauss-hetero":
        head = GaussHeteroHead(F_in, hidden=args.hidden).to(device)
        loss_fn = lambda y, x, Q: head.nll(y, x, Q)
        cdf_fn = lambda y, x, Q: head.cdf(y, x, Q)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head : {args.head}, params = {n_params}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train loop (full batch, vectorisé)
    t_start = time.perf_counter()
    losses = []; val_d2s = []
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = loss_fn(y_tr, x_tr, Q_tr)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(y_v, x_v, Q_v))
                pit = cdf_fn(y_v, x_v, Q_v).cpu().numpy()
                m = pit_metrics(pit)
            val_d2s.append((ep, m["d2"]))
            elapsed = time.perf_counter() - t_start
            print(f"ep {ep:4d}  train={loss.item():.4f}  val_loss={val_loss:.4f}  "
                  f"val_d2={m['d2']:.4f}  cov90={m['cov_90']:.3f}  iqr={m['frac_iqr']:.3f}  "
                  f"({elapsed:.1f}s elapsed)", flush=True)

    # Final eval
    head.eval()
    with torch.no_grad():
        pit_v = cdf_fn(y_v, x_v, Q_v).cpu().numpy()
        m_final = pit_metrics(pit_v)

    print(f"\n=== Final val metrics ({args.head} K={args.K} {args.features}) ===")
    for k, v in m_final.items():
        print(f"  {k:12s} : {v}")

    # Save
    torch.save({
        "head_state_dict": head.state_dict(),
        "head_class": args.head,
        "K": args.K, "hidden": args.hidden,
        "features_mode": args.features,
        "metrics": m_final,
    }, OUT.parent / "slso" / "checkpoints" / f"{out_prefix}.pt"
       if False else OUT / f"{out_prefix}_head.pt")

    # Plot PIT
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit_v, bins=20, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
    ax.axhline(len(pit_v)/20, ls="--", c="k", label=f"Uniforme ({len(pit_v)/20:.0f})")
    ax.set_xlabel("PIT u"); ax.set_ylabel("Effectif")
    ax.set_title(f"{args.head} K={args.K} feats={args.features} — d²={m_final['d2']:.3f} "
                 f"cov90={m_final['cov_90']:.3f}")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT / f"{out_prefix}_pit.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nPNG : {OUT / f'{out_prefix}_pit.png'}", flush=True)


if __name__ == "__main__":
    main()
