"""Évaluation held-out (test 2022-2024) de best-phase2-grace-quantile.pt.

Sorties :
  - KGE déterministe (sur la médiane = μ) → doit valoir ~0.81 (backbone gelé)
  - cov_50, cov_90 par appartenance inter-quantiles en m³/s
  - Pinball moyenne (NLL équivalent)
  - CRPS approximé (Gneiting-Ranjan)
  - Talagrand non-paramétrique m³/s (PNG)
"""
import os, math, torch, numpy as np, pandas as pd, xarray as xr, duckdb
from pathlib import Path
import matplotlib.pyplot as plt
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.routing.withdrawals import WithdrawalData
from meandre.utils.metrics import kge as _kge

dev = "cuda" if torch.cuda.is_available() else "cpu"
M_SAMPLES = 200  # tirages pour Talagrand non-paramétrique
CK = ".runs/slso/checkpoints/best-phase2-grace-quantile.pt"
DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
TEST_START, TEST_END = "2022-01-01", "2024-12-31"
OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)

# Charger modèle. L'ancien checkpoint n'a pas use_quantile_head dans init_kwargs
# (fix apporté après) → on force ici.
init = torch.load(CK, map_location="cpu", weights_only=False)["init_kwargs"]
init["use_quantile_head"] = True
init["quantile_taus"] = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
model = HydroModel(**init).to(dev); model.load(CK); model.eval()
print("quantile_head taus:", model.quantile_head.taus)

cache = BasinCache(DB); h = cache.load(device=torch.device(dev))
nc, terr = h["node_coords"], h["territorial"]
n_nodes = h["n_nodes"]

ds = xr.open_dataset(FORCING)
forcing = torch.from_numpy(ds["forcing"].values.astype(np.float32)).to(dev)
dates = pd.to_datetime(ds["time"].values).normalize()
ds.close()
doy = torch.tensor(dates.dayofyear.values, dtype=torch.long, device=dev)

# Obs test 2022-2024
con = duckdb.connect(DB, read_only=True)
stations_df = con.execute("select node_idx, station_id from stations order by node_idx").fetchdf()
obs_df = con.execute(
    f"select date, station_id, discharge as q from observations "
    f"where date between '{TEST_START}' and '{TEST_END}' order by date, station_id"
).fetchdf()
con.close()

test_mask = (dates >= pd.Timestamp(TEST_START)) & (dates <= pd.Timestamp(TEST_END))
test_idx = np.where(test_mask)[0]
d2t = {d: i for i, d in enumerate(dates[test_idx])}
sn = stations_df["node_idx"].values.astype(int)
s2c = {s: i for i, s in enumerate(stations_df["station_id"].values)}

q_obs = np.full((len(test_idx), len(stations_df)), np.nan, dtype=np.float32)
for _, r in obs_df.iterrows():
    d = pd.Timestamp(r["date"]).normalize()
    if d in d2t and r["station_id"] in s2c:
        q_obs[d2t[d], s2c[r["station_id"]]] = float(r["q"])
print(f"obs valides test: {(~np.isnan(q_obs)).sum()} / {q_obs.size}")

# Forward complet + quantile_head
with torch.no_grad():
    Q_sim_full, _ = model.simulate(
        forcing=forcing, initial_state=HydroState.zeros(n_nodes, device=dev),
        graph=h["graph"], node_coords=nc, territorial=terr,
        withdrawals=WithdrawalData.zeros(forcing.shape[0], n_nodes, device=dev),
        day_of_year=doy,
    )
    sp = model.spatial_encoder(nc, terr.to_tensor())
    # quantile_head : offsets (T, N, K)
    offsets = model.quantile_head(sp.to_tensor(), Q_sim_full.detach())
    q_pred_full = Q_sim_full.detach().unsqueeze(-1) + offsets  # (T, N, K)

# Slice test, stations
mu_test = Q_sim_full[test_mask][:, sn].cpu().numpy()                 # (T_test, n_st)
q_pred_test = q_pred_full[test_mask][:, sn, :].cpu().numpy()         # (T_test, n_st, K)
taus = list(model.quantile_head.taus)
K = len(taus)

# ── KGE déterministe sur la médiane (= μ) ─────────────────────────────────
v_all = ~np.isnan(q_obs) & ~np.isnan(mu_test)
yo = torch.from_numpy(q_obs[v_all]); ym = torch.from_numpy(mu_test[v_all])
kge_pooled = float(_kge(yo, ym))
print(f"\n=== Test held-out 2022-2024 ===")
print(f"KGE déterministe (μ = médiane) pooled : {kge_pooled:.4f}")

# Per-station KGE
kges = []
for s in range(len(stations_df)):
    v = ~np.isnan(q_obs[:, s]) & ~np.isnan(mu_test[:, s])
    if v.sum() >= 30:
        kges.append(float(_kge(torch.from_numpy(q_obs[v, s]), torch.from_numpy(mu_test[v, s]))))
kges = np.array(kges)
print(f"KGE per-station : median {np.median(kges):.4f}  mean {kges.mean():.4f}  "
      f"({(kges > 0.5).sum()}/{len(kges)} > 0.5, {(kges < 0).sum()}/{len(kges)} < 0)")

# ── Couvertures inter-quantiles ───────────────────────────────────────────
v = v_all
y = q_obs[v]
q_v = q_pred_test[v]              # (M, K)
print(f"\nCouvertures (n={v.sum()} obs valides) :")
for level, lo_tau, hi_tau in [(50, 0.25, 0.75), (90, 0.05, 0.95)]:
    if lo_tau in taus and hi_tau in taus:
        i_lo = taus.index(lo_tau); i_hi = taus.index(hi_tau)
        cov = ((y >= q_v[:, i_lo]) & (y <= q_v[:, i_hi])).mean()
        print(f"  cov_{level} (q_{lo_tau} ≤ obs ≤ q_{hi_tau}) = {cov:.3f}  (cible {level/100:.2f})")

# ── Pinball + CRPS ────────────────────────────────────────────────────────
taus_arr = np.array(taus)
resid = y[:, None] - q_v                                              # (M, K)
pinball_per_tau = np.maximum(taus_arr * resid, (taus_arr - 1) * resid).mean(axis=0)
pinball_mean = pinball_per_tau.mean()
crps = 2.0 * pinball_mean
print(f"\nPinball par τ : {dict(zip(taus, np.round(pinball_per_tau, 2)))}")
print(f"Pinball moyenne : {pinball_mean:.3f}")
print(f"CRPS (Gneiting-Ranjan 2·pinball moy) : {crps:.3f} m³/s")

# ── Talagrand non-paramétrique m³/s (Sample-based) ────────────────────────
# Échantillonne en supposant Gaussien autour de μ avec σ ≈ écart inter-quantile
# (proxy). Mieux : interpolation linéaire entre quantiles puis tirages — c'est
# ce qu'on fait ici.
rng = np.random.default_rng(0)
# Quantiles + bornes extrapolées : on ajoute τ=0 et τ=1 (extrapolation linéaire)
def quantile_at_u(u, quants, taus_a):
    """Interpolation linéaire des quantiles : pour u ∈ (0,1), retourne F^{-1}(u)."""
    # quants: (M, K), taus_a: (K,), u: (M, M_samples)
    # Extrémités : extrapolation linéaire vers τ=0 (= 2·q_τ_min − q_τ_min2) si nécessaire
    return np.interp(u.ravel(), taus_a, quants.ravel()).reshape(u.shape) if False else None

# Méthode plus simple : juste rang de l'obs parmi les K quantiles + jitter
# = Talagrand orthodoxe sur l'ensemble {q_τ}_τ + obs
ranks = (q_v < y[:, None]).sum(axis=1)                                # 0..K
# Jitter dans [0,1) pour casser les égalités → PIT = (rank + jitter) / (K+1)
pit = (ranks + rng.uniform(0, 1, size=len(y))) / (K + 1)
def delta2(p, B=21):
    h, _ = np.histogram(p, bins=B, range=(0, 1)); f = h / h.sum()
    return float(((f - 1/B) ** 2).mean() / (1/B) ** 2)
d2 = delta2(pit)
print(f"\nTalagrand non-paramétrique (ensemble K+1 = {K+1} points) : δ²={d2:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(pit, bins=21, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.85)
ax.axhline(len(pit)/21, ls="--", color="k", label=f"Uniforme ({len(pit)/21:.0f})")
ax.set_xlabel("PIT u"); ax.set_ylabel("Effectif")
ax.set_title(f"Talagrand test 2022-2024 (quantile, K+1={K+1}) — δ²={d2:.3f}")
ax.legend()
out_png = OUT / "talagrand_test_quantile.png"
plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\nPNG saved: {out_png}")
