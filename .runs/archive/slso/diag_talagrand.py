"""Talagrand non-paramétrique vs PIT paramétriques sur best-phase2-grace.pt.

4 PIT en miroir, calculées sur le set de validation (2019-2021) :

  1. NON-PARAMÉTRIQUE (Talagrand orthodoxe en m³/s)
     - Échantillonne M=200 tirages de N(T(μ), σ_BC) en espace Box-Cox(0.3)
     - Back-transforme en m³/s
     - Rang de l'obs parmi les M tirages (avec jitter pour les égalités)
     - Aucune hypothèse de forme côté vérification

  2. BC PARAMÉTRIQUE : PIT = Φ((T(y) − T(μ))/σ_BC). Ce que la loss optimise.

  3. LINÉAIRE-GAUSSIEN (delta) : PIT = Φ((y − μ)/σ_lin) avec σ_lin ≈ σ_BC·μ^(1−λ).
     Le panel GAUCHE de la figure d'Essi.

  4. LOG-NORMAL (delta) : PIT = Φ((log y − log μ)/σ_log) avec σ_log ≈ σ_BC/μ^λ.
     Le panel DROIT.
"""
import os, math, torch, numpy as np, pandas as pd, xarray as xr, duckdb
from pathlib import Path
import matplotlib.pyplot as plt
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.utils.noise_head import SpatialNoiseHead
from meandre.routing.withdrawals import WithdrawalData

dev = "cuda" if torch.cuda.is_available() else "cpu"
LAM = 0.3
M_SAMPLES = 200  # tirages pour le Talagrand non-paramétrique
CK = ".runs/slso/checkpoints/best-phase2-grace.pt"
DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)

# ── Charger modèle + données ───────────────────────────────────────────────
init = torch.load(CK, map_location="cpu", weights_only=False)["init_kwargs"]
model = HydroModel(**init).to(dev); model.load(CK); model.eval()
cache = BasinCache(DB); h = cache.load(device=torch.device(dev))
nc, terr = h["node_coords"], h["territorial"]
n_nodes = h["n_nodes"]
ds = xr.open_dataset(FORCING)
forcing = torch.from_numpy(ds["forcing"].values.astype(np.float32)).to(dev)
dates = pd.to_datetime(ds["time"].values).normalize()
ds.close()
doy = torch.tensor(dates.dayofyear.values, dtype=torch.long, device=dev)

# ── Obs aux stations sur la période de val ─────────────────────────────────
con = duckdb.connect(DB, read_only=True)
stations_df = con.execute("select node_idx, station_id from stations order by node_idx").fetchdf()
obs_df = con.execute(
    "select date, station_id, discharge as q_m3s from observations "
    "where date between '2019-01-01' and '2021-12-31' order by date, station_id"
).fetchdf()
con.close()
val_mask = (dates >= pd.Timestamp("2019-01-01")) & (dates <= pd.Timestamp("2021-12-31"))
val_idx = np.where(val_mask)[0]
date_to_vi = {d: i for i, d in enumerate(dates[val_idx])}

n_stations = len(stations_df)
station_node = stations_df["node_idx"].values.astype(int)
sid_to_col = {sid: i for i, sid in enumerate(stations_df["station_id"].values)}

q_obs_val = np.full((len(val_idx), n_stations), np.nan, dtype=np.float32)
for _, r_ in obs_df.iterrows():
    d = pd.Timestamp(r_["date"]).normalize()
    if d in date_to_vi and r_["station_id"] in sid_to_col:
        q_obs_val[date_to_vi[d], sid_to_col[r_["station_id"]]] = float(r_["q_m3s"])
print(f"obs valides val: {(~np.isnan(q_obs_val)).sum()} / {q_obs_val.size}")

# ── Forward + log σ aux stations ───────────────────────────────────────────
with torch.no_grad():
    Q_sim, _ = model.simulate(
        forcing=forcing, initial_state=HydroState.zeros(n_nodes, device=dev),
        graph=h["graph"], node_coords=nc, territorial=terr,
        withdrawals=WithdrawalData.zeros(forcing.shape[0], n_nodes, device=dev),
        day_of_year=doy,
    )
    sp = model.spatial_encoder(nc, terr.to_tensor())
    log_sigma_full = (
        model.noise_head(sp.to_tensor(), Q_sim.detach())
        if isinstance(model.noise_head, SpatialNoiseHead)
        else model.noise_head(Q_sim.detach())
    )

# Extraire μ, σ aux nœuds-stations sur la val
mu_val = Q_sim[val_mask][:, station_node].cpu().numpy()        # (T_val, n_stations)
sig_val = log_sigma_full[val_mask][:, station_node].exp().cpu().numpy()

# ── PIT 4 variantes ───────────────────────────────────────────────────────
def boxcox(x, lam=LAM, eps=1e-3):
    x = np.maximum(x, eps)
    return (x ** lam - 1.0) / lam
def boxcox_inv(z, lam=LAM):
    arg = 1.0 + lam * z
    return np.where(arg > 0, arg ** (1.0 / lam), 0.0)
def Phi(z): return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

# Aplatir et filtrer valides
v = ~np.isnan(q_obs_val) & (q_obs_val > 0) & (mu_val > 0)
y = q_obs_val[v]
mu = mu_val[v]
sig = sig_val[v]
print(f"n valides post-filtre: {len(y)}")

# 1) NON-PARAMÉTRIQUE : échantillons en BC → back-transform → rang
rng = np.random.default_rng(0)
z_samples = boxcox(mu, LAM)[:, None] + sig[:, None] * rng.standard_normal((len(y), M_SAMPLES))
q_samples = boxcox_inv(z_samples)  # (N, M) en m³/s
ranks = (q_samples < y[:, None]).sum(axis=1) + rng.uniform(0, 1, size=len(y))
pit_np = ranks / (M_SAMPLES + 1)

# 2) BC paramétrique
pit_bc = Phi((boxcox(y) - boxcox(mu)) / sig)

# 3) Linéaire-gaussien (delta) — σ_lin = σ_BC · μ^(1−λ)
sig_lin = sig * mu ** (1 - LAM)
pit_lin = Phi((y - mu) / sig_lin)

# 4) Log-normal delta — σ_log = σ_BC / μ^λ
sig_log = sig / (mu ** LAM)
pit_log = Phi((np.log(y) - np.log(mu)) / sig_log)

# ── Plot 4 panels ─────────────────────────────────────────────────────────
def delta2(pit, K=21):
    h, _ = np.histogram(pit, bins=K, range=(0, 1))
    f = h / h.sum()
    return float(((f - 1/K) ** 2).mean() / (1/K) ** 2)

fig, ax = plt.subplots(1, 4, figsize=(20, 4))
N = len(y); uniform_line = N / 21
titles = [
    f"Non-paramétrique m³/s — δ²={delta2(pit_np):.2f}",
    f"BC paramétrique (ce que la loss vise) — δ²={delta2(pit_bc):.2f}",
    f"Linéaire-gaussien (delta) — δ²={delta2(pit_lin):.2f}",
    f"Log-normal (delta) — δ²={delta2(pit_log):.2f}",
]
for a, pit, t in zip(ax, [pit_np, pit_bc, pit_lin, pit_log], titles):
    a.hist(pit, bins=21, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.8)
    a.axhline(uniform_line, ls="--", color="k", label=f"Uniforme ({uniform_line:.0f})")
    a.set_xlabel("PIT u"); a.set_title(t, fontsize=10)
    a.legend(fontsize=8)
ax[0].set_ylabel("Effectif")
plt.tight_layout()
out_png = OUT / "talagrand_diagnostic.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"saved -> {out_png}")

# Cov 50/90 sur le PIT non-paramétrique
for level in [50, 90]:
    a = (100 - level) / 200
    cov = ((pit_np >= a) & (pit_np <= 1 - a)).mean()
    print(f"cov_{level} (non-paramétrique m³/s) = {cov:.3f}")
