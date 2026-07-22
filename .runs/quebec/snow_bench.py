"""Banc FONTE hors-ligne (phase 2, design_modules_appris.md) : trois fonctions de
fonte dans la MÊME intégration SWE différentiable (SWE' = SWE + neige - fonte,
partition pluie/neige sigmoïde à seuil apprenable), supervisées par la présence
de neige MOD10A1 (mars-juin) :
  dd  : degré-jour calé (C_f, T_melt)                — l'existant, dans son meilleur jour
  eti : température-radiation calée (tf, srf)        — l'alternative physique
  mlp : fonte apprise MLP(météo, territoire, saison) — le candidat
Métriques : MAE date de disparition (jours) + précision de présence. Splits comme
et_bench : leave-region-out (gasp/sagu/mont) × temporel (test 2022-2024).

  python .runs/quebec/snow_bench.py                # banc complet (15 régions)
  SNOW_REGIONS="slso gasp" python .runs/quebec/snow_bench.py   # sous-ensemble (dev)
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
import duckdb
from meandre.data.basin_cache import BasinCache

REGIONS = os.environ.get("SNOW_REGIONS", "abit cnda cndb cndc cndd cnde gasp labi mont outm outv sagu slno slso vaud").split()
SPATIAL_HELD = [r for r in ["gasp", "sagu", "mont"] if r in REGIONS]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"
TRAIN_END, VAL_END = "2018-12-31", "2021-12-31"
NODE_CAP = 400            # séquences node×année : 400×24 ≈ 10k par région, suffisant
SEQ_START_MMDD, SEQ_DAYS = "-09-01", 303   # 1er sept -> ~30 juin
BATCH = 512
STEPS = int(os.environ.get("SNOW_STEPS", "400"))
DBS = {"slso": ".runs/slso/data/slso.duckdb"}
FORCINGS = {"slso": "D:/meandre-data/slso/forcing-casr-corr.nc"}
OUT_CSV = "reports/snow_bench_results.csv"
CKPT = "D:/meandre-data/quebec/checkpoints-etbench"
rng = np.random.default_rng(0)
torch.manual_seed(0)


def load_region(reg):
    db = DBS.get(reg, f"D:/meandre-data/quebec/{reg}.duckdb")
    fx = FORCINGS.get(reg, f"D:/meandre-data/quebec/forcing-{reg}-budyko.nc")
    if not os.path.exists(fx):
        fx = f"D:/meandre-data/quebec/forcing-{reg}.nc"
    cache = BasinCache(db)
    h = cache.load(device="cpu")
    coords = h["node_coords"].numpy()
    lat_col = 0 if 40 < np.nanmean(coords[:, 0]) < 62 else 1
    lat = coords[:, lat_col].astype(np.float32)
    static = h["territorial"].data.numpy().astype(np.float32)
    d = xr.open_dataset(fx)
    meteo = d["forcing"].values[:, :, :6].astype(np.float32)
    times = pd.to_datetime(d["time"].values); d.close()

    con = duckdb.connect(db, read_only=True)
    sf = con.execute("select date, node_idx, snow_frac from modis_snow where snow_frac is not null").fetchdf()
    con.close()
    sf["date"] = pd.to_datetime(sf["date"])
    tidx = {d: i for i, d in enumerate(times)}
    T = len(times)
    n_nodes = meteo.shape[1]
    snow = np.full((T, n_nodes), np.nan, dtype=np.float32)
    ti = sf["date"].map(tidx).values
    snow[ti.astype(int), sf["node_idx"].values.astype(int)] = sf["snow_frac"].values

    n_valid = (~np.isnan(snow)).sum(axis=0)
    keep = np.flatnonzero(n_valid >= 500)
    if len(keep) > NODE_CAP:
        keep = rng.choice(keep, NODE_CAP, replace=False); keep.sort()
    # séquences (node, année) : 1er sept année y-1 -> +303 j ; obs = fenêtre mars-juin année y
    seqs = []
    for y in range(2001, 2025):
        t0 = tidx.get(pd.Timestamp(f"{y-1}{SEQ_START_MMDD}"))
        if t0 is None or t0 + SEQ_DAYS > T: continue
        split = 0 if f"{y}-06-30" <= TRAIN_END else (1 if f"{y}-06-30" <= VAL_END else 2)
        for n in keep:
            seqs.append((t0, int(n), y, split))
    seqs = np.array(seqs, dtype=np.int32)
    print(f"[{reg}] {len(keep)} nœuds | {len(seqs):,} séquences node×année", flush=True)
    return dict(name=reg, meteo=meteo, snow=snow, static=static, lat=lat, times=times, seqs=seqs)


regions = [load_region(r) for r in REGIONS]
F_STATIC = regions[0]["static"].shape[1]
train_regs = [r for r in regions if r["name"] not in SPATIAL_HELD]

# normalisation météo (réutilise le train des régions d'entraînement)
_s = []
for r in train_regs:
    idx = rng.choice(len(r["seqs"]), min(50, len(r["seqs"])), replace=False)
    for t0, n, y, sp in r["seqs"][idx]:
        _s.append(r["meteo"][t0:t0 + SEQ_DAYS, n])
_s = torch.tensor(np.stack(_s)).reshape(-1, 6)
M_MEAN, M_STD = _s.mean(0), _s.std(0) + 1e-6
del _s


def gather(r, idx):
    """(B, 303, 6) météo brute ; (B, 303) présence obs (NaN hors obs) ; (B, F+3) statiques."""
    t0, n = r["seqs"][idx, 0], r["seqs"][idx, 1]
    w = np.stack([r["meteo"][a:a + SEQ_DAYS, b] for a, b in zip(t0, n)])
    o = np.stack([r["snow"][a:a + SEQ_DAYS, b] for a, b in zip(t0, n)])
    doy0 = np.array([r["times"][a].dayofyear for a in t0], dtype=np.float32)
    stat = np.concatenate([r["static"][n], (r["lat"][n] / 50.0)[:, None]], axis=1)
    return (torch.tensor(w), torch.tensor(o), torch.tensor(stat), torch.tensor(doy0))


class MeltFn(nn.Module):
    """kind: dd (C_f, T_melt) | eti (tf, srf) | mlp (météo+statiques+saison)."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.t_snow = nn.Parameter(torch.tensor(0.5))   # seuil pluie/neige (partagé, apprenable)
        if kind == "dd":
            self.cf = nn.Parameter(torch.tensor(4.5)); self.tm = nn.Parameter(torch.tensor(0.0))
        elif kind == "eti":
            self.tf = nn.Parameter(torch.tensor(1.2)); self.srf = nn.Parameter(torch.tensor(0.008))
        else:
            self.net = nn.Sequential(nn.Linear(6 + F_STATIC + 1 + 2, 48), nn.ReLU(), nn.Linear(48, 1), nn.Softplus())

    def melt(self, met_raw, met_norm, stat, doy):
        tmean = 0.5 * (met_raw[:, 1] + met_raw[:, 2])
        if self.kind == "dd":
            return torch.relu(self.cf) * torch.relu(tmean - self.tm)
        if self.kind == "eti":
            return torch.relu(self.tf * torch.relu(tmean) + torch.relu(self.srf) * torch.relu(met_raw[:, 3]))
        x = torch.cat([met_norm, stat, torch.sin(2 * np.pi * doy / 365.25)[:, None],
                       torch.cos(2 * np.pi * doy / 365.25)[:, None]], dim=1)
        return self.net(x).squeeze(-1) * 10.0   # échelle mm/j

    def forward(self, w, stat, doy0):
        """Intègre SWE sur la séquence ; retourne SWE (B, 303)."""
        B = w.shape[0]
        wn = (w - M_MEAN.to(w.device)) / M_STD.to(w.device)
        swe = torch.zeros(B, device=w.device)
        out = []
        for t in range(w.shape[1]):
            met = w[:, t]
            tmean = 0.5 * (met[:, 1] + met[:, 2])
            fsnow = torch.sigmoid((self.t_snow - tmean) / 1.0)
            doy = (doy0 + t) % 365.25
            m = self.melt(met, wn[:, t], stat, doy)
            swe = torch.clamp(swe + met[:, 0] * fsnow - m, min=0.0)
            out.append(swe)
        return torch.stack(out, dim=1)


def presence(swe):
    return torch.sigmoid((swe - 10.0) / 4.0)   # SWE 10 mm ~ seuil de couvert détectable


def train_model(kind):
    model = MeltFn(kind).to(DEVICE)
    n_steps = STEPS * 3 if kind == "mlp" else STEPS   # 3 params convergent vite, pas le MLP
    opt = torch.optim.Adam(model.parameters(), lr=3e-3 if kind != "mlp" else 1e-3)
    pools = [(r, np.flatnonzero(r["seqs"][:, 3] == 0)) for r in train_regs]
    weights = np.array([len(p) for _, p in pools], dtype=np.float64); weights /= weights.sum()
    for step in range(n_steps):
        r, pool = pools[rng.choice(len(pools), p=weights)]
        idx = rng.choice(pool, min(BATCH, len(pool)), replace=False)
        w, o, stat, doy0 = [x.to(DEVICE) for x in gather(r, idx)]
        swe = model(w, stat, doy0)
        p = presence(swe)
        v = ~torch.isnan(o)
        obs = (o >= 0.5).float()
        loss = nn.functional.binary_cross_entropy(p[v].clamp(1e-5, 1 - 1e-5), obs[v])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"[{kind}] step {step} | BCE {float(loss):.4f}", flush=True)
    torch.save(model.state_dict(), f"{CKPT}/snow-{kind}.pt")
    return model


def meltout(days, is_snow):
    """Date de disparition ROBUSTE : premier jour valide après lequel TOUTES les
    obs valides restantes sont sans neige (dernière transition neige->sol nu).
    Insensible à un jour neigeux isolé tardif (contrairement au max des jours neigeux)."""
    if not is_snow.any():
        return int(days[0])
    last_snow_pos = np.flatnonzero(is_snow).max()
    return int(days[min(last_snow_pos + 1, len(days) - 1)])


def evaluate(model, r, idx):
    maes, accs = [], []
    for lo in range(0, len(idx), 2048):
        sub = idx[lo:lo + 2048]
        w, o, stat, doy0 = [x.to(DEVICE) for x in gather(r, sub)]
        with torch.no_grad():
            p = presence(model(w, stat, doy0)).cpu().numpy()
        o = o.cpu().numpy()
        # fenêtre mars-juin = jours 181+ de la séquence (1er sept -> 1er mars ~ j181)
        for b in range(p.shape[0]):
            v = np.flatnonzero(~np.isnan(o[b, 180:])) + 180
            if len(v) < 20: continue
            obs_p = o[b, v] >= 0.5
            pred_p = p[b, v] >= 0.5
            accs.append(float((obs_p == pred_p).mean()))
            # lissage anti-jour-isolé : un jour neigeux entouré de sol nu est ignoré
            def _despeckle(a):
                a = a.copy()
                iso = np.flatnonzero(a[1:-1] & ~a[:-2] & ~a[2:]) + 1
                a[iso] = False
                return a
            maes.append(abs(meltout(v, _despeckle(obs_p)) - meltout(v, _despeckle(pred_p))))
    return dict(mae_days=float(np.mean(maes)), acc=float(np.mean(accs)), n=len(maes))


print(f"\n[banc fonte] {len(regions)} régions | tenues {SPATIAL_HELD} | device {DEVICE}", flush=True)
models = {k: train_model(k) for k in ["dd", "eti", "mlp"]}
rows = []
for r in regions:
    held = r["name"] in SPATIAL_HELD
    periods = {"test_2022_2024": np.flatnonzero(r["seqs"][:, 3] == 2)}
    if held:
        periods["val_2019_2021"] = np.flatnonzero(r["seqs"][:, 3] == 1)
    for pname, idx in periods.items():
        if not len(idx): continue
        for k, m in models.items():
            met = evaluate(m, r, idx)
            rows.append(dict(region=r["name"], spatial_held=held, period=pname, model=k, **met))
            print(f"  {r['name']} {'(TENUE)' if held else '':7s} {pname} {k}: MAE date {met['mae_days']:.1f} j | acc {met['acc']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
for scope, sel in [("TENUES (leave-region-out)", df[df.spatial_held]),
                   ("TEMPOREL 2022-2024", df[~df.spatial_held & (df.period == "test_2022_2024")])]:
    if sel.empty: continue
    print(f"\n== {scope} ==")
    print(sel.groupby("model")[["mae_days", "acc"]].median().round(3).to_string())
print("[banc fonte] DONE")
