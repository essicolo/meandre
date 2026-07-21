"""Banc ET HORS-LIGNE (design reports/design_et_appris.md) : apprendre la relation
météo + territoire -> ETR 8-jours MOD16, contre les formules calées au mieux.

Splits pré-enregistrés :
  spatial  : leave-region-out (train sur 12 régions, tenues = gasp/sagu/mont)
  temporel : train <= 2018, val 2019-2021 (early stop), test 2022-2024 (toutes régions)
Critère de succès : le module bat McGuinness×K_c ET Linacre×coeff (LSQ sur train)
à la fois hors-région et hors-période. Sortie : reports/et_bench_results.csv.

  python .runs/quebec/et_bench.py          # banc complet
  ET_SMOKE=1 python .runs/quebec/et_bench.py   # câblage rapide (3 régions, 3 epochs)
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
from meandre.data.basin_cache import BasinCache
from hydrotel_clone.mcguinness import mcguinness_etp
from hydrotel_clone.linacre import linacre_etp

SMOKE = os.environ.get("ET_SMOKE", "0") == "1"
REGIONS = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont",
           "outm", "outv", "sagu", "slno", "slso", "vaud"]
SPATIAL_HELD = ["gasp", "sagu", "mont"]
if SMOKE:
    REGIONS = ["labi", "cndc", "vaud"]
    SPATIAL_HELD = ["vaud"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"
TRAIN_END, VAL_END = "2018-12-31", "2021-12-31"
H_HIST, H_COMP = 90, 8           # 90 j d'historique + les 8 j du composite
H_SEQ = H_HIST + H_COMP
NODE_CAP = 1500                  # borne RAM : nœuds terrestres échantillonnés par région
BATCH = 4096
STEPS_PER_EPOCH = 20 if SMOKE else 80
MAX_EPOCHS = 3 if SMOKE else 40
PATIENCE = 4
PLATFORMS = "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
DBS = {"slso": ".runs/slso/data/slso.duckdb"}
FORCINGS = {"slso": "D:/meandre-data/slso/forcing-casr-corr.nc"}
OUT_CSV = "reports/et_bench_results.csv"
CKPT = "D:/meandre-data/quebec/checkpoints-etbench"
rng = np.random.default_rng(0)
torch.manual_seed(0)


def load_region(reg):
    db = DBS.get(reg, f"D:/meandre-data/quebec/{reg}.duckdb")
    fx = FORCINGS.get(reg, f"D:/meandre-data/quebec/forcing-{reg}-budyko.nc")
    if not os.path.exists(fx):   # vaud : pas de variante budyko (région ajoutée en dernier)
        fx = f"D:/meandre-data/quebec/forcing-{reg}.nc"
    cache = BasinCache(db)
    h = cache.load(device="cpu")
    n_nodes, node_ids = h["n_nodes"], h["node_ids"]
    coords = h["node_coords"].numpy()
    lat_col = 0 if 40 < np.nanmean(coords[:, 0]) < 62 else 1
    lat = coords[:, lat_col].astype(np.float32)
    static = h["territorial"].data.numpy().astype(np.float32)
    columns = list(h["territorial"].columns)

    d = xr.open_dataset(fx)
    meteo = d["forcing"].values[:, :, :6].astype(np.float32)   # P, Tmin, Tmax, R_n, u2, e_a
    times = pd.to_datetime(d["time"].values); d.close()
    assert str(times[0])[:10] == DATE_START and str(times[-1])[:10] == DATE_END, f"{reg}: fenêtre forçage"

    et = cache.load_modis_et(DATE_START, DATE_END, device="cpu").numpy()   # (T, N) mm/j au jour de début de composite

    # nœuds retenus : terrestres (>= 200 composites valides), plafonnés NODE_CAP
    n_valid = np.isfinite(et).sum(axis=0)
    keep = np.flatnonzero(n_valid >= 200)
    if len(keep) > NODE_CAP:
        keep = rng.choice(keep, NODE_CAP, replace=False); keep.sort()
    meteo, et, static, lat = meteo[:, keep], et[:, keep], static[keep], lat[keep]

    lin = None
    try:
        from meandre.data.hydrotel_calib import load_linacre_nodes
        lp = load_linacre_nodes(f"{PLATFORMS}/{reg.upper()}_LN24HA_2020", node_ids, device="cpu")
        lin = tuple(p[keep].numpy().astype(np.float32) if p.ndim else float(p) for p in lp)
    except Exception as e:
        print(f"[{reg}] linacre indisponible ({type(e).__name__}) — banc Linacre = n/a")

    # paires (t, n) valides, fenêtre complète
    tt, nn = np.nonzero(np.isfinite(et))
    ok = (tt >= H_HIST) & (tt + H_COMP <= len(times))
    tt, nn = tt[ok].astype(np.int32), nn[ok].astype(np.int32)
    dates = times[tt]
    split = np.where(dates <= TRAIN_END, 0, np.where(dates <= VAL_END, 1, 2))
    print(f"[{reg}] {len(keep)} nœuds | {len(tt):,} paires (train {int((split==0).sum()):,})", flush=True)
    return dict(name=reg, meteo=meteo, et=et, static=static, lat=lat, times=times,
                columns=columns, lin=lin, tt=tt, nn=nn, split=split)


regions = [load_region(r) for r in REGIONS]
cols0 = regions[0]["columns"]
for r in regions:
    assert r["columns"] == cols0, f"{r['name']}: colonnes territoriales != {regions[0]['name']}"
F_STATIC = regions[0]["static"].shape[1]
train_regs = [r for r in regions if r["name"] not in SPATIAL_HELD]
OFFS = np.arange(-H_HIST, H_COMP, dtype=np.int32)   # fenêtre relative au début de composite


def gather(r, idx):
    """Retourne (raw_windows (B,98,6), static (B,F+3), y (B,)) pour les paires idx d'une région."""
    t, n = r["tt"][idx], r["nn"][idx]
    w = torch.from_numpy(r["meteo"][ (t[:, None] + OFFS)[..., None], n[:, None, None], np.arange(6)[None, None, :] ])
    doy = r["times"].dayofyear.values[t + 4].astype(np.float32)
    stat = np.concatenate([r["static"][n],
                           np.sin(2 * np.pi * doy / 365.25)[:, None],
                           np.cos(2 * np.pi * doy / 365.25)[:, None],
                           (r["lat"][n] / 50.0)[:, None]], axis=1)
    y = r["et"][t, n]
    return w, torch.from_numpy(stat), torch.from_numpy(y)


# normalisation météo : stats sur un échantillon train (régions d'entraînement seulement)
_s = []
for r in train_regs:
    idx = np.flatnonzero(r["split"] == 0)
    _s.append(gather(r, rng.choice(idx, min(2000, len(idx)), replace=False))[0])
_s = torch.cat(_s).reshape(-1, 6)
M_MEAN, M_STD = _s.mean(0), _s.std(0) + 1e-6
del _s


class GruET(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + F_STATIC + 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, w, stat):
        h = self.gru((w - M_MEAN.to(w.device)) / M_STD.to(w.device))[1][-1]
        return self.head(torch.cat([h, stat], dim=1)).squeeze(-1)


class MlpET(nn.Module):
    """Ablation sans mémoire : agrégats de fenêtre (moyennes 8 j et 90 j par canal)."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(12 + F_STATIC + 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, w, stat):
        wn = (w - M_MEAN.to(w.device)) / M_STD.to(w.device)
        agg = torch.cat([wn[:, -H_COMP:].mean(1), wn[:, :H_HIST].mean(1)], dim=1)
        return self.head(torch.cat([agg, stat], dim=1)).squeeze(-1)


def formula_preds(r, idx, which):
    """ETP formule (coeff 1) moyennée sur les 8 j du composite. mm/j."""
    t, n = r["tt"][idx], r["nn"][idx]
    w = torch.from_numpy(r["meteo"][ (t[:, None] + np.arange(H_COMP, dtype=np.int32))[..., None], n[:, None, None], np.arange(6)[None, None, :] ])
    tmin, tmax = w[:, :, 1], w[:, :, 2]
    lat = torch.from_numpy(r["lat"][n])[:, None]
    doy = torch.from_numpy(np.stack([r["times"].dayofyear.values[ti:ti + H_COMP] for ti in t]).astype(np.float32))
    if which == "mcg":
        return mcguinness_etp(tmin, tmax, lat, doy).mean(1)
    if r["lin"] is None:
        return None
    llat, alti, tf, tc, alb, _ = r["lin"]
    z = torch.zeros_like(tmin)
    return linacre_etp(tmin, tmax, torch.from_numpy(llat[n])[:, None], torch.from_numpy(alti[n])[:, None], z, z,
                       t_froid=torch.from_numpy(tf[n])[:, None], t_chaud=torch.from_numpy(tc[n])[:, None],
                       albedo=torch.from_numpy(alb[n])[:, None], coeff=1.0).mean(1)


def fit_lsq(which):
    """k = argmin ||k·ETP - y|| sur le train des régions d'entraînement (la formule dans son meilleur jour)."""
    sxy = sxx = 0.0
    for r in train_regs:
        if which == "lin" and r["lin"] is None: continue
        idx = np.flatnonzero(r["split"] == 0)
        idx = rng.choice(idx, min(40000, len(idx)), replace=False)
        x = formula_preds(r, idx, which)
        y = torch.from_numpy(r["et"][r["tt"][idx], r["nn"][idx]])
        v = torch.isfinite(x) & torch.isfinite(y)
        sxy += float((x[v] * y[v]).sum()); sxx += float((x[v] ** 2).sum())
    return sxy / max(sxx, 1e-9)


def train_model(model, tag):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pools = [(r, np.flatnonzero(r["split"] == 0)) for r in train_regs]
    weights = np.array([len(p) for _, p in pools], dtype=np.float64); weights /= weights.sum()
    best, best_state, bad = -9e9, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            r, pool = pools[rng.choice(len(pools), p=weights)]
            w, stat, y = gather(r, rng.choice(pool, BATCH))
            w, stat, y = w.to(DEVICE), stat.to(DEVICE), y.to(DEVICE)
            loss = ((model(w, stat) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # early stop : R² pooled val 2019-2021 des régions d'entraînement
        model.eval(); se = sv = 0.0
        with torch.no_grad():
            for r in train_regs:
                idx = np.flatnonzero(r["split"] == 1)
                idx = rng.choice(idx, min(20000, len(idx)), replace=False)
                w, stat, y = gather(r, idx)
                p = model(w.to(DEVICE), stat.to(DEVICE)).cpu()
                se += float(((p - y) ** 2).sum()); sv += float(((y - y.mean()) ** 2).sum())
        r2 = 1 - se / max(sv, 1e-9)
        marker = ""
        if r2 > best + 1e-4:
            best, bad = r2, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " -> best"
        else:
            bad += 1
        print(f"[{tag}] epoch {epoch:2d} | val R² {r2:.4f}{marker}", flush=True)
        if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best_state, f"{CKPT}/{tag}.pt")
    return model


def evaluate(pred_fn, r, idx):
    ps, ys = [], []
    for lo in range(0, len(idx), 20000):
        sub = idx[lo:lo + 20000]
        p = pred_fn(r, sub)
        if p is None: return None
        ps.append(p); ys.append(torch.from_numpy(r["et"][r["tt"][sub], r["nn"][sub]]))
    p, y = torch.cat(ps), torch.cat(ys)
    v = torch.isfinite(p) & torch.isfinite(y)
    p, y = p[v].numpy(), y[v].numpy()
    return dict(r2=1 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9),
                rmse=float(np.sqrt(((p - y) ** 2).mean())),
                bias_pct=100.0 * (p.mean() - y.mean()) / max(y.mean(), 1e-9), n=len(y))


print(f"\n[banc] {len(regions)} régions | tenues spatiales {SPATIAL_HELD} | device {DEVICE}", flush=True)
k_mcg = fit_lsq("mcg"); k_lin = fit_lsq("lin")
print(f"[banc] K_c McGuinness (LSQ train) = {k_mcg:.3f} | coeff Linacre (LSQ train) = {k_lin:.3f}", flush=True)
gru = train_model(GruET(), "gru")
mlp = train_model(MlpET(), "mlp")

preds = {
    "gru": lambda r, i: torch.no_grad()(lambda: gru(*[x.to(DEVICE) for x in gather(r, i)[:2]]).cpu())(),
    "mlp": lambda r, i: torch.no_grad()(lambda: mlp(*[x.to(DEVICE) for x in gather(r, i)[:2]]).cpu())(),
    "mcg": lambda r, i: formula_preds(r, i, "mcg") * k_mcg,
    "lin": lambda r, i: (lambda x: None if x is None else x * k_lin)(formula_preds(r, i, "lin")),
}
rows = []
for r in regions:
    held = r["name"] in SPATIAL_HELD
    periods = {"test_2022_2024": np.flatnonzero(r["split"] == 2)}
    if held:
        periods["val_2019_2021"] = np.flatnonzero(r["split"] == 1)
    for pname, idx in periods.items():
        if not len(idx): continue
        for mname, fn in preds.items():
            m = evaluate(fn, r, idx)
            if m is None: continue
            rows.append(dict(region=r["name"], spatial_held=held, period=pname, model=mname, **m))
            print(f"  {r['name']} {'(TENUE)' if held else '':7s} {pname} {mname}: R² {m['r2']:.3f} | RMSE {m['rmse']:.3f} | biais {m['bias_pct']:+.1f}%", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\n[banc] résultats -> {OUT_CSV}")
for scope, sel in [("TENUES (leave-region-out, 2019-2024)", df[df.spatial_held]),
                   ("TEMPOREL 2022-2024 (régions d'entraînement)", df[~df.spatial_held & (df.period == "test_2022_2024")])]:
    if sel.empty: continue
    print(f"\n== {scope} ==")
    piv = sel.groupby("model").apply(lambda g: pd.Series(dict(r2_med=g.r2.median(), rmse_med=g.rmse.median(), bias_med=g.bias_pct.median())), include_groups=False)
    print(piv.round(3).to_string())
print("[banc] DONE")
