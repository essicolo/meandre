"""EXP6 : correcteur d'erreurs ATTRIBUT-CONDITIONNÉ (transformer) post-hoc sur physique gelée.
Hypothèse (Essi) : les erreurs résiduelles systématiques sont dues aux ATTRIBUTS des bassins
(taille, pente, occupation, texture) ; un réseau attention peut les apprendre et les corriger.
Design anti-surapprentissage (leçons exp5/5b) :
  - correction multiplicative BORNÉE [0.74, 1.35] (exp(0.3*tanh)), physique intouchée ;
  - features jour MINIMALES (Q_sim, saison, P récente) ; le signal riche = les attributs ;
  - VALIDATION LOSO (leave-one-station-out) : le correcteur d'une station n'a JAMAIS vu
    ses obs ; s'il gagne quand même, il généralise aux bassins NON JAUGÉS (régionalisation).
Deux modes : FULL (fit toutes stations, potentiel max) et LOSO (preuve de généralisation).
  python .runs/slso/exp6_attr_transformer.py [PARQUET]     ENV: MODE=full|loso|both (déf both)
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, torch, torch.nn as nn, xarray as xr
from meandre.data.basin_cache import BasinCache

torch.manual_seed(0); np.random.seed(0)
DB = ".runs/slso/data/slso.duckdb"
PARQUET = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso/results/reach-physitel-hydrotel-casr-corr.parquet"
FORCING = os.environ.get("FORCING", "D:/meandre-data/slso/forcing-casr-corr.nc")
MODE = os.environ.get("MODE", "both")
TRAIN = ("2000-01-01", "2018-12-31"); DEV = ("2019-01-01", "2021-12-31"); TEST = ("2022-01-01", "2024-12-31")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── données ──────────────────────────────────────────────────────────────────
h = BasinCache(DB).load(device="cpu")
terr = h["territorial"]; A_cols = terr.columns; A_all = terr.data.numpy()  # (n_nodes, 16)
c = duckdb.connect(DB, read_only=True)
stations = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute("SELECT station_id, date, discharge FROM observations").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()

fx = xr.open_dataset(FORCING)
fx_time = pd.to_datetime(fx["time"].values)
sta_rows = []
for _, st in stations.iterrows():
    rid = int(st.node_idx) + 1
    s = sim[sim.reach_id == rid].set_index("date")["Q_sim_m3s"].sort_index()
    o = obs[obs.station_id == st.station_id].set_index("date")["discharge"].sort_index()
    if len(s) == 0: continue
    j = pd.DataFrame({"qs": s}).join(o.rename("qo"), how="left")
    n_test_obs = j.loc[TEST[0]:TEST[1], "qo"].notna().sum()
    n_train_obs = j.loc[TRAIN[0]:TRAIN[1], "qo"].notna().sum()
    if n_test_obs < 100 or n_train_obs < 365: continue
    p = pd.Series(fx["forcing"].values[:, int(st.node_idx), 0], index=fx_time).reindex(j.index).fillna(0.0)
    j["p3"] = p.rolling(3, min_periods=1).sum(); j["p14"] = p.rolling(14, min_periods=1).sum()
    sta_rows.append((st.station_id, int(st.node_idx), j))
fx.close()
print(f"{len(sta_rows)} stations utilisables | device {DEVICE}")

A_mu, A_sd = A_all.mean(0), A_all.std(0) + 1e-9  # stats non supervisées (tous nœuds)
def kge_np(qs, qo):
    m = np.isfinite(qs) & np.isfinite(qo); qs, qo = qs[m], qo[m]
    if len(qs) < 60 or qo.std() < 1e-9: return np.nan
    r = np.corrcoef(qs, qo)[0, 1]; b = qs.mean()/qo.mean(); g = (qs.std()/qs.mean())/(qo.std()/qo.mean())
    return 1 - np.sqrt((r-1)**2 + (b-1)**2 + (g-1)**2)

def make_tensors(j, node_idx):
    doy = j.index.dayofyear.values
    qs = j["qs"].values.astype(np.float32); scale = max(np.nanmean(qs), 1e-3)
    X = np.stack([qs/scale, np.log1p(qs), np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                  np.log1p(j["p3"].values), np.log1p(j["p14"].values)], 1).astype(np.float32)
    A = ((A_all[node_idx] - A_mu) / A_sd).astype(np.float32)
    return X, A, qs, j["qo"].values.astype(np.float32), scale

class AttrCorrector(nn.Module):
    """Tokens = 16 attributs (valeur×w+b appris par attribut) + token jour ; encodeur transformer ;
    sortie du token jour -> facteur multiplicatif borné."""
    def __init__(self, n_attr=16, d=32):
        super().__init__()
        self.attr_w = nn.Parameter(torch.randn(n_attr, d) * 0.1)
        self.attr_b = nn.Parameter(torch.zeros(n_attr, d))
        self.day_in = nn.Linear(6, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=64,
                                         batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # départ = identité
    def forward(self, X, A):
        # X (T, 6) jour ; A (16,) attributs -> facteur (T,)
        T = X.shape[0]
        tok_a = A[:, None] * self.attr_w + self.attr_b            # (16, d)
        tok_d = self.day_in(X)                                    # (T, d)
        seq = torch.cat([tok_d[:, None, :], tok_a[None].expand(T, -1, -1)], 1)  # (T, 17, d)
        z = self.enc(seq)[:, 0, :]                                # token jour
        return torch.exp(0.3 * torch.tanh(self.out(z).squeeze(-1)))

def fit_eval(train_stations, eval_stations, tag, epochs=60):
    """Entraîne sur train_stations (période TRAIN, early-stop DEV), évalue TEST sur eval_stations."""
    model = AttrCorrector().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    packs = []
    for sid, nid, j in train_stations:
        X, A, qs, qo, scale = make_tensors(j, nid)
        tr = (j.index >= TRAIN[0]) & (j.index <= TRAIN[1]) & np.isfinite(qo)
        dv = (j.index >= DEV[0]) & (j.index <= DEV[1]) & np.isfinite(qo)
        packs.append((torch.tensor(X, device=DEVICE), torch.tensor(A, device=DEVICE),
                      torch.tensor(qs, device=DEVICE), torch.tensor(np.nan_to_num(qo), device=DEVICE),
                      torch.tensor(tr.values if hasattr(tr, "values") else tr, device=DEVICE),
                      dv, j, scale))
    REL = os.environ.get("REL", "1") == "1"  # contrainte RELATIVE : zéro-somme du log-facteur
    best_dev, best_state, patience = -9e9, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss = 0.0; logf_means = []
        for X, A, qs, qo, tr, dv, j, scale in packs:
            f = model(X, A); qc = qs * f
            m = tr
            l = ((torch.log1p(qc[m]) - torch.log1p(qo[m]))**2).mean() \
                + 0.5*(((qc[m] - qo[m])/scale)**2).mean()
            loss = loss + l
            logf_means.append(torch.log(f[m]).mean())
        loss = loss/len(packs)
        if REL:  # le correcteur ne peut PAS apprendre un décalage de niveau global
            loss = loss + 10.0 * torch.stack(logf_means).mean()**2
        loss.backward(); opt.step()
        model.eval(); devk = []
        with torch.no_grad():
            for X, A, qs, qo, tr, dv, j, scale in packs:
                qc = (qs * model(X, A)).cpu().numpy()
                devk.append(kge_np(qc[dv.values if hasattr(dv, "values") else dv],
                                   j["qo"].values[dv.values if hasattr(dv, "values") else dv]))
        dmed = np.nanmedian(devk)
        if dmed > best_dev + 1e-4: best_dev, best_state, patience = dmed, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); res = []
    with torch.no_grad():
        for sid, nid, j in eval_stations:
            X, A, qs, qo, scale = make_tensors(j, nid)
            f = model(torch.tensor(X, device=DEVICE), torch.tensor(A, device=DEVICE)).cpu().numpy()
            qc = qs * f
            te = (j.index >= TEST[0]) & (j.index <= TEST[1])
            res.append((sid, kge_np(qs[te], qo[te]), kge_np(qc[te], qo[te]), f[te].mean()))
    return res, best_dev

if MODE in ("full", "both"):
    res, bdev = fit_eval(sta_rows, sta_rows, "FULL")
    df = pd.DataFrame(res, columns=["sta", "kge_base", "kge_corr", "f_moy"])
    print(f"\n== FULL (toutes stations vues ; potentiel max) ==  dev_med={bdev:.3f}")
    print(f"held-out médian : base {df.kge_base.median():.4f} -> corrigé {df.kge_corr.median():.4f} "
          f"(delta {df.kge_corr.median()-df.kge_base.median():+.4f}) | gagnées {(df.kge_corr>df.kge_base).sum()}/{len(df)}")

if MODE in ("loso", "both"):
    # k-fold leave-group-out : même preuve de généralisation que le LOSO strict, k× moins cher
    FOLDS = int(os.environ.get("FOLDS", "6"))
    order = np.random.RandomState(0).permutation(len(sta_rows))
    rows = []
    for f_i in range(FOLDS):
        held_idx = set(order[f_i::FOLDS].tolist())
        train_s = [s for k, s in enumerate(sta_rows) if k not in held_idx]
        eval_s = [s for k, s in enumerate(sta_rows) if k in held_idx]
        res, _ = fit_eval(train_s, eval_s, f"FOLD-{f_i}", epochs=30)
        rows.extend(res)
        for r in res: print(f"  fold{f_i} {r[0]}: base {r[1]:.3f} -> corr {r[2]:.3f}", flush=True)
    df = pd.DataFrame(rows, columns=["sta", "kge_base", "kge_corr", "f_moy"])
    print(f"\n== LOSO {FOLDS}-fold (stations JAMAIS vues ; preuve régionalisation) ==")
    print(f"held-out médian : base {df.kge_base.median():.4f} -> corrigé {df.kge_corr.median():.4f} "
          f"(delta {df.kge_corr.median()-df.kge_base.median():+.4f}) | gagnées {(df.kge_corr>df.kge_base).sum()}/{len(df)}")
    df.to_csv(".runs/slso/results/exp6-loso.csv", index=False)
