"""Pourquoi le kge_med de la lignée pointue (Hortonien) reste bas ? Décompose
PAR STATION vs baseline (ksat05) : KGE et ses composantes r/beta/gamma, croisées
avec l'aire et le BFI. Répond : plafond physique généralisé, ou quelques petits
UHRH réactifs qui plombent la médiane ? Test 2022-2024. N'utilise pas le GPU.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb

T0, T1 = "2022-01-01", "2024-12-31"
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 AS area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

def kge_decomp(s, o):
    m = np.isfinite(s) & np.isfinite(o) & (o >= 0)
    if m.sum() < 60: return None
    s, o = s[m], o[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0: return None
    r = np.corrcoef(s, o)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())
    kge = 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
    return kge, r, beta, gamma

def load(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    return d["Q"], pd.to_datetime(d["dates"].astype(str)).normalize()

rows = []
for tag in ["ksat05", "hortonian"]:
    Q, dd = load(tag)
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date")
        r = kge_decomp(mo.qm.to_numpy(), mo.discharge.to_numpy())
        if r is None: continue
        rows.append(dict(tag=tag, sid=s.station_id, area=s.area_km2,
                         kge=r[0], r=r[1], beta=r[2], gamma=r[3]))
df = pd.DataFrame(rows)
p = df.pivot(index="sid", columns="tag", values=["kge", "r", "beta", "gamma"]).dropna()
area = df.groupby("sid").area.first()

print(f"{len(p)} stations communes\n")
for tag in ["ksat05", "hortonian"]:
    k = p["kge"][tag]
    print(f"{tag:10s} : KGE med {k.median():.3f} | moy {k.mean():.3f} | "
          f"min {k.min():.3f} | <0.5 : {(k<0.5).sum()}/{len(k)} | <0 : {(k<0).sum()}")
print()
# qui plombe : delta hortonian - ksat05 par station, trié
d = (p["kge"]["hortonian"] - p["kge"]["ksat05"]).sort_values()
print("5 stations où l'Hortonien PERD le plus vs baseline :")
for sid in d.index[:5]:
    a = area[sid]
    print(f"  {sid} (aire {a:6.0f} km²) : KGE {p['kge']['ksat05'][sid]:.2f}->{p['kge']['hortonian'][sid]:.2f}  "
          f"| r {p['r']['ksat05'][sid]:.2f}->{p['r']['hortonian'][sid]:.2f}  "
          f"gamma {p['gamma']['ksat05'][sid]:.2f}->{p['gamma']['hortonian'][sid]:.2f}")
print("\n5 stations où l'Hortonien GAGNE le plus :")
for sid in d.index[-5:]:
    a = area[sid]
    print(f"  {sid} (aire {a:6.0f} km²) : KGE {p['kge']['ksat05'][sid]:.2f}->{p['kge']['hortonian'][sid]:.2f}  "
          f"| r {p['r']['ksat05'][sid]:.2f}->{p['r']['hortonian'][sid]:.2f}  "
          f"gamma {p['gamma']['ksat05'][sid]:.2f}->{p['gamma']['hortonian'][sid]:.2f}")
# corrélation perte vs aire
print(f"\ncorrélation (delta KGE, log aire) : {np.corrcoef(d.values, np.log(area[d.index].values))[0,1]:+.2f}")
print("  (négatif = les PETITS bassins perdent le plus = UHRH réactifs plombent)")
# composante moyenne où on perd
for comp in ["r", "beta", "gamma"]:
    hc, kc = p[comp]["hortonian"], p[comp]["ksat05"]
    print(f"  {comp} médian : baseline {kc.median():.3f} -> hortonien {hc.median():.3f} "
          f"(écart à 1.0 : {abs(kc.median()-1):.3f} -> {abs(hc.median()-1):.3f})")
