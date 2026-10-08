"""Que vaut VRAIMENT l'incumbent ? KGE par station d'Hydrotel OPÉRATIONNEL (calé
par bassin, sorties officielles) vs meandre (qb=PHYSITEL, ksat05=CaSR), par taille
de bassin. Combien de stations Hydrotel lui-même franchit 0.8 ? Test 2022-2024.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb

T0, T1 = "2022-01-01", "2024-12-31"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

ds = xr.open_dataset(HYDRO); sidh = ds["station_id"].values.astype(str)
dis = ds["Dis"].sel(time=slice(T0, T1)); th = pd.to_datetime(dis["time"].values).normalize()
disv = dis.values; ds.close()

def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o) & (o >= 0)
    if m.sum() < 60: return None
    s, o = s[m], o[m]
    if s.std() == 0 or o.std() == 0: return None
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def meandre_kge(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()
    out = {}
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date")
        if len(mo) >= 60: out[s.station_id] = kge(mo.qm.to_numpy(), mo.discharge.to_numpy())
    return out
mq = meandre_kge("qb"); mk = meandre_kge("ksat05")

rows = []
for _, s in st.iterrows():
    sid = s.station_id; o = obs[obs.station_id == sid][["date", "discharge"]]
    if len(o) < 60: continue
    idx = np.where(sidh == sid)[0]
    kh = np.nan
    if len(idx):
        hm = o.merge(pd.DataFrame({"date": th, "qh": disv[:, idx[0]]}), on="date")
        if len(hm) >= 60: kh = kge(hm.qh.to_numpy(), hm.discharge.to_numpy())
    rows.append((sid, s.drainage_area_km2, kh, mq.get(sid, np.nan), mk.get(sid, np.nan)))
df = pd.DataFrame(rows, columns=["station", "area", "hydrotel", "meandre_phys", "meandre_casr"]).sort_values("area", ascending=False)
pd.set_option("display.width", 160)
print(df.round(2).to_string(index=False))
print(f"\n--- médianes ---  Hydrotel {np.nanmedian(df.hydrotel):.3f}  meandre_PHYS {np.nanmedian(df.meandre_phys):.3f}  meandre_CaSR {np.nanmedian(df.meandre_casr):.3f}")
for col in ["hydrotel", "meandre_phys", "meandre_casr"]:
    v = df[col].dropna()
    print(f"  {col:13s} : >0.8 {int((v>0.8).sum())}/{len(v)}  >0.7 {int((v>0.7).sum())}/{len(v)}  médiane {v.median():.3f}")
# Hydrotel atteint-il 0.8 sur les GROS bassins ?
big = df[df.area > 500]
print(f"\nGros bassins (>500 km², n={len(big)}) : Hydrotel méd {np.nanmedian(big.hydrotel):.3f}  meandre_CaSR méd {np.nanmedian(big.meandre_casr):.3f}")
