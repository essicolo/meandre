"""Prototype de PRÉTRAITEMENT HYDROLOGIQUE de CaSR (littérature 2026-07-02).
Biais documentés CaPA v3.2 : sur-estime R95pTOT ×2 (volume) + biais CRACHIN
(trop de petits <2mm, sous-estime les gros) qui ÉTALE le timing. Corrige :
  1. DÉ-CRACHINAGE : les jours 0<P<θ voient leur masse déplacée sur le pic d'orage
     local (±win jours) si c'en est un — concentre les vrais orages, conserve la masse.
  2. VOLUME : échelle globale vers la fermeture du bilan (ET MODIS ~450 + Q).
Mesure plafond de timing (UH linéaire) + RC avant/après. Écrit forcing-casr-hydro.nc.
CPU seulement.  python .runs/slso/build_casr_hydro.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb

IN = ".runs/slso/data/forcing-casr-riox-intens.nc"
OUT = ".runs/slso/data/forcing-casr-hydro.nc"
THETA = 1.0        # seuil crachin (mm/j)
WIN = 2            # fenêtre de concentration (±jours)
VOL_SCALE = 0.93   # bilan flux-tower : 1147 cible (ET 450 + Q 697) / 1229 CaSR

ds = xr.open_dataset(IN); VARS = list(ds["var"].values.astype(str))
F = ds["forcing"].values.copy(); times = pd.to_datetime(ds["time"].values).normalize()
ds.close()
P = F[:, :, 0].copy()                      # (T,N)
T, N = P.shape

# ── 1. DÉ-CRACHINAGE : concentration locale conservatrice ──
Pd = P.copy()
for t in range(T):
    driz = (Pd[t] > 0.0) & (Pd[t] < THETA)          # (N,) crachin au jour t
    if not driz.any(): continue
    lo, hi = max(0, t - WIN), min(T, t + WIN + 1)
    win = Pd[lo:hi]                                   # (w,N)
    rel = win.argmax(0)                              # jour du pic local par nœud
    peak = win.max(0)
    move = driz & (peak >= THETA) & (rel != (t - lo))  # déplacer si vrai orage voisin
    if move.any():
        day_idx = (lo + rel)[move]
        node_idx = np.where(move)[0]
        np.add.at(Pd, (day_idx, node_idx), Pd[t, move])
        Pd[t, move] = 0.0
moved_frac = 1.0 - (Pd > 0).sum() / max((P > 0).sum(), 1)
print(f"dé-crachinage : jours pluvieux {(P>0).sum()/(T*N)*100:.1f}% -> {(Pd>0).sum()/(T*N)*100:.1f}% "
      f"({moved_frac*100:.0f}% des jours pluvieux dé-crachinés), masse conservée : "
      f"{Pd.sum()/P.sum():.4f}")

# ── 2. plafond de timing AVANT/APRÈS (UH linéaire, échantillon stations) ──
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute("SELECT station_id,date,discharge FROM observations WHERE date>='2004-01-01' AND date<='2021-12-31'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
Tmean = 0.5 * (F[:, :, 1] + F[:, :, 2])
K = 45
def melt_apport(p, tm):
    swe = 0.0; A = np.empty_like(p)
    for i in range(len(p)):
        if tm[i] < 0: swe += p[i]; A[i] = 0.0
        else:
            m = min(swe, 3.0 * tm[i]); swe -= m; A[i] = p[i] + m
    return A
def ceiling_r(Psrc):
    rs = []
    for _, s in st.iterrows():
        ni = int(s.node_idx)
        A = melt_apport(Psrc[:, ni], Tmean[:, ni])
        o = obs[obs.station_id == s.station_id]
        mo = pd.DataFrame({"date": times}).merge(o[["date", "discharge"]], on="date", how="left")
        q = mo.discharge.to_numpy()
        X = np.zeros((T, K + 1))
        for k in range(K + 1): X[k:, k] = A[:T - k]
        m = np.isfinite(q)
        if m.sum() < 300: continue
        h = np.linalg.solve(X[m].T @ X[m] + np.eye(K + 1), X[m].T @ q[m])
        qh = X @ h
        rs.append(np.corrcoef(qh[m], q[m])[0, 1])
    return np.median(rs)
r_before = ceiling_r(P)
r_after = ceiling_r(Pd)
print(f"plafond timing (UH médian) : CaSR brut {r_before:.3f} -> dé-crachiné {r_after:.3f}  ({r_after-r_before:+.3f})")

# ── 3. VOLUME + écriture ──
Ph = Pd * VOL_SCALE
print(f"volume : P {P.mean()*365.25:.0f} -> {Ph.mean()*365.25:.0f} mm/an (×{VOL_SCALE})")
F[:, :, 0] = Ph.astype(np.float32)
if os.path.exists(OUT):
    os.remove(OUT)
out = xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
                 coords={"time": times.values, "node": np.arange(N), "var": VARS})
out.to_netcdf(OUT, engine="h5netcdf")
out.close()
print(f"[ok] écrit {OUT}")
