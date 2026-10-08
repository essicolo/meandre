"""r plat : forçage-limité ou modèle ? Ajuste le MEILLEUR filtre linéaire possible
(hydrogramme unitaire = convolution ridge de l'apport CaSR) par station, sans
aucune contrainte physique. Son r vs obs = plafond d'information LINÉAIRE du
forçage. Si méandre (Hortonien) est déjà à ce plafond, le timing n'est pas dans
CaSR -> irréductible. S'il est loin dessous, r est réparable (routage affamé).
Apport = pluie + fonte degré-jour (bucket neige simple). CPU seulement.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, xarray as xr

T0, T1 = "2004-01-01", "2021-12-31"   # train+dev (le ceiling est un ajustement, pas une prédiction)
K = 45            # longueur de l'hydrogramme unitaire (jours)
RIDGE = 1.0       # régularisation
MF = 3.0          # mm/°C/j fonte degré-jour

c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

FORCING = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso/data/forcing-casr-riox-intens.nc"
print(f"### forçage testé : {FORCING}")
ds = xr.open_dataset(FORCING)
ft = pd.to_datetime(ds["time"].values).normalize()
sl = (ft >= pd.Timestamp(T0)) & (ft <= pd.Timestamp(T1))
P = ds["forcing"].values[sl][..., 0]      # (T,n) pluie+neige mm/j
Tmin = ds["forcing"].values[sl][..., 1]; Tmax = ds["forcing"].values[sl][..., 2]
ftn = ft[sl]; ds.close()
Tmean = 0.5 * (Tmin + Tmax)

def apport(p, tm):
    """pluie + fonte : neige s'accumule si T<0, fond à MF·T si T>0."""
    swe = 0.0; A = np.empty_like(p)
    for t in range(len(p)):
        if tm[t] < 0:
            swe += p[t]; A[t] = 0.0
        else:
            melt = min(swe, MF * tm[t]); swe -= melt
            A[t] = p[t] + melt
    return A

def uh_r(A, q, mask):
    """ajuste UH ridge (Q = conv(A, h)) et retourne r(Q_hat, Q_obs) sur mask."""
    T = len(A)
    X = np.zeros((T, K + 1))
    for k in range(K + 1):
        X[k:, k] = A[:T - k]
    m = mask & np.isfinite(q)
    if m.sum() < 300: return None
    Xm = X[m]; qm = q[m]
    # ridge : (X'X + λI) h = X'q
    XtX = Xm.T @ Xm + RIDGE * np.eye(K + 1)
    h = np.linalg.solve(XtX, Xm.T @ qm)
    qhat = X @ h
    return np.corrcoef(qhat[m], qm)[0, 1]

# méandre r (Hortonien) sur la même période, pour comparer
dH = np.load(".runs/slso/results/eval_hortonian.npz", allow_pickle=True)
QH = dH["Q"]; ddH = pd.to_datetime(dH["dates"].astype(str)).normalize()

rows = []
for _, s in st.iterrows():
    ni = int(s.node_idx)
    A = apport(P[:, ni], Tmean[:, ni])
    o = obs[obs.station_id == s.station_id][["date", "discharge"]]
    if len(o) < 300: continue
    mo = pd.DataFrame({"date": ftn}).merge(o, on="date", how="left")
    q = mo.discharge.to_numpy()
    # ceiling annuel + estival
    r_all = uh_r(A, q, np.ones(len(A), bool))
    r_sum = uh_r(A, q, np.array(pd.Series(ftn).dt.month.isin([6, 7, 8, 9])))
    # méandre r (période test dispo dans l'éval, on prend ce qui recoupe)
    mm = o.merge(pd.DataFrame({"date": ddH, "qm": QH[:, ni]}), on="date")
    r_me = np.corrcoef(mm.qm, mm.discharge)[0, 1] if len(mm) > 200 else np.nan
    if r_all is not None:
        rows.append(dict(sid=s.station_id, r_ceiling=r_all, r_ceiling_ete=r_sum, r_meandre=r_me))

df = pd.DataFrame(rows)
print(f"{len(df)} stations | UH K={K}j, ridge={RIDGE}, fonte MF={MF}\n")
print(f"r PLAFOND forçage (UH linéaire optimal)  : médian {df.r_ceiling.median():.3f}  (annuel)")
print(f"r PLAFOND forçage été (jun-sep)          : médian {df.r_ceiling_ete.median():.3f}")
print(f"r MÉANDRE Hortonien (test)               : médian {df.r_meandre.median():.3f}")
gap = df.r_ceiling.median() - df.r_meandre.median()
print(f"\nÉCART plafond - méandre : {gap:+.3f}")
if gap < 0.05:
    print(">>> méandre est AU plafond du forçage : r IRRÉDUCTIBLE, le timing n'est pas dans CaSR.")
elif gap < 0.12:
    print(">>> écart modéré : un peu de marge, mais le forçage limite déjà fort.")
else:
    print(">>> GROS écart : le forçage CONTIENT le timing, méandre le laisse sur la table -> RÉPARABLE.")
