"""Fusion CaSR + jauges GHCN (conditional merging) sur la précipitation.
Pour chaque nœud n et jour t :
    P_fused(n,t) = P_CaSR(n,t) + Σ_g w(n,g)·[P_gauge(g,t) − P_CaSR@gauge(g,t)]
w = IDW (rayon limité), renormalisé chaque jour sur les jauges DISPONIBLES. Garde
la structure CaSR loin des jauges, l'épingle aux jauges près d'elles (corrige la
DATE et le cumul des orages). Les autres canaux (T, R_n, u2, e_a, DT_eff) inchangés.
Entrée : forcing-casr-riox-intens.nc (7 canaux). Sortie : forcing-casr-fused.nc.
  python .runs/slso/build_fused_forcing.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache
from meandre.data.ghcn_loader import fetch_ghcn_precip

IN = ".runs/slso/data/forcing-casr-riox-intens.nc"
OUT = ".runs/slso/data/forcing-casr-fused.nc"
RADIUS_KM = 40.0          # rayon d'influence d'une jauge
POWER = 2.0               # IDW puissance

h = BasinCache(".runs/slso/data/slso.duckdb").load(device="cpu")
nc = h["node_coords"].numpy(); nlon, nlat = nc[:, 0], nc[:, 1]; n_nodes = len(nc)
ds = xr.open_dataset(IN); times = pd.to_datetime(ds["time"].values).normalize()
F = ds["forcing"].values.copy(); VARS = list(ds["var"].values.astype(str)); ds.close()
P_casr = F[:, :, 0]        # (T, n) mm/j
print(f"nœuds {n_nodes}, jours {len(times)}, canaux {VARS}")

# 1) jauges GHCN (dans le domaine + marge)
g = fetch_ghcn_precip((-73.5, 44.0, -69.0, 48.0), str(times[0].date()), str(times[-1].date()))
gid = g.station_id.unique()
gcoord = g.groupby("station_id")[["lon", "lat"]].first().loc[gid].values
# matrice jauge (date × gauge), NaN si manquant
gp = g.pivot_table(index="date", columns="station_id", values="prcp_mm").reindex(times)
gp = gp[gid]                                          # ordre cohérent
G = gp.values                                         # (T, n_gauge) NaN=manquant
print(f"jauges {len(gid)} ; couverture moyenne {np.isfinite(G).mean()*100:.0f}% des jours")

# 2) CaSR au point de jauge = nœud le plus proche
_deg2km = 111.0
node_tree = cKDTree(np.c_[nlon * np.cos(np.radians(nlat.mean())), nlat])
_, gnode = node_tree.query(np.c_[gcoord[:, 0] * np.cos(np.radians(nlat.mean())), gcoord[:, 1]])
P_casr_at_g = P_casr[:, gnode]                        # (T, n_gauge)
resid = G - P_casr_at_g                               # (T, n_gauge) résidu jauge-CaSR

# 3) poids IDW gauge -> node (n_node × n_gauge), rayon limité
gx = gcoord[:, 0] * np.cos(np.radians(nlat.mean())); gy = gcoord[:, 1]
nx = nlon * np.cos(np.radians(nlat.mean())); ny = nlat
dist_km = np.sqrt((nx[:, None] - gx[None, :])**2 + (ny[:, None] - gy[None, :])**2) * _deg2km
W = np.where(dist_km < RADIUS_KM, 1.0 / np.clip(dist_km, 1.0, None)**POWER, 0.0)   # (n_node, n_gauge)
print(f"IDW : {(W>0).sum(1).mean():.1f} jauges/nœud en moyenne dans le rayon {RADIUS_KM}km")

# 4) fusion jour par jour (vectorisé sur nœuds) : masque les jauges manquantes
P_fused = P_casr.copy()
for t in range(len(times)):
    r = resid[t]; avail = np.isfinite(r)
    if not avail.any(): continue
    Wt = W[:, avail]; rr = r[avail]
    wsum = Wt.sum(1)
    corr = np.where(wsum > 0, (Wt @ rr) / np.clip(wsum, 1e-9, None), 0.0)
    P_fused[t] = np.clip(P_casr[t] + corr, 0.0, None)

F[:, :, 0] = P_fused
print(f"P moyen : CaSR {P_casr.mean()*365.25:.0f} -> fused {P_fused.mean()*365.25:.0f} mm/an")
# combien la fusion a bougé, près vs loin des jauges
near = (W > 0).any(1)
print(f"|ΔP| moyen : nœuds PRÈS d'une jauge {np.abs(P_fused-P_casr)[:,near].mean():.2f} mm/j, "
      f"LOIN {np.abs(P_fused-P_casr)[:,~near].mean():.2f} mm/j (loin ~0 attendu)")

out = xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
                 coords={"time": times.values, "node": np.arange(n_nodes), "var": VARS})
out.to_netcdf(OUT)
print(f"[ok] cache fusionné écrit : {OUT}")
