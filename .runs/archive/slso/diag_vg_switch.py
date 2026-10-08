"""Diagnostic de l'hypothèse INTERRUPTEUR van Genuchten (Essi 2026-06-16).

Si le NeRF pousse n vers le régime raide et que K(Se) s'effondre brutalement,
le sol bascule tout-ou-rien (absorbe tout / rejette tout) au lieu du continuum.
On mesure, sur un checkpoint entraîné :
  1. distribution du vg_n appris (poussé vers la borne 2.7 = raide ?)
  2. raideur de K(Se)/K_sat (largeur de Se où K passe de 0.5 à 0.05)
  3. fréquence où Se colle aux bornes [0.01, 0.99] en simulation (= interrupteur)

  python .runs/slso/diag_vg_switch.py [checkpoint]
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CKPT = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso/checkpoints/best-phase1-grace.pt"
CFG = ".runs/slso/config/slso.toml"
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device=dev); n = h["n_nodes"]
ds = xr.open_dataset(".runs/slso/" + cfg["paths"]["forcing_cache"])
ff = ds["forcing"].values.astype("float32"); times = pd.to_datetime(ds["time"].values); ds.close()
# 3 ans pour des trajectoires theta representatives
fc = torch.from_numpy(ff[:1095]).to(dev)
doy = torch.tensor(times[:1095].dayofyear.values, dtype=torch.long, device=dev)
wd = cache.load_withdrawals(str(times[0].date()), str(times[1094].date()), device=dev)

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"])
m = HydroModel(**kw).to(dev); m.load(CKPT); m.temperature = None
m.eval()

# 1. params NeRF par nœud
with torch.no_grad():
    terr_t = h["territorial"].to_tensor() if hasattr(h["territorial"], "to_tensor") else h["territorial"]
    params = m.spatial_encoder(h["node_coords"], terr_t)
vg_n = params.vg_n.detach().cpu().numpy()
por = params.porosity_1.detach().cpu().numpy()
fc1 = params.theta_fc_1.detach().cpu().numpy()
wp1 = params.theta_wp_1.detach().cpu().numpy()
Ks1 = params.K_sat_1.detach().cpu().numpy()
print("=== 1. vg_n appris (borné [1.3, 2.7], init 1.5) ===")
print(f"  min {vg_n.min():.2f}  p25 {np.percentile(vg_n,25):.2f}  med {np.median(vg_n):.2f}  p75 {np.percentile(vg_n,75):.2f}  max {vg_n.max():.2f}")
print(f"  fraction n>2.4 (très raide) : {(vg_n>2.4).mean()*100:.1f}%   fraction n<1.5 : {(vg_n<1.5).mean()*100:.1f}%")

# 2. raideur K(Se) pour n médian et n du p90
def K_of_Se(Se, nn):
    mm = 1.0 - 1.0/nn
    Se = np.clip(Se, 1e-4, 1-1e-6)
    inner = 1.0 - np.clip(Se**(1.0/mm), 0, 1-1e-6)
    return Se**0.5 * (1.0 - inner**mm)**2
Se = np.linspace(0.01, 0.99, 400)
print("=== 2. raideur K(Se)/K_sat : plage de Se où K/Ks passe de 0.5 à 0.05 ===")
for tag, nn in [("n médian", float(np.median(vg_n))), ("n p90", float(np.percentile(vg_n,90))), ("n=2.7 max", 2.7)]:
    K = K_of_Se(Se, nn)
    # Se où K=0.5 et K=0.05
    def se_at(target):
        idx = np.argmin(np.abs(K - target)); return Se[idx]
    s50, s05 = se_at(0.5), se_at(0.05)
    print(f"  {tag:12} (n={nn:.2f}) : K=0.5 à Se={s50:.2f}, K=0.05 à Se={s05:.2f}  -> chute sur ΔSe={s50-s05:.2f}")

# 3. trajectoires theta : Se colle-t-il aux bornes ?
with torch.no_grad():
    _, _, diag = m.simulate(forcing=fc, initial_state=HydroState.zeros(n, device=dev),
                            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                            withdrawals=wd, day_of_year=doy, return_diagnostics=True)
# theta layers depuis diag si dispo, sinon recalcul impossible -> on prend theta1 via state?
th1 = getattr(diag, "theta1", None)
if th1 is None:
    print("=== 3. (theta non exposé dans diag, saute) ===")
else:
    th1 = th1.detach().cpu().numpy()  # (T, N)
    Se1 = (th1 - wp1[None,:]) / (por[None,:] - wp1[None,:] + 1e-6)
    Se1 = np.clip(Se1, 0, 1)
    near_lo = (Se1 < 0.05).mean(); near_hi = (Se1 > 0.95).mean(); mid = ((Se1>=0.05)&(Se1<=0.95)).mean()
    print("=== 3. Se couche 1 en simulation (3 ans) ===")
    print(f"  fraction du temps×nœud : Se<0.05 (sol vide) {near_lo*100:.1f}%  |  Se>0.95 (saturé) {near_hi*100:.1f}%  |  milieu {mid*100:.1f}%")
    print(f"  -> si les extrêmes dominent le milieu, le sol fonctionne en interrupteur")
print("DONE")
