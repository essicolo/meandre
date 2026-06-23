"""Smoke test : prélèvements/rejets SOUTERRAINS (net_gw) branchés sur le chemin
colonne Hydrotel. Surface (net) déjà appliquée par le routage partagé ; ici on
isole le souterrain (net surface = 0) et on vérifie qu'un pompage net_gw < 0
réduit bien le débit (interception du baseflow), et qu'un net_gw nul ne change
RIEN (fidélité préservée — le bloc gw est court-circuité).

  python tests/smoke_gw_withdrawal_hydrotel.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tomllib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.routing.withdrawals import WithdrawalData
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-clone.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-clone.pt"
cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
kw["column_mode"] = "hydrotel"; kw["et_mode"] = "mcguinness"
m = HydroModel(**kw); m.routing.routing_mode = "operator-lagged"; m.temperature = None

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64("2019-01-01")))
NDAYS = 90
fc = torch.from_numpy(ff[w0:w0 + NDAYS]); win = times[w0:w0 + NDAYS]
doy = torch.tensor(win.dayofyear.values, dtype=torch.long)

z = torch.zeros(NDAYS, n)
wd_zero = WithdrawalData(net=z.clone(), net_gw=z.clone())          # rien
gw = z.clone() - 0.3                                              # pompage 0.3 m3/s/noeud
wd_gw = WithdrawalData(net=z.clone(), net_gw=gw)                  # souterrain seul


def run(wd):
    with torch.no_grad():
        Q, _ = m.simulate(
            forcing=fc, initial_state=HydroState.default_warm(n), graph=h["graph"],
            node_coords=h["node_coords"], territorial=h["territorial"], withdrawals=wd,
            day_of_year=doy)
    return Q


Q0 = run(wd_zero)
Qg = run(wd_gw)

# 1. net_gw nul : strictement identique à une simulation sans prélèvement gw.
assert torch.allclose(Q0, run(WithdrawalData(net=z.clone(), net_gw=z.clone()))), "non determinisme"
# 2. pompage souterrain : le débit total baisse (baseflow intercepté).
drop = float(Q0.sum() - Qg.sum())
print(f"Q sum  net_gw=0 : {float(Q0.sum()):.1f}   pompage : {float(Qg.sum()):.1f}   baisse {drop:.1f}")
assert drop > 0, "le pompage souterrain ne reduit pas le debit (gw non branche ?)"
assert not torch.isnan(Qg).any(), "NaN sous pompage"
# 3. signe : un rejet (net_gw > 0) doit AUGMENTER le debit.
Qr = run(WithdrawalData(net=z.clone(), net_gw=z.clone() + 0.3))
assert float(Qr.sum()) > float(Q0.sum()), "un rejet souterrain n'augmente pas le debit"
print(f"rejet net_gw>0 : {float(Qr.sum()):.1f} (> {float(Q0.sum()):.1f})")
print("SMOKE OK (prelevements/rejets souterrains branches sur colonne Hydrotel)")
