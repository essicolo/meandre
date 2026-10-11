"""Le recalcul par segments (MEANDRE_SEGMENT_GRADIENT) doit rendre le MEME gradient qu'un
seul passage. Banc mini de la colonne Hydrotel, 60 jours, processeur.

  python tests/scripts/verifier_segments_gradient.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tomllib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-clone.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-clone.pt"
cfg = tomllib.load(open(CFG, "rb"))
cache = BasinCache(".runs/slso-od/" + cfg["paths"]["basin_db"])
h = cache.load(device="cpu")
n = h["n_nodes"]
kw = dict(torch.load(CKPT, map_location="cpu", weights_only=False)["init_kwargs"])
kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
kw["column_mode"] = "hydrotel"
kw["et_mode"] = "mcguinness"
torch.manual_seed(0)
m = HydroModel(**kw)
m.routing.routing_mode = "operator-lagged"
m.temperature = None

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values)
ff = ds["forcing"].values.astype(np.float32)
ds.close()
w0 = int(np.searchsorted(times, np.datetime64("2019-03-01")))
NDAYS = 60
fc = torch.from_numpy(ff[w0:w0 + NDAYS])
win = times[w0:w0 + NDAYS]
doy = torch.tensor(win.dayofyear.values, dtype=torch.long)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device="cpu")


def passe(segment: int, poursuivre: bool):
    os.environ["MEANDRE_SEGMENT_GRADIENT"] = str(segment)
    m.zero_grad()
    Q, st, diag = m.simulate(forcing=fc, initial_state=HydroState.default_warm(n), graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"], withdrawals=wd, day_of_year=doy, return_diagnostics=True, poursuivre_etat=poursuivre)
    # Perte qui depend du debit en fin de periode, donc de tout l'historique, et d'un diagnostic.
    perte = (Q[-20:] ** 2).mean() + diag.etr.mean() + st.theta3.mean()
    perte.backward()
    g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    aux = m.vertical_column._aux
    return Q.detach().clone(), g.clone(), float(aux.theta3.sum()), float(perte)


# Une premiere passe pose l'etat interne ; les suivantes le poursuivent, comme les blocs du trainer.
passe(0, False)
etat = m._instantane_attributs()
Q0, g0, a0, l0 = passe(0, True)
m._restaurer_attributs(etat)
Q7, g7, a7, l7 = passe(7, True)
dq = float((Q0 - Q7).abs().max())
dg = float((g0 - g7).abs().max() / g0.abs().max())
print(f"perte {l0:.6f} contre {l7:.6f} | ecart max du debit {dq:.2e} | ecart relatif max du gradient {dg:.2e} | etat interne final {a0:.4f} contre {a7:.4f} | norme du gradient {float(g0.norm()):.4e}")
assert dq < 1e-4 and dg < 1e-4 and abs(a0 - a7) < 1e-3, "le recalcul par segments change le resultat"
print("SEGMENTS OK : meme debit, meme gradient, meme etat interne final")
