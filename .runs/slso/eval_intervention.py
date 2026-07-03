"""Sweep en EVAL (forward seul, no_grad, pas d'entrainement) du modele entraine,
avec UNE intervention a la fois sur le mecanisme du freshet. Sort le debit aux
noeuds jauges -> scored par eval_score.py (tete-a-tete vs Hydrotel + obs).

Interventions (argv[1]) :
  none      baseline (modele tel quel)
  tair-2    refroidit Tmin/Tmax de 2 C (teste biais chaud / lapse rate -> fonte precoce)
  tair-4    refroidit de 4 C
  melt0.5   divise les facteurs de fonte degre-jour par 2 (teste fonte trop agressive)
  melt0.7   x0.7
  frostoff  porte de gel OFF (use_frost=False)
  froston   porte de gel ON (force)
  ksat0.3   K_sat des 3 couches x0.3 (teste sol trop drainant -> fonte s'infiltre)
  puredv    routage advection pure (cinematique, teste gamma)

Tourne sur WSL (forcing natif). Sortie : .runs/slso/results/eval_<interv>.npz
  python .runs/slso/eval_intervention.py <interv>
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import math, tomllib
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.routing.withdrawals import WithdrawalData

INTERV = sys.argv[1] if len(sys.argv) > 1 else "none"
CKPT = os.environ.get("EVAL_CKPT", ".runs/slso/checkpoints/best-physitel-hydrotel-overnight.pt")
# Chemins Windows LOCAUX (CPU stable, evite le wedge WSL/GPU D-state).
FORC = os.environ.get("EVAL_FORC", ".runs/slso/data/forcing.nc")
DB = ".runs/slso/data/slso.duckdb"
SPIN0, TEST1 = "2020-01-01", "2024-12-31"   # 2 ans spinup + 3 test
dev = "cuda" if torch.cuda.is_available() else "cpu"

cache = BasinCache(DB); h = cache.load(device=dev); n = h["n_nodes"]
ck = torch.load(CKPT, map_location=dev, weights_only=False)
kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False  # pas de Triton Windows
m = HydroModel(**kw).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval()

# ── forcing fenetre ──
ds = xr.open_dataset(FORC); times = pd.to_datetime(ds["time"].values).normalize()
w0 = int(np.searchsorted(times, np.datetime64(SPIN0)))
w1 = int(np.searchsorted(times, np.datetime64(TEST1))) + 1
ff = ds["forcing"].values[w0:w1].astype(np.float32); ds.close()
win = times[w0:w1]
fc = torch.from_numpy(ff).to(dev)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=dev)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=dev)

# ── INTERVENTION(S) : tokens joints par "+" (ex "melt0.5+ksat0.3") ──
print(f"[interv] {INTERV}")
_ksat_scales = []
def apply_one(tok):
    if tok in ("none", ""):
        return
    if tok.startswith("lapse"):
        # correction altitudinale relative : dT = -gamma * (elev - elev_moy).
        # gamma optionnel apres "lapse" (ex lapse6.5 = 6.5 C/km), defaut 6.5.
        g = float(tok.replace("lapse", "")) if tok != "lapse" else 6.5
        elev = h["territorial"].get_physical("mean_elevation_m")
        if elev is None:
            fn = getattr(h["territorial"], "feature_names", []) or []
            elev = h["territorial"].to_tensor()[:, fn.index("mean_elevation_m")] if "mean_elevation_m" in fn else None
        assert elev is not None, "mean_elevation_m introuvable"
        elev = elev.to(fc.device).float()
        dTn = (-(g / 1000.0) * (elev - elev.mean())).reshape(1, -1)   # (1, n) C
        print(f"  [lapse] gamma={g} C/km, elev {float(elev.min()):.0f}-{float(elev.max()):.0f}m, dT {float(dTn.min()):.1f}..{float(dTn.max()):.1f}C")
        fc[:, :, 1] = fc[:, :, 1] + dTn
        fc[:, :, 2] = fc[:, :, 2] + dTn
    elif tok.startswith("tair"):
        dT = float(tok.replace("tair", ""))
        fc[:, :, 1] = fc[:, :, 1] + dT
        fc[:, :, 2] = fc[:, :, 2] + dT
    elif tok.startswith("melt"):
        s = float(tok.replace("melt", ""))
        with torch.no_grad():
            for nm in ("sp_fonte_conif", "sp_fonte_feu", "sp_fonte_dec"):
                p = getattr(m.vertical_column, nm, None)
                if p is not None:
                    eff = torch.nn.functional.softplus(p) * s
                    p.copy_(torch.log(torch.expm1(eff.clamp(min=1e-4))))
    elif tok == "frostoff":
        m.vertical_column.use_frost = False
    elif tok == "froston":
        m.vertical_column.use_frost = True
    elif tok.startswith("ksat"):
        _ksat_scales.append(float(tok.replace("ksat", "")))
    elif tok == "puredv":
        m.routing.pure_advection = True
    elif tok.startswith("dqcel"):
        # célérité dépendante du débit (onde cinématique DANS le Muskingum) :
        # K_eff = K·(Qref/(Q+Qref))^beta → pic voyage plus vite, s'atténue moins.
        # beta optionnel après "dqcel" (ex dqcel0.6), défaut = celui du modèle.
        m.routing.dq_celerity = True
        b = tok.replace("dqcel", "")
        if b:
            m.routing.dq_beta = float(b)
    else:
        raise ValueError(f"intervention inconnue: {tok}")

for tok in INTERV.split("+"):
    apply_one(tok)
if _ksat_scales:
    sc = float(np.prod(_ksat_scales))
    orig = m.spatial_encoder.forward
    def patched(*a, **k):
        sp = orig(*a, **k)
        for nm in ("K_sat_1", "K_sat_2", "K_sat_3"):
            setattr(sp, nm, getattr(sp, nm) * sc)
        return sp
    m.spatial_encoder.forward = patched

# ── EVAL forward ──
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.default_warm(n, device=dev),
                      graph=h["graph"], node_coords=h["node_coords"],
                      territorial=h["territorial"], withdrawals=wd, day_of_year=doy)
Q = Q.detach().cpu().numpy().astype(np.float32)   # (T, n)
out = f".runs/slso/results/eval_{INTERV}.npz"
np.savez_compressed(out, Q=Q, dates=win.values.astype("datetime64[D]").astype(str))
print(f"[ok] {out}  Q{Q.shape}  mean {np.nanmean(Q):.2f}  NaN={bool(np.isnan(Q).any())}")
