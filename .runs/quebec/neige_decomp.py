"""DECOMPOSITION DU RETARD : la neige fond-elle au mauvais moment, ou fond-elle à
l'heure et l'eau met-elle trop de temps à sortir ? Le champ krigé a MESURE que le
modèle comprime le gradient nord-sud de la date de crue de 40 %. Ce banc localise le
défaut, avec des observations indépendantes du débit (MOD10A1, couvert nival) :

  date de disparition du couvert SIMULEE vs OBSERVEE  -> défaut du module de fonte
  date de disparition SIMULEE vs date de crue SIMULEE -> défaut du transfert sol/gel

  PYTHONIOENCODING=utf-8 python .runs/quebec/neige_decomp.py mont gasp sagu
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.basin_cache import BasinCache
from joint_data import load_region, DATE_START, DATE_END
from et_module import compute_demand

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

CKPT = os.environ.get("NEIGE_CKPT", ".runs/quebec/checkpoints/best-gasp-etl-ds.pt")
SWE_REF = 15.0   # mm, même conversion SWE -> fraction que la loss (snow_swe_ref)
SCF_SEUIL = 0.5  # fraction de couvert définissant la disparition
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
rows = []

for REG in [a.lower() for a in sys.argv[1:]]:
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
        column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
        use_temperature=False, use_latent_codes=False, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(CKPT); m.eval(); m.vertical_column.etp_channel = 6
    with torch.no_grad():
        Q, _, diag = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                                graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                                withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                                return_diagnostics=True)
    times = pd.to_datetime(r["times"]); t0 = td.train_slice.start
    tt = pd.DatetimeIndex(times[t0:])
    scf_sim = (1.0 - torch.exp(-diag.swe / SWE_REF)).cpu().numpy()
    obs = BasinCache(".runs/slso/data/slso.duckdb" if REG == "slso" else f"{_DATA_ROOT}/quebec/{REG}.duckdb") \
        .load_modis_snow(DATE_START, DATE_END, device="cpu")
    scf_obs = obs.numpy()[t0:t0 + len(tt)] if obs is not None else None
    Qs = Q.cpu().numpy()

    def date_disparition(scf):
        """Jour julien où le couvert passe sous le seuil (médiane inter-annuelle), par nœud."""
        acc = []
        for y in range(2001, 2025):
            s = np.asarray((tt >= f"{y}-02-15") & (tt <= f"{y}-07-15"))
            w = scf[s]
            if w.shape[0] < 100: continue
            doy = tt[s].dayofyear.values.astype(float)
            below = np.where(np.isfinite(w), w, 1.0) < SCF_SEUIL
            # dernier jour AU-DESSUS du seuil = date de disparition
            idx = np.where(~below, np.arange(len(doy))[:, None], -1).max(axis=0)
            acc.append(np.where(idx >= 0, doy[np.clip(idx, 0, None)], np.nan))
        return np.nanmedian(np.stack(acc), axis=0)

    def cm_freshet(Qa):
        acc = []
        for y in range(2001, 2025):
            s = np.asarray((tt >= f"{y}-03-01") & (tt <= f"{y}-06-30"))
            w = Qa[s]
            if w.shape[0] < 90: continue
            doy = tt[s].dayofyear.values.astype(float)[:, None]
            tot = np.clip(w.sum(0), 1e-6, None)
            acc.append((w * doy).sum(0) / tot)
        return np.nanmedian(np.stack(acc), axis=0)

    d_sim = date_disparition(scf_sim)
    d_obs = date_disparition(scf_obs) if scf_obs is not None else np.full(n, np.nan)
    cm = cm_freshet(Qs)
    ok = np.isfinite(d_sim) & np.isfinite(d_obs)
    rows.append(dict(region=REG, n_nodes=n, n_ok=int(ok.sum()),
                     fonte_sim=round(float(np.nanmedian(d_sim)), 1),
                     fonte_obs=round(float(np.nanmedian(d_obs[ok])), 1),
                     biais_fonte=round(float(np.nanmedian(d_sim[ok] - d_obs[ok])), 1),
                     cm_sim=round(float(np.nanmedian(cm)), 1),
                     delai_fonte_crue=round(float(np.nanmedian(cm - d_sim)), 1)))
    print(f"[{REG}] disparition neige sim j{np.nanmedian(d_sim):.1f} vs MOD10 j{np.nanmedian(d_obs[ok]):.1f} "
          f"(biais {np.nanmedian(d_sim[ok]-d_obs[ok]):+.1f} j) | CM crue sim j{np.nanmedian(cm):.1f} "
          f"| delai fonte->crue {np.nanmedian(cm-d_sim):+.1f} j", flush=True)
    del m, Q, diag; torch.cuda.empty_cache()

df = pd.DataFrame(rows); df.to_csv("reports/neige_decomp.csv", index=False)
print(df.to_string(index=False))
