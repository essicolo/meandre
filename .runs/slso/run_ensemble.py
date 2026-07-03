"""Inférence d'ENSEMBLE : propage un ensemble de forçages (PyGMET) à travers un
checkpoint méandre ENTRAÎNÉ pour obtenir l'incertitude de FORÇAGE sur le débit.

C'est là qu'est l'avantage décisif de méandre pour l'Atlas : entraîner UNE fois,
puis propager N membres en inférence (torch.no_grad), quasi gratuit grâce au
routage par opérateur. Là où faire passer 100 membres dans Hydrotel/Raven est
infaisable, méandre le fait en minutes.

  python .runs/slso/run_ensemble.py <config.toml> "D:/.../forcing-pygmet-ens*.nc"

Sortie : ensemble de Q (M, T, N), bandes par quantile, et couverture aux stations
(fraction des obs dans la bande 90% = fiabilité de l'incertitude de forçage).
L'incertitude de MODÈLE (tête sigma/quantile apprise) s'ajoute par-dessus.
"""
import os, sys, glob, time
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch, xarray as xr, tomllib, duckdb
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState


def build_model(cfg, n_nodes, n_territorial, n_forcing, device):
    """Reconstruit HydroModel depuis la config (fidèle à slso.py)."""
    m = cfg.get("model", {}); s = cfg.get("soil", {}); tr = cfg.get("training", {})
    sb = {"z2_min": s.get("z2_min", 0.30), "z2_max": s.get("z2_max", 1.50),
          "z3_min": s.get("z3_min", 0.50), "z3_max": s.get("z3_max", 4.00),
          "rain_hours_min": s.get("rain_hours_min", 3.0), "rain_hours_max": s.get("rain_hours_max", 24.0)}
    return HydroModel(
        n_nodes=n_nodes, n_territorial=n_territorial, n_forcing=n_forcing,
        context_window=m.get("context_window", 30), residual_history=m.get("residual_history", 14),
        max_travel_time=m.get("max_travel_time", 20),
        use_temporal=tr.get("enable_temporal_epoch", 0) < 9999,
        use_residual=tr.get("enable_residual_epoch", 9999) < 9999,
        use_travel_time_attn=tr.get("enable_travel_epoch", 9999) < 9999,
        param_mode=m.get("param_mode", "nerf"), dropout=m.get("dropout", 0.0),
        soil_z1=s.get("z1", 0.30), soil_vsa_b=s.get("vsa_b", 2.5),
        soil_separate_infil_capacity=s.get("separate_infil_capacity", False),
        soil_frozen_gate=s.get("frozen_gate", False), soil_mode=s.get("mode", "meandre"),
        soil_clone_substep=s.get("clone_substep", 48), soil_clone_krec_init=s.get("clone_krec_init", 1e-5),
        et_mode=cfg.get("et", {}).get("mode", "penman"), column_mode=m.get("column_mode", "meandre"),
        column_theta_init_frac=m.get("column_theta_init_frac", 0.9),
        use_frost_rankinen=m.get("use_frost_rankinen", True),
        use_hillslope_uh=m.get("use_hillslope_uh", False), melt_mode=m.get("melt_mode", "degree_day"),
        use_aquifer=m.get("use_aquifer", False), use_hortonian=m.get("use_hortonian", False),
        soil_bounds=sb, routing_mode=m.get("routing_mode", "level"),
        predict_lake_params=m.get("predict_lake_params", False), n_coord_freqs=m.get("n_coord_freqs", 6),
        discharge_dependent_celerity=m.get("discharge_dependent_celerity", False),
        dq_beta=float(m.get("dq_beta", 0.4)),
    ).to(device)


def quantile_bands(Q_ens, taus=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """(M,T,N) -> dict tau -> (T,N). Bandes d'incertitude de forçage."""
    return {t: np.quantile(Q_ens, t, axis=0) for t in taus}


def station_coverage(bands, obs_df, dates, node_idx, lo=0.05, hi=0.95):
    """Fraction des obs dans la bande [lo,hi] par station (fiabilité)."""
    covs = []
    ql, qh = bands[lo], bands[hi]
    for sid, ni in node_idx.items():
        o = obs_df[obs_df.station_id == sid][["date", "discharge"]]
        mo = o.merge(pd.DataFrame({"date": dates, "lo": ql[:, ni], "hi": qh[:, ni]}), on="date")
        mo = mo.dropna()
        if len(mo) < 60: continue
        inside = ((mo.discharge >= mo.lo) & (mo.discharge <= mo.hi)).mean()
        covs.append(inside)
    return np.array(covs)


def main():
    cfg_path = sys.argv[1]; member_glob = sys.argv[2]
    cfg = tomllib.load(open(cfg_path, "rb"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    basin_db = cfg["paths"].get("basin_db", "data/slso.duckdb")
    if not os.path.isabs(basin_db):
        basin_db = os.path.join(".runs/slso", basin_db)
    ckpt = cfg["paths"]["checkpoint"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(".runs/slso", ckpt)

    h = BasinCache(basin_db).load(device=dev)
    graph = h["graph"]; node_coords = h["node_coords"]; territorial = h["territorial"]
    n_nodes = node_coords.shape[0]; n_terr = territorial.to_tensor().shape[1]

    members = sorted(glob.glob(member_glob))
    print(f"[ensemble] {len(members)} membres | {n_nodes} nœuds | device {dev}")
    ds0 = xr.open_dataset(members[0]); n_forcing = ds0["forcing"].shape[2]
    times = pd.to_datetime(ds0["time"].values).normalize(); ds0.close()

    model = build_model(cfg, n_nodes, n_terr, n_forcing, dev)
    model.load(ckpt); model.eval()
    doy = torch.tensor(times.dayofyear.values, dtype=torch.long, device=dev)
    wd = h.get("withdrawals") or BasinCache(basin_db).load_withdrawals(
        str(times[0].date()), str(times[-1].date()), device=dev)
    state = h.get("initial_state") or HydroState.zeros(n_nodes, device=dev)

    Q_list = []; t0 = time.time()
    for i, mf in enumerate(members):
        F = torch.from_numpy(xr.open_dataset(mf)["forcing"].values.astype("float32")).to(dev)
        with torch.no_grad():
            Q, _ = model.simulate(forcing=F, initial_state=state, graph=graph,
                                  node_coords=node_coords, territorial=territorial,
                                  withdrawals=wd, day_of_year=doy)
        Q_list.append(Q.cpu().numpy())
        print(f"  membre {i+1}/{len(members)} : Q moy {Q.mean():.2f}  ({time.time()-t0:.1f}s cumulé)")
    Q_ens = np.stack(Q_list)  # (M, T, N)
    dt = time.time() - t0
    print(f"[ensemble] {len(members)} membres propagés en {dt:.1f}s ({dt/len(members):.1f}s/membre)")

    bands = quantile_bands(Q_ens)
    np.savez_compressed(".runs/slso/results/ensemble_pygmet.npz",
                        Q_ens=Q_ens.astype("float32"), dates=times.astype(str).values,
                        node_idx=np.array([int(s.node_idx) for _, s in _stations(basin_db).iterrows()]))
    # couverture aux stations
    st = _stations(basin_db)
    nidx = {s.station_id: int(s.node_idx) for _, s in st.iterrows()}
    obs = _obs(basin_db, str(times[0].date()), str(times[-1].date()))
    cov = station_coverage(bands, obs, times, nidx)
    if len(cov):
        print(f"[fiabilité] couverture bande 90% : médiane {np.median(cov):.2f} (cible 0.90)")
        print(f"  spread médian (P95-P05)/moy : {np.median((bands[0.95]-bands[0.05]).mean(0)/(Q_ens.mean((0,1))+1e-6)):.2f}")
    print("[ok] ensemble_pygmet.npz écrit")


def _stations(db):
    c = duckdb.connect(db, read_only=True)
    df = c.execute("SELECT station_id, node_idx FROM stations").fetchdf(); c.close(); return df

def _obs(db, t0, t1):
    c = duckdb.connect(db, read_only=True)
    df = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{t0}' AND date<='{t1}'").fetchdf()
    c.close(); df["date"] = pd.to_datetime(df["date"]).dt.normalize(); return df


if __name__ == "__main__":
    main()
