"""IMPORT DIRECT DES LACS D'HYDROTEL depuis troncon.trl (découverte 2026-08-08).

Le C++ (`troncons.cpp::LectureLac`) lit pour chaque tronçon-lac : longueur, SURFACE (km²),
C, K — la loi de tarage calibrée Q = c·h^k (k = 1.5, déversoir). Le parseur de méandre
lisait ces champs mais JETAIT c et k, et prenait la surface pour une largeur. Or la loi
de méandre Q = k_lake·(S/A)^beta·A est EXACTEMENT équivalente avec beta = k,
A = surface·1e6 (m²) et k_lake = c/A. On importe donc les lacs d'Hydrotel tels quels,
sans apprentissage : tarage calibré + vraie surface d'eau.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/lacs_trl.py outv slno sagu
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_PLAT_ROOT = _osp.environ.get("MEANDRE_PLATFORMS", "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel")

CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
rows = []

def lire_lacs_trl(reg, node_ids):
    p = Path(f"{_PLAT_ROOT}/LN24HA/{reg.upper()}_LN24HA_2020/physitel/troncon.trl")
    lignes = [l.strip() for l in p.read_text(encoding="latin-1").splitlines() if l.strip()]
    d = {}
    for l in lignes[3:]:
        t = l.split()
        if int(t[1]) != 1:
            ptr = 4 + int(t[3])
            d[int(t[0])] = (float(t[ptr+1]), float(t[ptr+2]), float(t[ptr+3]))  # surface km2, c, k
    idx = {int(i): j for j, i in enumerate(node_ids)}
    n = len(node_ids)
    surf = np.full(n, np.nan); c = np.full(n, np.nan); k = np.full(n, np.nan)
    for tid, (s_, c_, k_) in d.items():
        if tid in idx:
            surf[idx[tid]], c[idx[tid]], k[idx[tid]] = s_, c_, k_
    return surf, c, k

for REG in [a.lower() for a in sys.argv[1:]]:
    ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]
    surf, c, k = lire_lacs_trl(REG, r["node_ids"])
    lac = td.graph.is_lake.bool().cpu().numpy()
    couv = np.isfinite(surf) & lac & (surf > 0)
    A_m2 = surf * 1e6
    k_trl = np.where(couv, c / np.clip(A_m2, 1.0, None), np.nan)
    print(f"[{REG}] lacs trl : {int(couv.sum())}/{int(lac.sum())} | surface méd {np.nanmedian(surf[couv]):.2f} km² "
          f"| c méd {np.nanmedian(c[couv]):.1f} | k_lake méd {np.nanmedian(k_trl[couv]):.2e} /s "
          f"| beta = {np.nanmedian(k[couv]):.2f}", flush=True)
    lat_ok = a_des_latents(ck, n)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
        * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=lat_ok, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
    _orig = m.spatial_encoder.lake_params
    couv_t = torch.tensor(couv, device=DEVICE)
    kt = torch.tensor(np.nan_to_num(k_trl, nan=1e-4), dtype=torch.float32, device=DEVICE)
    bt = torch.tensor(np.nan_to_num(k, nan=1.5), dtype=torch.float32, device=DEVICE)

    def run(mode):
        if mode == "trl":
            m.set_lake_area(torch.tensor(np.where(couv, surf, np.nan_to_num(surf, nan=1.0)),
                                         dtype=torch.float32))
            def lp(*a, _o=_orig, **kw):
                kk, bb = _o(*a, **kw)
                return torch.where(couv_t, kt, kk), torch.where(couv_t, bt, bb)
            m.spatial_encoder.lake_params = lp
        else:
            m.set_lake_area(None)
            m.spatial_encoder.lake_params = _orig
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        del Q; torch.cuda.empty_cache()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        return float(np.median(ks))

    base = run("ref")
    v = run("trl")
    rows.append(dict(region=REG, ref=round(base, 4), lacs_trl=round(v, 4), delta=round(v-base, 4)))
    print(f"[{REG}] reference {base:.4f} | lacs Hydrotel (trl) {v:.4f} ({v-base:+.4f})", flush=True)
    m.set_lake_area(None); m.spatial_encoder.lake_params = _orig
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv("reports/lacs_trl.csv", index=False)
