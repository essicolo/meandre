"""RÉSERVOIR LINÉAIRE PHYSIQUE : k = 1/temps_de_séjour (HydroLAKES), beta = 1.
Avec beta = 1, la loi implémentée Q = k*(S/A)^beta*A se réduit à Q = k*S : le coefficient
de vidange est alors exactement l'inverse du temps de séjour, quantité MESURÉE par lac
(volume / débit moyen, Messager et al. 2016). Aucun seuil ni exposant à choisir.

Constat qui motive le test : les temps de séjour observés (93-368 j selon la région)
donnent k entre 2.7e-3 et 1.1e-2 /j, alors que le modèle utilise 1e-4 — ses lacs vident
30 à 100 fois trop lentement, ce qui est le symptôme d'amortissement excessif mesuré sur
les stations lacustres (-0.22 de KGE au-dessus de 5 % de nœuds-lacs amont). La borne
supérieure du paramètre (1e-2) est de surcroît juste sous la valeur physique des lacs les
plus rapides : le réseau ne pouvait pas l'atteindre.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/lacs_hydrolakes_ab.py outv slno sagu gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
HL = pd.read_parquet(f"{_DATA_ROOT}/quebec/lacs_hydrolakes.parquet")
KMAX = float(os.environ.get("HL_KMAX", "0.01"))    # borne haute actuelle du modele
# La borne BASSE du modele (1e-6 /s) est 30x AU-DESSUS du k physique d'un lac de 368 j
# (3.1e-8 /s) : les bornes ont ete calibrees pour beta = 1.5, ou k n'a pas les memes
# unites. On desserre le plancher pour le test du reservoir lineaire.
KMIN = float(os.environ.get("HL_KMIN", "1e-9"))
rows = []

for REG in [a.lower() for a in sys.argv[1:]]:
    ck = f".runs/quebec/checkpoints/{LOCAUX[REG]}.pt"
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]
    lat = a_des_latents(ck, n)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
        * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=lat, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
    _orig = m.spatial_encoder.lake_params

    hl = HL[HL.region == REG].set_index("node_idx")
    tau = np.full(n, np.nan)
    tau[hl.index.values] = hl["res_time_j"].values
    couv = np.isfinite(tau) & (tau > 0)
    # UNITES : Q_out = k * (S/A)^beta * A avec Q en m3/s et S en m3, donc avec beta = 1
    # le coefficient est en INVERSE DE SECONDE. k = 1/(tau_jours * 86400). Premiere version
    # du test : k = 1/tau_jours, soit 86400x trop grand — les lacs se vidaient en un pas de
    # temps, d'ou la saturation mesuree (-0.0078, exactement la valeur du regime sans lac).
    k_phys = torch.tensor(np.where(couv, 1.0 / (np.clip(tau, 1.0, None) * 86400.0), np.nan),
                          dtype=torch.float32, device=DEVICE)
    couv_t = torch.tensor(couv, device=DEVICE)
    print(f"[{REG}] {int(couv.sum())} nœuds avec temps de séjour | k physique méd "
          f"{np.nanmedian(1.0/np.clip(tau[couv],1,None)):.2e} /j (modèle : 1e-4)", flush=True)

    def essai(mode, kmax=KMAX):
        if mode == "ref":
            m.spatial_encoder.lake_params = _orig
        else:
            def lp(*a, _o=_orig, **kw):
                kk, bb = _o(*a, **kw)
                kp = torch.clamp(k_phys, KMIN, kmax)
                kk = torch.where(couv_t, kp, kk)
                if mode == "lin":
                    bb = torch.where(couv_t, torch.ones_like(bb), bb)
                return kk, bb
            m.spatial_encoder.lake_params = lp
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        del Q; torch.cuda.empty_cache()
        return float(np.median(ks))

    base = essai("ref")
    print(f"[{REG}] reference {base:.4f}", flush=True)
    # CORRECTION DE SURFACE : le module de lac recoit l'aire de DRAINAGE du troncon
    # (territorial.area_km2_physical) au lieu de la surface du PLAN D'EAU. Or
    # Q = k*(S/A)^beta*A = k*S^beta/A^(beta-1) : avec beta=1.5, Q varie comme 1/racine(A).
    # Surestimer A d'un facteur 1000 divise le debit sortant par ~30, ce qui est
    # exactement l'amortissement mesure sur les stations lacustres. Emulation de la
    # surface correcte sans toucher au routage : k_eff = k * (A_drainage/A_lac)^(beta-1).
    _Adr = r["territorial"].area_km2_physical
    _Adr = _Adr.to(DEVICE) if _Adr is not None else None
    _Alac = torch.full((n,), float("nan"), device=DEVICE)
    _Alac[torch.tensor(hl.index.values, device=DEVICE)] = torch.tensor(
        hl["lake_area_km2"].values, dtype=torch.float32, device=DEVICE)
    _corr = None
    if _Adr is not None:
        _rap = torch.clamp(_Adr / torch.clamp(_Alac, min=1e-3), min=1.0)
        _corr = _rap
        print(f"[{REG}] rapport aire drainage / surface lac : med {float(_rap[couv_t].median()):.0f}x "
              f"-> correction de k mediane x{float((_rap[couv_t]**0.5).median()):.1f}", flush=True)

    def essai_surface():
        def lp(*a, _o=_orig, **kw):
            kk, bb = _o(*a, **kw)
            fac = torch.where(couv_t, torch.clamp(_corr, 1.0, 1e6) ** (bb - 1.0),
                              torch.ones_like(kk))
            return torch.clamp(kk * fac, 1e-8, 1.0), bb
        m.spatial_encoder.lake_params = lp
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s2 in range(Qs.shape[1]):
            o, si = qo[:, s2], Qs[:, s2]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        del Q; torch.cuda.empty_cache()
        return float(np.median(ks))

    if _corr is not None:
        v = essai_surface()
        rows.append(dict(region=REG, essai="surface de lac corrigee", kge=round(v, 4), delta=round(v-base, 4)))
        print(f"[{REG}] {'surface de lac corrigee':26s} KGE {v:.4f} ({v-base:+.4f})", flush=True)
    for mode, lib in [("k", "k = 1/tau, beta inchange"), ("lin", "k = 1/tau, beta = 1")]:
        v = essai(mode)
        rows.append(dict(region=REG, essai=lib, kge=round(v, 4), delta=round(v-base, 4)))
        print(f"[{REG}] {lib:26s} KGE {v:.4f} ({v-base:+.4f})", flush=True)
    m.spatial_encoder.lake_params = _orig
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv("reports/lacs_hydrolakes_ab.csv", index=False)
