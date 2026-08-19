"""APPARIEMENT MODULE PAR MODULE avec Hydrotel, sur ses fichiers d'ETAT.

Hydrotel écrit dans etat/ l'état interne de chaque sous-modèle à des dates données :
bilan_vertical (THETA 1/2/3 par UHRH), fonte_neige (stocks par classe de couvert),
acheminement_riviere (DEBIT AMONT/AVAL et APPORT par tronçon). Méandre expose les mêmes
variables (theta1/2/3, swe, lateral_mm, q_lateral). On compare donc les ETAGES de la
colonne au lieu du seul débit en bout de chaîne — ce qui localise la divergence au lieu
de la déduire par élimination.

Ne demande PAS de réexécuter Hydrotel (WSL en panne, ressources insuffisantes) : les états
du 2023-08-01 et du 2026-02-19 sont déjà sur disque, et le premier tombe dans la période
tenue de côté 2022-2024.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/apparier_hydrotel.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.physitel_loader import _parse_troncon
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
DATE = os.environ.get("HT_DATE", "2023-08-01")
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
ETAT = f"{PROJ}/etat"
CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tag = DATE.replace("-", "") + "00"

def lire(nom):
    f = f"{ETAT}/{nom}_{tag}.csv"
    return pd.read_csv(f, sep=";", skiprows=3) if os.path.exists(f) else None

bv = lire("bilan_vertical"); ach = lire("acheminement_riviere")
print(f"[hydrotel] {REG} {DATE} | bilan_vertical {None if bv is None else bv.shape} | "
      f"acheminement {None if ach is None else ach.shape}", flush=True)
if bv is None:
    raise SystemExit("etat bilan_vertical absent")

# UHRH -> troncon (meme agregation que hydrotel_calib : moyenne ponderee par l'aire UHRH)
from pathlib import Path as _P
tr = _parse_troncon(_P(PROJ) / "physitel" / "troncon.trl")
uh = pd.read_csv(f"{PROJ}/physitel/uhrh.csv", sep=";")
col_a = [c for c in uh.columns if "aire" in c.lower() or "area" in c.lower() or "km" in c.lower()]
print(f"[physitel] {len(tr)} troncons", flush=True)

# Le fichier d'acheminement est deja PAR TRONCON et de meme cardinalite que les noeuds
# de meandre : correspondance directe, aucune agregation UHRH necessaire.
ach.columns = [c.strip() for c in ach.columns]
ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
assert len(ach) == n, f"{len(ach)} troncons Hydrotel vs {n} noeuds meandre"
lat_ok = a_des_latents(ck, n)
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)     * AD.get(REG, {}).get("debias_et", 1.0)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
    use_latent_codes=lat_ok, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
with torch.no_grad():
    Q, _, diag = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                            graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                            withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                            return_diagnostics=True)
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
i = int(np.flatnonzero(tt == pd.Timestamp(DATE))[0])
print(f"[meandre] etat au {DATE} (indice {i})", flush=True)

def comp(nom, a_, b_, unite=""):
    a_ = np.asarray(a_, float); b_ = np.asarray(b_, float)
    v = np.isfinite(a_) & np.isfinite(b_)
    rr = np.corrcoef(a_[v], b_[v])[0, 1] if v.sum() > 10 and a_[v].std() > 0 and b_[v].std() > 0 else np.nan
    print(f"  {nom:22s} hydrotel med {np.median(a_[v]):10.4g} | meandre med {np.median(b_[v]):10.4g} "
          f"| rapport {np.median(b_[v])/max(np.median(a_[v]),1e-12):6.2f} | corr {rr:+.3f} {unite}", flush=True)

print(f"=== {REG} {DATE} : ETAGES DE LA COLONNE ===")
for j, nm in enumerate(["theta1", "theta2", "theta3"], start=1):
    hv = bv[f"THETA {j}"].values
    mv = getattr(diag, nm)[i].cpu().numpy()
    print(f"  {nm:22s} hydrotel(UHRH) med {np.median(hv):.4f} | meandre(troncon) med {np.median(mv):.4f} "
          f"| rapport {np.median(mv)/max(np.median(hv),1e-12):.2f}", flush=True)
print()
comp("apport lateral", ach["APPORT"].values, diag.q_lateral[i].cpu().numpy(), "(m3/s)")
comp("debit aval", ach["DEBIT AVAL"].values, Q[i].cpu().numpy(), "(m3/s)")
pd.DataFrame(dict(node=np.arange(n), apport_ht=ach["APPORT"].values,
                  apport_me=diag.q_lateral[i].cpu().numpy(),
                  q_ht=ach["DEBIT AVAL"].values, q_me=Q[i].cpu().numpy())
             ).to_csv(f"reports/apparier_hydrotel_{REG}.csv", index=False)
print()
print(f"-> reports/apparier_hydrotel_{REG}.csv")
