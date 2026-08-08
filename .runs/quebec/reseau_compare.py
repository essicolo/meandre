"""TEST DE L'ENSEMBLE (demande d'Essi) : méandre vs Hydrotel sur TOUS les tronçons,
pas seulement aux jauges. `<projet>/simulation/simulation/resultat/debit_aval.nc` porte le
débit aval d'Hydrotel pour chaque tronçon, 2020-2026, sur EXACTEMENT le même maillage que
méandre (idtroncon = node_id). On compare les séries nœud par nœud sur 2022-2024 et on
regarde COMMENT l'écart se structure : déjà présent en tête de bassin -> génération ;
croissant vers l'aval -> accumulation/routage ; localisé -> caractéristique de territoire.
Une erreur systémique (aire, unité, accumulation) laisse une signature géométrique
évidente sur 3412 points, invisible sur 16 jauges agrégées.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/reseau_compare.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch, xarray as xr
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"

dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values)
mht = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mht]            # (T, n) dans l'ordre idtroncon 1..n
ids = dht["idtroncon"].values
dht.close()
print(f"[hydrotel] {QH.shape[1]} troncons x {QH.shape[0]} jours ({T0} -> {T1})", flush=True)

ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = np.asarray(r["node_ids"])
assert QH.shape[1] == n, f"{QH.shape[1]} vs {n}"
# reordonner Hydrotel dans l'ordre des noeuds meandre : colonne j de QH = idtroncon j+1
# correspondance EXPLICITE colonne Hydrotel -> noeud meandre (node_ids = id troncon)
pos = {int(i): j for j, i in enumerate(ids)}
QH = QH[:, [pos[int(i)] for i in node_ids]]
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
m.vertical_column.compile_column = False
with torch.no_grad():
    Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                      graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                      withdrawals=td.withdrawals, day_of_year=td.day_of_year)
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
mme = np.asarray((tt >= T0) & (tt <= T1))
QM = Q[torch.tensor(mme, device=DEVICE)].cpu().numpy()
del Q; torch.cuda.empty_cache()
nT = min(QM.shape[0], QH.shape[0])
QM, QH = QM[:nT], QH[:nT]
print(f"[meandre] {QM.shape[1]} noeuds x {QM.shape[0]} jours", flush=True)

A = r["territorial"].get_physical("area_km2_local").cpu().numpy()
# aire drainee cumulee par noeud (topologie)
import duckdb
con = duckdb.connect(f"D:/meandre-data/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
Acum = A.copy()
import collections
enfants = collections.defaultdict(list)
deg = np.zeros(n, int)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enfants[int(s_)].append(int(d_)); deg[int(d_)] += 1
from meandre.routing.graph import RiverGraph
topo = td.graph.topo_order.cpu().numpy()
for u in topo:
    for v in enfants.get(int(u), []):
        Acum[v] += Acum[u]

lac = td.graph.is_lake.bool().cpu().numpy()
# lissages : si r remonte fortement en hebdomadaire, la decorrelation quotidienne vient
# du FORCAGE (pluie datee differemment) ; si elle reste basse, elle est STRUCTURELLE.
def lisse(X, w):
    c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
    return (c[w:] - c[:-w]) / w
QH7, QM7 = lisse(QH, 7), lisse(QM, 7)
QH30, QM30 = lisse(QH, 30), lisse(QM, 30)
# saisons pour le rapport de volume
mois = pd.DatetimeIndex(pd.date_range(T0, periods=QH.shape[0], freq="D")).month
ete = np.isin(mois, [6, 7, 8, 9]); hiver = np.isin(mois, [1, 2, 3])
res = []
for j in range(n):
    h, mn = QH[:, j], QM[:, j]
    v = np.isfinite(h) & np.isfinite(mn)
    if v.sum() < 300 or h[v].std() < 1e-9: continue
    rr = float(np.corrcoef(h[v], mn[v])[0, 1])
    r7 = float(np.corrcoef(QH7[:, j], QM7[:, j])[0, 1])
    r30 = float(np.corrcoef(QH30[:, j], QM30[:, j])[0, 1])
    beta = float(mn[v].mean() / max(h[v].mean(), 1e-9))
    be = float(mn[ete].mean() / max(h[ete].mean(), 1e-9))
    bh = float(mn[hiver].mean() / max(h[hiver].mean(), 1e-9))
    res.append((j, Acum[j], bool(lac[j]), rr, r7, r30, beta, be, bh))
d = pd.DataFrame(res, columns=["node", "aire_cum", "lac", "r", "r7", "r30", "beta", "beta_ete", "beta_hiver"])
d.to_csv(f"reports/reseau_{REG}.csv", index=False)
print(f"\n=== {REG} : meandre vs Hydrotel sur {len(d)} troncons (2022-2024) ===")
print(f"correlation mediane {d.r.median():.3f} | rapport de volume median {d.beta.median():.3f}")
d["classe"] = pd.cut(d.aire_cum, [0, 10, 50, 200, 1000, 5000, 1e9],
                     labels=["<10", "10-50", "50-200", "200-1k", "1k-5k", ">5k"])
g = d.groupby("classe", observed=True).agg(n=("node", "size"), r_med=("r", "median"),
                                           r7=("r7", "median"), r30=("r30", "median"),
                                           beta_med=("beta", "median"),
                                           b_ete=("beta_ete", "median"),
                                           b_hiver=("beta_hiver", "median")).round(3)
print("\npar aire drainee cumulee (km2) :"); print(g.to_string())
g2 = d.groupby("lac").agg(n=("node", "size"), r_med=("r", "median"), r7=("r7", "median"),
                          r30=("r30", "median"), beta_med=("beta", "median"),
                          b_ete=("beta_ete", "median")).round(3)
print("\nlac vs riviere :"); print(g2.to_string())
