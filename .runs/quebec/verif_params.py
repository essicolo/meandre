"""Les paramètres s'ajustent-ils REELLEMENT pendant l'entraînement ? (exigence Essi)
Compare un checkpoint à l'initialisation littérature : dispersion spatiale de chaque
paramètre, proximité des bornes, et déplacement de la tête de lac. Un paramètre dont le
coefficient de variation reste au niveau de l'init n'a jamais appris ; un paramètre collé
à sa borne ne peut plus s'ajuster.

  PYTHONIOENCODING=utf-8 python .runs/quebec/verif_params.py outv best-outv-etl-airelac
"""
import os, sys, dataclasses
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import torch, numpy as np
from meandre.model import HydroModel
from meandre.data.basin_cache import BasinCache

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKS = sys.argv[2:] or [f"best-{REG}-etl-qc"]
db = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"D:/meandre-data/quebec/{REG}.duckdb"
h = BasinCache(db).load(device="cpu"); n = h["n_nodes"]; terr = h["territorial"]
lac = h["graph"].is_lake.bool()

def champ(ck):
    m = HydroModel(n_nodes=n, n_territorial=terr.n_features, n_forcing=6, use_temporal=False,
        use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
        column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel",
        et_mode="mcguinness", use_temperature=False, use_latent_codes=True,
        latent_mode="additive", spatial_melt=True, routing_mode="operator-lagged",
        predict_lake_params=True, compile_soil=False, use_aquifer=True)
    if ck: m.load(f".runs/quebec/checkpoints/{ck}.pt")
    m.eval()
    with torch.no_grad():
        sp = m.spatial_encoder(h["node_coords"], terr.data)
        k, b = m.spatial_encoder.lake_params(h["node_coords"], terr.data)
    return sp, k, b

sp0, k0, b0 = champ(None)      # init litterature, jamais entrainee
noms = [f.name for f in dataclasses.fields(sp0)]
for ck in CKS:
    try: sp, k, b = champ(ck)
    except Exception as e:
        print(f"{ck}: {type(e).__name__}"); continue
    X0 = sp0.to_tensor().double().numpy(); X = sp.to_tensor().double().numpy()
    cv0 = X0.std(0) / (np.abs(X0.mean(0)) + 1e-12)
    cv = X.std(0) / (np.abs(X.mean(0)) + 1e-12)
    fige = [noms[i] for i in range(len(noms)) if cv[i] < 3 * max(cv0[i], 1e-6) and cv[i] < 0.02]
    print(f"\n=== {ck} ===")
    print(f"parametres encore FIGES ({len(fige)}/{len(noms)}) : {', '.join(fige) if fige else 'aucun'}")
    kk = k[lac].double().numpy(); bb = b[lac].double().numpy()
    plancher = float((kk < 1.5e-6).mean() * 100); plafond = float((kk > 6.7e-3).mean() * 100)
    print(f"tete de lac : k med {np.median(kk):.3e} | etendue x{kk.max()/max(kk.min(),1e-30):.1f} | "
          f"beta med {np.median(bb):.3f} | {plancher:.1f} % au plancher, {plafond:.1f} % au plafond")
