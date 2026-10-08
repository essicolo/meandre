"""Le modele a-t-il ABSORBE les prelevements dans ses parametres ?

Enjeu. Pour qu'une renaturalisation ait un sens, les parametres doivent representer le
systeme NATUREL. Si le calage s'est fait sans les prelevements, le modele a pu les
absorber : une conductivite ou une recharge compensant une prise d'eau. Mettre les
prelevements a zero ne renaturalise alors rien.

Test. On correle chaque parametre spatial appris avec l'intensite des prelevements EN
AMONT du noeud. Une correlation nulle = pas d'absorption detectable.

PIEGE, et c'est le coeur du test : les prelevements sont la ou il y a des gens, donc
correles a l'urbanisation, qui est une VRAIE propriete physique que le champ spatial
doit representer. Une correlation brute ne prouve donc rien. On calcule aussi la
correlation PARTIELLE en controlant f_urban et f_agriculture : c'est elle qui compte.

  python .runs/quebec/test_absorption.py outv .runs/quebec/checkpoints/best-outv-etl-aq30.pt
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".runs/quebec")

from collections import defaultdict, deque

import duckdb
import numpy as np
import pandas as pd
import torch

from meandre.utils import paths as _mpaths

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKPT = sys.argv[2] if len(sys.argv) > 2 else ".runs/quebec/checkpoints/best-outv-etl-aq30.pt"

db = _mpaths.data_path("quebec", f"{REG}.duckdb")
con = duckdb.connect(db, read_only=True)
edges = con.execute("SELECT src, dst FROM edges").df()
n_nodes = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
w = con.execute("""SELECT node_idx, SUM(ABS(net_surface)) + SUM(ABS(net_gw)) v,
                          COUNT(DISTINCT date) nd FROM withdrawals GROUP BY node_idx""").df()
con.close()

# intensite CUMULEE en amont de chaque noeud
amont = defaultdict(list)
for s_, d_ in zip(edges.src.values, edges.dst.values):
    amont[int(d_)].append(int(s_))
poids = np.zeros(n_nodes)
if len(w):
    nd = max(int(w.nd.max()), 1)
    for k, v in zip(w.node_idx.astype(int), w.v.astype(float)):
        poids[k] = v / nd
cumul = np.zeros(n_nodes)
for i in range(n_nodes):
    q, vu = deque([i]), set()
    while q:
        x = q.popleft()
        if x in vu:
            continue
        vu.add(x)
        q.extend(amont.get(x, ()))
    cumul[i] = poids[list(vu)].sum()
print(f"[absorption] {REG} : {n_nodes} noeuds | {(cumul > 0).sum()} avec des prelevements "
      f"en amont | intensite mediane {np.median(cumul[cumul > 0]) if (cumul > 0).any() else 0:.4f} m3/s")

# PARAMETRES PAR NOEUD : le point de reprise ne stocke que les poids du perceptron,
# le champ spatial doit etre FAIT TOURNER pour rendre ses 38 sorties par noeud.
import tomllib

from joint_data import load_region
from meandre.model import HydroModel

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device="cpu")
td = r["train_data"]
m = HydroModel(n_nodes=r["n_nodes"], n_territorial=r["territorial"].n_features, n_forcing=6,
               use_temporal=False, use_residual=False, use_travel_time_attn=False,
               use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
               column_mode="hydrotel", et_mode="linacre", use_temperature=False,
               use_latent_codes=False, latent_mode="additive", spatial_melt=True,
               routing_mode="operator-lagged", predict_lake_params=True,
               compile_soil=False, use_aquifer=True)
m.eval()
m.spatial_encoder.init_from_literature({})
m.load(CKPT)
print(f"[absorption] {os.path.basename(CKPT)}")
with torch.no_grad():
    sp = m.spatial_encoder(td.node_coords, r["territorial"].to_tensor())
import dataclasses
champs = {f.name: getattr(sp, f.name).detach().cpu().numpy().astype(float)
          for f in dataclasses.fields(sp)
          if torch.is_tensor(getattr(sp, f.name))
          and getattr(sp, f.name).ndim == 1
          and getattr(sp, f.name).shape[0] == n_nodes}
print(f"[absorption] {len(champs)} parametres par noeud extraits du champ")

# territorial brut, pour le controle
rw = pd.read_parquet(_mpaths.data_path("quebec", "territorial-raw-QC.parquet"))
rw = rw[rw.region == REG]
if len(rw) != n_nodes:
    print(f"[absorption] territorial indisponible ({len(rw)} vs {n_nodes})")
    sys.exit(0)

def partielle(x, y, z):
    """Correlation partielle de x et y en controlant les colonnes de z."""
    Z = np.column_stack([np.ones(len(x))] + [z[:, j] for j in range(z.shape[1])])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ bx, y - Z @ by
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

ctrl = np.column_stack([rw["f_urban"].values, rw["f_agriculture"].values,
                        rw["f_forest"].values, np.log1p(rw["drainage_area_km2"].values)])
ok = np.isfinite(cumul) & np.isfinite(ctrl).all(axis=1)
xl = np.log1p(cumul)

print(f"\n{'parametre (brut du champ)':32s} {'r brut':>8s} {'r partiel':>10s}")
lignes = []
for k, y in champs.items():
    if not np.isfinite(y).all() or y.std() < 1e-12:
        continue
    rb = float(np.corrcoef(xl[ok], y[ok])[0, 1])
    rp = partielle(xl[ok], y[ok], ctrl[ok])
    lignes.append((abs(rp) if np.isfinite(rp) else 0.0, k, rb, rp))
if not lignes:
    print("  (aucun parametre exploitable)")
else:
    for _, k, rb, rp in sorted(lignes, reverse=True)[:20]:
        print(f"{k[:32]:32s} {rb:+8.3f} {rp:+10.3f}")
print("\nlecture : r PARTIEL controle f_urban, f_agriculture, f_forest et l'aire drainee.")
print("Proche de zero = pas d'absorption detectable. Fort = le parametre suit les")
print("prelevements au-dela de ce que l'occupation du sol explique.")
