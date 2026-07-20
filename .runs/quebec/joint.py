"""Entraînement CONJOINT multi-régions : UN NeRF + UNE colonne pour N régions.
Design : reports/design_conjoint.md. Pilote 3 régions (slso, mont, gasp) par défaut.

  python .runs/quebec/joint.py slso mont gasp
  JOINT_EPOCHS=30 JOINT_TAG=pilote3 python .runs/quebec/joint.py slso mont gasp

Orchestration : un HydroModel partagé (latents z_n dimensionnés au total des nœuds,
tranche par région via spatial_encoder.latent_offset) ; un Trainer par région lié au
même modèle et au même optimiseur ; rotation des régions dans l'epoch (ordre mélangé),
_train_epoch régional = un pas d'optimisation par région ; val par région chaque epoch ;
checkpoint sur la médiane pondérée par jauges. Held-out 2022-24 par région en fin de run.
"""
import os, sys, random
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tomllib
import numpy as np
import torch
from meandre.model import HydroModel
from meandre.training.trainer import Trainer, TrainingConfig
from meandre.training.scheduler import build_scheduler
from meandre.utils.metrics import kge as kge_fn
from meandre.utils.state import HydroState
from joint_data import load_region

REGIONS = [a.lower() for a in sys.argv[1:]] or ["slso", "mont", "gasp"]
TAG = os.environ.get("JOINT_TAG", "pilote3")
N_EPOCHS = int(os.environ.get("JOINT_EPOCHS", "30"))
LR_OVERRIDE = float(os.environ.get("JOINT_LR", "0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CFG = ".runs/quebec/config/gasp-v4.toml"   # source des poids loss + hyperparams
CKPT = f".runs/quebec/checkpoints/best-joint-{TAG}.pt"

cfg = tomllib.load(open(BASE_CFG, "rb"))
lcfg = cfg["loss"]; tcfg = cfg["training"]; mcfg = cfg["model"]

# ── données par région ───────────────────────────────────────────────────────
print(f"[joint] régions : {REGIONS} | device {DEVICE}")
regions = [load_region(r, lcfg, device=DEVICE) for r in REGIONS]
offsets = np.cumsum([0] + [r["n_nodes"] for r in regions])
n_total = int(offsets[-1])
n_gauges = {r["name"]: r["n_gauges"] for r in regions}
w_reg = {r["name"]: max(r["n_gauges"], 1) for r in regions}
print(f"[joint] {n_total} nœuds au total | jauges {n_gauges}")

# ── modèle partagé ───────────────────────────────────────────────────────────
model = HydroModel(
    n_nodes=n_total,
    n_territorial=regions[0]["territorial"].n_features,
    n_forcing=6,
    use_temporal=False,
    use_residual=False,
    use_travel_time_attn=False,
    use_frost_rankinen=bool(mcfg.get("use_frost_rankinen", True)),
    column_theta_init_frac=float(mcfg.get("column_theta_init_frac", 0.9)),
    param_mode="nerf",
    column_mode="hydrotel",
    et_mode="mcguinness",
    use_temperature=False,
    use_latent_codes=bool(mcfg.get("use_latent_codes", True)),
    latent_mode="additive",
    spatial_melt=bool(mcfg.get("spatial_melt", True)),
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=bool(mcfg.get("predict_lake_params", True)),
    compile_soil=bool(mcfg.get("compile_soil", True)),
).to(DEVICE)
model.spatial_encoder.init_from_literature(cfg.get("literature_prior"))
print(f"[joint] modèle partagé : {sum(p.numel() for p in model.parameters()):,} params")

# ── trainers régionaux (modèle + optimiseur partagés) ───────────────────────
tconf = TrainingConfig(
    n_epochs=N_EPOCHS,
    lr=LR_OVERRIDE if LR_OVERRIDE > 0 else float(tcfg.get("lr", 5e-4)),
    chunk_steps=int(tcfg.get("chunk_steps", 180)),
    tbptt_steps=int(tcfg.get("tbptt_steps", 365)),
    grad_clip=float(tcfg.get("clip_grad_norm", 1.0)),
    w_prior=float(tcfg.get("w_prior", 0.005)),
)
trainers = []
shared_opt = None
for i, r in enumerate(regions):
    tr = Trainer(model=model, loss_fn=r["loss_fn"], train_data=r["train_data"],
                 val_data=r["val_data"], config=tconf, run_name=f"joint-{TAG}-{r['name']}",
                 checkpoint_path=None, optimizer=shared_opt)
    if shared_opt is None:
        shared_opt = tr.optimizer   # le premier crée l'optimiseur discriminatif, les autres le partagent
    trainers.append(tr)
scheduler = build_scheduler(shared_opt, N_EPOCHS, warmup_epochs=int(tcfg.get("warmup_epochs", 3)))

def set_region(i):
    model.spatial_encoder.latent_offset = int(offsets[i])
    model.n_nodes = int(regions[i]["n_nodes"])   # allocations internes de simulate

# ── boucle conjointe ─────────────────────────────────────────────────────────
os.makedirs(".runs/quebec/checkpoints", exist_ok=True)
best_agg = -9e9
rollbacks = 0
rng = random.Random(0)
for epoch in range(N_EPOCHS):
    order = list(range(len(regions))); rng.shuffle(order)
    trainers[0]._apply_curriculum(epoch)
    for i in order:
        set_region(i)
        trainers[i]._cur_epoch = epoch
        loss, comps = trainers[i]._train_epoch()
    scheduler.step()
    # validation par région
    meds, parts = [], []
    for i, r in enumerate(regions):
        set_region(i)
        m = trainers[i]._val_epoch()
        med = m.get("val_kge_median", m.get("kge_median", float("nan")))
        meds.append((r["name"], med, w_reg[r["name"]]))
        parts.append(f"{r['name']} {med:.3f}")
    agg = float(np.average([m for _, m, _ in meds], weights=[w for _, _, w in meds]))
    marker = ""
    if agg > best_agg + 1e-4:
        best_agg = agg
        model.save(CKPT)
        marker = " -> best"
    # GARDE-FOU divergence (leçon pilote3 : effondrement epoch 7 sans retour) :
    # régression > 20% sous le best -> recharge best + LR/2, max 3 fois.
    elif best_agg > 0 and agg < 0.8 * best_agg and rollbacks < 3 and os.path.exists(CKPT):
        rollbacks += 1
        model.load(CKPT)
        for g in shared_opt.param_groups:
            g["lr"] *= 0.5
        marker = f" -> ROLLBACK {rollbacks}/3 (LR/2)"
    print(f"[joint] epoch {epoch:3d} | agrégé {agg:.4f} | " + " | ".join(parts) + marker, flush=True)

# ── held-out 2022-2024 par région (best checkpoint) ─────────────────────────
print(f"\n[joint] HELD-OUT 2022-2024 (best agrégé {best_agg:.4f})")
model.load(CKPT); model.eval()
for i, r in enumerate(regions):
    set_region(i)
    td = r["train_data"]
    with torch.no_grad():
        Q, _ = model.simulate(forcing=td.forcing,
                              initial_state=HydroState.zeros(r["n_nodes"], device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords,
                              territorial=td.territorial, withdrawals=td.withdrawals,
                              day_of_year=td.day_of_year)
    times = r["times"]
    sl = (times >= "2022-01-01") & (times <= "2024-12-31")
    slt = torch.tensor(sl.values if hasattr(sl, "values") else sl, device=DEVICE)
    Qs = Q[slt][:, td.station_idx].cpu()
    qo = td.q_obs[td.train_slice.start:][...]  # q_obs commence à train_slice.start
    # reconstruire l'index absolu des obs : q_obs de mk() démarre à sl_.start du train
    q_full = td.q_obs  # (T - train_start, n_st) aligné sur times[train_start:]
    t0 = td.train_slice.start
    qo_test = q_full[np.flatnonzero(sl)[0] - t0 : np.flatnonzero(sl)[-1] - t0 + 1].cpu()
    ks = []
    for s in range(Qs.shape[1]):
        v = ~torch.isnan(qo_test[:, s]) & ~torch.isnan(Qs[:, s])
        if v.sum() < 60: continue
        ks.append(float(kge_fn(qo_test[v, s], Qs[v, s])))
    ks = np.array(ks)
    print(f"  {r['name']}: n={len(ks)} | médian {np.median(ks):.4f} | pooled ~ | mean {ks.mean():.4f}")
print("[joint] DONE")
