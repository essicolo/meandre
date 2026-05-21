"""Light-weight section-level timing instrumentation for the simulate loop.

Avoids torch.profiler (memory hog on Windows for our use case). Instead,
attaches forward hooks to the main modules and accumulates wall time per
section, synchronizing GPU between each.

Sections measured per timestep:
  - temporal_encoder.encode_sequence (one-shot at start)
  - vertical_column.forward (per timestep)
  - routing.forward (per timestep)
  - temperature module (per timestep, when active)

Output: prints total time per section over a 30-day simulation, plus the
mean per-timestep cost. Also runs a backward pass.
"""
from __future__ import annotations

import os
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import HydroModel
from meandre.routing.withdrawals import WithdrawalData
from meandre.utils.state import HydroState


CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".models/stfran/config/stfran.toml")
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main() -> None:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    paths = cfg["paths"]
    mcfg = cfg["model"]
    tcfg = cfg["temporal"]
    soil_z1 = cfg["soil"].get("z1", 0.30)
    soil_bounds = {k: cfg["soil"][k] for k in (
        "z2_min", "z2_max", "z3_min", "z3_max", "rain_hours_min", "rain_hours_max"
    ) if k in cfg["soil"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)

    cache = BasinCache(paths["basin_db"])
    hydro = cache.load(device=device)
    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
    print(f"Nodes  : {n_nodes}, Lakes: {int(graph.is_lake.sum())}", flush=True)

    forcing_full = extract_forcing(
        zarr_path=paths["weather_grid"],
        node_coords=node_coords,
        node_elev=None,
        date_start=tcfg["date_start"],
        date_end=tcfg["date_end"],
        cache_nc=paths["forcing_cache"],
        device=device,
    )
    forcing = forcing_full[:N_DAYS]
    doy = torch.arange(1, N_DAYS + 1, dtype=torch.long, device=device)
    withdrawals = WithdrawalData.zeros(N_DAYS, n_nodes, device=device)

    model = HydroModel(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=mcfg["n_forcing"],
        context_window=mcfg["context_window"],
        residual_history=mcfg["residual_history"],
        max_travel_time=mcfg["max_travel_days"],
        use_temporal=True,
        use_residual=False,
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=mcfg.get("dropout", 0.0),
        param_mode=mcfg.get("param_mode", "nerf"),
        soil_z1=soil_z1,
        soil_bounds=soil_bounds,
    ).to(device)
    model.spatial_encoder.init_from_literature()

    # ── Section accumulators ────────────────────────────────────────────
    totals = defaultdict(float)
    counts = defaultdict(int)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def make_hook(name: str):
        """Returns (pre_hook, post_hook) that time the wrapped module."""
        t_start: dict[str, float] = {}

        def pre(module, args):
            _sync()
            t_start[name] = time.perf_counter()

        def post(module, args, output):
            _sync()
            totals[name] += time.perf_counter() - t_start[name]
            counts[name] += 1

        return pre, post

    hooks_handles = []
    for mod_name, mod in [
        ("temporal_encoder", model.temporal_encoder),
        ("spatial_encoder", model.spatial_encoder),
        ("vertical_column", model.vertical_column),
        ("routing", model.routing),
        ("temperature", model.temperature),
    ]:
        if mod is None:
            continue
        pre, post = make_hook(mod_name)
        hooks_handles.append(mod.register_forward_pre_hook(pre))
        hooks_handles.append(mod.register_forward_hook(post))

    state0 = HydroState.zeros(n_nodes, device=device)

    # ── Forward + backward timing ──────────────────────────────────────
    print(f"\n--- Forward + backward on {N_DAYS} days ---", flush=True)
    _sync()
    t_full_start = time.perf_counter()

    Q_sim, _ = model.simulate(
        forcing=forcing,
        initial_state=state0,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
    )

    _sync()
    t_forward = time.perf_counter() - t_full_start

    t0 = time.perf_counter()
    loss = Q_sim.mean()
    loss.backward()
    _sync()
    t_backward = time.perf_counter() - t0

    t_full = t_forward + t_backward
    print(f"FORWARD   : {t_forward:7.2f}s", flush=True)
    print(f"BACKWARD  : {t_backward:7.2f}s", flush=True)
    print(f"TOTAL     : {t_full:7.2f}s ({t_full / N_DAYS:.2f}s/day)", flush=True)

    # ── Section breakdown (forward only — hooks don't intercept backward) ──
    print(f"\n--- Forward breakdown by module ---", flush=True)
    sum_sections = 0.0
    for name, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        n = counts[name]
        print(f"  {name:18s} {t:7.2f}s  ({n} calls, {t/max(n,1)*1000:6.1f} ms/call)",
              flush=True)
        sum_sections += t
    overhead = t_forward - sum_sections
    print(f"  {'(unaccounted)':18s} {overhead:7.2f}s", flush=True)


if __name__ == "__main__":
    main()
