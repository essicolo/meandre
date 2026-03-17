"""Simple physics training with good parameter initialization."""
import tomllib
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.metrics import kge as compute_kge
from meandre.training.trainer import Trainer
from meandre.training.loss import CompositeKGELoss


def good_physics_init(model):
    """Initialize with known good physics parameters."""
    with torch.no_grad():
        # Good physics values that should work
        params = torch.tensor([
            # K_sat (m/day) - good values for flow
            0.6, 0.25, 0.08,

            # Porosity - typical soil values
            0.44, 0.44, 0.44,

            # Field capacity - good retention
            0.28, 0.28, 0.28,

            # Wilting point - reasonable
            0.14, 0.14, 0.14,

            # Soil depths (mm) - good storage
            220.0, 350.0, 700.0,

            # Snow parameters - moderate
            5.5, 0.0, 1.8,

            # Surface parameters
            0.9, 0.032, 0.65,

            # Flow parameters
            0.12, 1.1, 0.018,

            # Groundwater
            0.006,

            # Temperature
            9.5, 0.25, 0.06,
        ], device=model.spatial_encoder.static_params.device)

        model.spatial_encoder.static_params.data = params

        print("=== GOOD PHYSICS INITIALIZATION ===")
        print(f"K_sat: {params[0:3].cpu().numpy()} m/day")
        print(f"Soil depths: {params[12:15].cpu().numpy()} mm")
        print(f"Snow melt: {params[15]:.1f} mm/°C/day")


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== SIMPLE PHYSICS TRAINING ===")
    print("Strategy: Good initialization + simple model")

    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Basin: {n_nodes} nodes")

    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/physics_simple_forcing.nc"),
        device=device,
    )

    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=100,
    )
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)

    print(f"Observations: {n_stations} stations")

    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True

    q_obs_tensor = torch.from_numpy(obs["discharge"][:, station_indices]).to(device)
    doy = torch.tensor([i % 365 + 1 for i in range(len(forcing))], dtype=torch.long, device=device)

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Simple but effective model
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=25,
        residual_history=10,
        max_travel_time=15,
        use_temporal=False,
        use_residual=False,
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=0.05,
        param_mode="static",
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply good physics initialization
    good_physics_init(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=8e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=12)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="physics_simple",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== TRAINING SIMPLE PHYSICS MODEL ===")
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=350,
        print_every=15,
    )

    print("\n=== SIMPLE PHYSICS TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()