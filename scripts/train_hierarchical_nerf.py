"""Training with hierarchical NeRF architecture."""
import argparse
import tomllib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState
from meandre.training.trainer import Trainer
from meandre.training.loss import CompositeKGELoss
from meandre.spatial.field_network import SpatialParams


class HierarchicalSpatialNetwork(nn.Module):
    """Hierarchical NeRF: Global + Local parameter networks.

    Separates parameters by spatial scale:
    - Global: Snow/temperature parameters (basin-wide)
    - Local: Soil/geology parameters (spatially varying)
    """

    def __init__(
        self,
        n_territorial: int = 17,
        n_coord_freqs: int = 4,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_territorial = n_territorial

        # Global parameters (same across entire basin)
        # Snow, temperature, and large-scale hydrology
        self.global_params = nn.Parameter(torch.randn(12) * 0.1)  # 12 global params

        # Local parameter network (spatially varying)
        # Soil properties, local hydrology
        coord_dim = 2 + 2 * n_coord_freqs * 2  # (lon, lat) + sin/cos frequencies
        in_dim = coord_dim + n_territorial

        self.coord_freqs = n_coord_freqs
        self.local_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 16),  # 16 local params
        )

    def positional_encoding(self, coords):
        """Fourier positional encoding for coordinates."""
        # coords: (n_nodes, 2) [lon, lat]
        encoded = [coords]  # Include raw coordinates

        for i in range(self.coord_freqs):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * torch.pi * coords))
            encoded.append(torch.cos(freq * torch.pi * coords))

        return torch.cat(encoded, dim=-1)

    def forward(self, coords, territorial):
        """
        Args:
            coords: (n_nodes, 2) [lon, lat] normalized
            territorial: (n_nodes, n_territorial)
        Returns:
            SpatialParams with hierarchical parameter structure
        """
        n_nodes = coords.shape[0]

        # Global parameters (broadcast to all nodes)
        global_expanded = self.global_params.unsqueeze(0).expand(n_nodes, -1)

        # Local parameters (spatially varying)
        coords_enc = self.positional_encoding(coords)
        local_input = torch.cat([coords_enc, territorial], dim=-1)
        local_params = self.local_net(local_input)

        # Combine global and local parameters
        all_params = torch.cat([global_expanded, local_params], dim=-1)

        return self._apply_constraints(all_params)

    def _apply_constraints(self, raw):
        """Apply physics constraints to raw network outputs."""
        # Global parameters (indices 0-11)
        # Snow and temperature parameters
        C_f = 2.0 + 8.0 * torch.sigmoid(raw[:, 0])  # [2, 10] melt factor
        T_melt = -2.0 + 4.0 * torch.tanh(raw[:, 1])  # [-2, 2] melt temp
        T_snow = 0.0 + 4.0 * torch.sigmoid(raw[:, 2])  # [0, 4] snow temp
        interception = 0.1 + 1.9 * torch.sigmoid(raw[:, 3])  # [0.1, 2.0] mm

        # Flow routing (global characteristics)
        manning_n = 0.01 + 0.09 * torch.sigmoid(raw[:, 4])  # [0.01, 0.1]
        f_wetland = 0.05 + 0.45 * torch.sigmoid(raw[:, 5])  # [0.05, 0.5]

        # Groundwater (regional parameters)
        k_gw = 0.001 + 0.049 * torch.sigmoid(raw[:, 6])  # [0.001, 0.05]
        T_gw = 3.0 + 10.0 * torch.sigmoid(raw[:, 7])  # [3, 13] °C
        K_atm = 0.05 + 0.5 * torch.sigmoid(raw[:, 8])  # [0.05, 0.55]

        # Regional flow characteristics
        slope_factor = 0.1 + 1.9 * torch.sigmoid(raw[:, 9])  # [0.1, 2.0]
        krec = 0.001 + 0.199 * torch.sigmoid(raw[:, 10])  # [0.001, 0.2]
        alpha_T = 0.01 + 0.14 * torch.sigmoid(raw[:, 11])  # [0.01, 0.15]

        # Local parameters (indices 12-27) - spatially varying
        # Soil hydraulics (most important for spatial variability)
        K_sat_1 = 0.1 + 2.9 * torch.sigmoid(raw[:, 12])  # [0.1, 3.0] surface
        K_sat_2 = 0.05 + 1.45 * torch.sigmoid(raw[:, 13])  # [0.05, 1.5] subsurface
        K_sat_3 = 0.01 + 0.49 * torch.sigmoid(raw[:, 14])  # [0.01, 0.5] deep

        porosity_1 = 0.2 + 0.4 * torch.sigmoid(raw[:, 15])  # [0.2, 0.6]
        porosity_2 = 0.2 + 0.4 * torch.sigmoid(raw[:, 16])  # [0.2, 0.6]
        porosity_3 = 0.2 + 0.4 * torch.sigmoid(raw[:, 17])  # [0.2, 0.6]

        theta_fc_1 = 0.1 + 0.4 * torch.sigmoid(raw[:, 18])  # [0.1, 0.5]
        theta_fc_2 = 0.1 + 0.4 * torch.sigmoid(raw[:, 19])  # [0.1, 0.5]
        theta_fc_3 = 0.1 + 0.4 * torch.sigmoid(raw[:, 20])  # [0.1, 0.5]

        theta_wp_1 = 0.05 + 0.25 * torch.sigmoid(raw[:, 21])  # [0.05, 0.3]
        theta_wp_2 = 0.05 + 0.25 * torch.sigmoid(raw[:, 22])  # [0.05, 0.3]
        theta_wp_3 = 0.05 + 0.25 * torch.sigmoid(raw[:, 23])  # [0.05, 0.3]

        # Local soil depths (spatially varying based on topography)
        depth_1 = 50.0 + 450.0 * torch.sigmoid(raw[:, 24])   # [50, 500] mm
        depth_2 = 100.0 + 400.0 * torch.sigmoid(raw[:, 25])  # [100, 500] mm
        depth_3 = 200.0 + 800.0 * torch.sigmoid(raw[:, 26])  # [200, 1000] mm

        # Local frost factor (elevation/aspect dependent)
        frost_alpha = 0.1 + 0.8 * torch.sigmoid(raw[:, 27])  # [0.1, 0.9]

        return SpatialParams(
            K_sat_1=K_sat_1, K_sat_2=K_sat_2, K_sat_3=K_sat_3,
            porosity_1=porosity_1, porosity_2=porosity_2, porosity_3=porosity_3,
            theta_fc_1=theta_fc_1, theta_fc_2=theta_fc_2, theta_fc_3=theta_fc_3,
            theta_wp_1=theta_wp_1, theta_wp_2=theta_wp_2, theta_wp_3=theta_wp_3,
            depth_1=depth_1, depth_2=depth_2, depth_3=depth_3,
            C_f=C_f, T_melt=T_melt, T_snow=T_snow,
            interception=interception, manning_n=manning_n,
            frost_alpha=frost_alpha, f_wetland=f_wetland,
            slope_factor=slope_factor, krec=krec, k_gw=k_gw,
            T_gw=T_gw, K_atm=K_atm, alpha_T=alpha_T,
        )


class HierarchicalYHydro(YHydro):
    """YHydro model with hierarchical spatial parameter network."""

    def __init__(self, **kwargs):
        # Remove spatial encoder args that don't apply to hierarchical version
        spatial_args = {k: v for k, v in kwargs.items()
                       if k in ['n_territorial', 'n_coord_freqs', 'hidden', 'dropout']}
        model_args = {k: v for k, v in kwargs.items()
                     if k not in spatial_args}
        model_args['param_mode'] = 'nerf'  # Force NeRF mode for hierarchical

        super().__init__(**model_args)

        # Replace spatial encoder with hierarchical version
        self.spatial_encoder = HierarchicalSpatialNetwork(**spatial_args)


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # 6-month training period
    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== HIERARCHICAL NERF TRAINING ===")
    print("Architecture: Global (12) + Local (16) = 28 parameters")
    print("Global: Snow, temperature, regional hydrology")
    print("Local: Soil properties, depths, local hydraulics")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Nodes: {n_nodes}")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/hierarchical_forcing.nc"),
        device=device,
    )

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=100,
    )
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)

    print(f"Stations: {n_stations}")

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

    # Create hierarchical model
    model = HierarchicalYHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,
        use_residual=False,
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=0.1,  # Some dropout for regularization
        n_coord_freqs=4,  # Moderate spatial resolution
        hidden=128,   # Reasonable capacity
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    spatial_params = sum(p.numel() for p in model.spatial_encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Spatial network parameters: {spatial_params:,}")

    # Test hierarchical structure
    print("\n=== HIERARCHICAL STRUCTURE TEST ===")
    with torch.no_grad():
        # Get territorial data as tensor for testing
        territorial_tensor = territorial.to_tensor()[:5]  # First 5 nodes
        params = model.spatial_encoder(node_coords[:5], territorial_tensor)

        print("Global parameter ranges (same for all nodes):")
        print(f"C_f: {params.C_f[:5].min():.2f} - {params.C_f[:5].max():.2f}")
        print(f"T_gw: {params.T_gw[:5].min():.1f} - {params.T_gw[:5].max():.1f}")

        print("Local parameter ranges (spatially varying):")
        print(f"K_sat_1: {params.K_sat_1[:5].min():.3f} - {params.K_sat_1[:5].max():.3f}")
        print(f"depth_1: {params.depth_1[:5].min():.1f} - {params.depth_1[:5].max():.1f}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15, verbose=True)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="hierarchical_nerf",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== TRAINING HIERARCHICAL MODEL ===")
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=200,
        print_every=5,
    )

    print("\n=== TRAINING COMPLETE ===")
    print("Hierarchical NeRF separates global vs. local parameters")
    print("This should provide better spatial representation while maintaining")
    print("physical coherence across the basin.")


if __name__ == "__main__":
    main()