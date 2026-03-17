"""Training with HEAVY DIAGNOSTICS to identify physics problems."""
import argparse
import logging
import os
import tomllib

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import torch
import xarray as xr
import pandas as pd

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.training.loss import differentiable_composite_kge_loss
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState


def heavy_diagnostics(model, territorial, node_coords, forcing, doy, device, epoch):
    """Heavy diagnostics to identify physics problems."""
    print(f"\n=== HEAVY DIAGNOSTICS (Epoch {epoch}) ===")

    # 1. SPATIAL PARAMETER DIAGNOSTICS
    coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)
    sample_indices = torch.arange(min(20, len(node_coords)), device=device)
    sample_coords = coords_norm[sample_indices]

    # Get territorial features (handle different access patterns)
    if hasattr(territorial, 'features'):
        sample_features = territorial.features[sample_indices]
    elif hasattr(territorial, 'data'):
        sample_features = territorial.data[sample_indices]
    else:
        print("ERROR: Cannot access territorial features for diagnostics")
        return

    with torch.no_grad():
        spatial_params = model.spatial_encoder(sample_coords, sample_features)

    # Check parameter ranges and constraints
    param_names = [
        "K_sat_1", "K_sat_2", "K_sat_3",
        "theta_fc_1", "theta_fc_2", "theta_fc_3",
        "theta_wp_1", "theta_wp_2", "theta_wp_3",
        "depth_1", "depth_2", "depth_3",
        "C_f", "T_melt", "T_snow", "k_gw", "f_wetland"
    ]

    print("SPATIAL PARAMETER ANALYSIS:")
    for i, name in enumerate(param_names):
        if i < spatial_params.shape[1]:
            param_vals = spatial_params[:, i].cpu()
            print(f"  {name:12s}: {param_vals.min():.4f} to {param_vals.max():.4f} (μ={param_vals.mean():.4f}, σ={param_vals.std():.4f})")

            # Check for critical physics violations
            if name.startswith("K_sat"):
                if param_vals.max() < 0.01:
                    print(f"    ❌ CRITICAL: {name} extremely low - blocking water movement!")
                elif param_vals.min() > 100:
                    print(f"    ❌ CRITICAL: {name} extremely high - unrealistic!")

            elif name.startswith("theta_fc"):
                if param_vals.max() > 0.7:
                    print(f"    ❌ CRITICAL: {name} > 0.7 - impossible field capacity!")
                elif param_vals.min() < 0.05:
                    print(f"    ❌ CRITICAL: {name} < 0.05 - no water storage!")

            elif name.startswith("theta_wp"):
                if param_vals.max() > 0.6:
                    print(f"    ❌ CRITICAL: {name} > 0.6 - wilting point too high!")

            elif name.startswith("depth"):
                if param_vals.max() < 50:
                    print(f"    ❌ CRITICAL: {name} < 50mm - soil layer too thin!")
                elif param_vals.min() > 10000:
                    print(f"    ❌ CRITICAL: {name} > 10m - soil layer too thick!")

    # 2. WATER BALANCE DIAGNOSTICS
    print("\nWATER BALANCE ANALYSIS:")

    # Run single timestep with diagnostics
    from meandre.vertical.column import VerticalColumn
    from meandre.spatial.field_network import SpatialParams

    column = VerticalColumn()
    test_nodes = slice(0, 5)  # First 5 nodes

    try:
        test_params = SpatialParams.from_tensor(spatial_params[test_nodes])
        test_forcing = forcing[:1, sample_indices[test_nodes]]  # Just first day
        test_state = HydroState.default_warm(len(test_nodes), device=device)

        output = column(
            test_forcing,
            test_state,
            test_params,
            doy[:1],
            return_diagnostics=True
        )

        lateral_inflow = output.lateral_inflow[0]  # mm/day
        diag = output.diag

        # Input analysis
        precip = test_forcing[0, :, 0]  # Precipitation mm/day
        print(f"  Input precipitation: {precip.mean():.2f} mm/day (range: {precip.min():.1f}-{precip.max():.1f})")

        # Output analysis
        print(f"  Lateral inflow: {lateral_inflow.mean():.4f} mm/day (range: {lateral_inflow.min():.4f}-{lateral_inflow.max():.4f})")

        if diag:
            etr = diag.get('etr', torch.zeros_like(lateral_inflow))[0]
            snowmelt = diag.get('snowmelt', torch.zeros_like(lateral_inflow))[0]
            recharge = diag.get('recharge', torch.zeros_like(lateral_inflow))[0]

            print(f"  ETR: {etr.mean():.3f} mm/day")
            print(f"  Snowmelt: {snowmelt.mean():.3f} mm/day")
            print(f"  Recharge: {recharge.mean():.3f} mm/day")

            # Mass balance check
            water_in = precip.mean() + snowmelt.mean()
            water_out = etr.mean() + lateral_inflow.mean() + recharge.mean()
            balance_error = water_in - water_out

            print(f"  Mass balance: IN={water_in:.3f} - OUT={water_out:.3f} = ERROR={balance_error:.4f} mm/day")

            if abs(balance_error) > 0.5:
                print(f"    ❌ CRITICAL: Mass balance violation > 0.5 mm/day!")
            else:
                print(f"    ✅ Mass balance OK")

        # Physics reasonableness check
        runoff_ratio = lateral_inflow.mean() / (precip.mean() + 1e-6)
        print(f"  Runoff ratio: {runoff_ratio:.4f} (expect 0.05-0.3 for winter)")

        if lateral_inflow.mean() < 0.001:
            print(f"    ❌ CRITICAL: Lateral inflow nearly zero - sealed soil system!")
        elif lateral_inflow.mean() > 50:
            print(f"    ❌ CRITICAL: Lateral inflow too high - unrealistic runoff!")
        else:
            print(f"    ✅ Lateral inflow magnitude reasonable")

    except Exception as e:
        print(f"ERROR in water balance diagnostics: {e}")

    # 3. GRADIENT DIAGNOSTICS
    print("\nGRADIENT FLOW ANALYSIS:")

    # Check if spatial encoder gradients are flowing
    spatial_grads = []
    for name, param in model.spatial_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            spatial_grads.append(grad_norm)
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"  {name}: NO GRADIENT")

    if spatial_grads:
        mean_grad = np.mean(spatial_grads)
        print(f"  Mean spatial grad norm: {mean_grad:.6f}")
        if mean_grad < 1e-8:
            print("    ❌ CRITICAL: Vanishing gradients in spatial encoder!")
        elif mean_grad > 1e2:
            print("    ❌ CRITICAL: Exploding gradients in spatial encoder!")
        else:
            print("    ✅ Gradient flow OK")
    else:
        print("    ❌ CRITICAL: No gradients computed!")


def main():
    parser = argparse.ArgumentParser(description="Heavy diagnostics training")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])
    CHECKPOINT = Path("notebooks/slso/checkpoints/heavy_diagnostics.pt")

    # Short training for diagnostics focus
    DATE_START = "2001-01-01"
    DATE_END = "2002-06-30"
    TRAIN_END = "2002-03-31"
    VAL_START = "2002-04-01"
    EPOCHS = 20  # Just focus on diagnosing first few epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== HEAVY DIAGNOSTICS TRAINING ===")
    print("Focus: Identify exactly WHY physics is broken")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
    print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/diagnostics_forcing.nc"),
        device=device,
    )

    ds_time = xr.open_dataset(Path("/tmp/diagnostics_forcing.nc"))
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=50,
    )
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)

    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True

    discharge_np = obs["discharge"]
    q_obs_tensor = torch.from_numpy(discharge_np[:, station_indices]).to(device)
    print(f"Stations: {n_stations}")

    # Date handling
    def dates_to_slice(dates, start, end):
        days = dates.astype("datetime64[D]")
        s = int(np.searchsorted(days, np.datetime64(start, "D")))
        e = int(np.searchsorted(days, np.datetime64(end, "D"), side="right"))
        return slice(s, e)

    train_sl = dates_to_slice(all_dates, DATE_START, TRAIN_END)
    val_sl = dates_to_slice(all_dates, VAL_START, DATE_END)

    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    # Simple model for diagnostics
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,  # Disable for simpler debugging
        use_residual=False,  # Disable for simpler debugging
        use_travel_time_attn=False,  # Disable for simpler debugging
        use_temperature=True,
        dropout=0.0,  # No dropout for clearer diagnostics
        param_mode="nerf",  # Use nerf for spatial smoothness
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Simple optimizer for diagnostics
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Heavy Diagnostics Training (Focus on Problem Identification) ===")

    CHUNK_SIZE = 30  # Small chunks
    WARMUP = 15

    # INITIAL DIAGNOSTICS (before any training)
    print("\n" + "="*80)
    print("INITIAL STATE DIAGNOSTICS (Epoch -1)")
    print("="*80)
    heavy_diagnostics(model, territorial, node_coords, forcing, doy, device, -1)

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        epoch_kges = []

        # Use warm initialization
        state = HydroState.default_warm(n_nodes, device=device)

        # Single chunk for simplicity in diagnostics
        train_days = train_sl.stop - train_sl.start

        # Use just first chunk for detailed analysis
        start_idx = train_sl.start
        end_idx = min(start_idx + CHUNK_SIZE + WARMUP, train_sl.stop)

        chunk_forcing = forcing[start_idx:end_idx]
        chunk_doy = doy[start_idx:end_idx]

        # Forward pass
        Q_sim, state = model.simulate(
            forcing=chunk_forcing,
            initial_state=state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=chunk_doy,
        )

        # Use non-warmup period for loss
        Q_sim_train = Q_sim[WARMUP:, station_mask]
        Q_obs_train = q_obs_tensor[start_idx + WARMUP:end_idx, :]

        # Simple loss computation
        chunk_losses = []
        for i in range(n_stations):
            q_o = Q_obs_train[:, i]
            q_s = Q_sim_train[:, i]

            valid = ~torch.isnan(q_o)
            if valid.sum() < 5:
                continue

            q_o_v = q_o[valid]
            q_s_v = q_s[valid]

            # Use simple MSE for clearer gradients
            mse_loss = torch.mean((q_o_v - q_s_v) ** 2)
            chunk_losses.append(mse_loss)

            # Track KGE separately
            with torch.no_grad():
                kge_val = compute_kge(q_o_v.cpu(), q_s_v.cpu())
                epoch_kges.append(float(kge_val))

        if len(chunk_losses) > 0:
            loss = torch.stack(chunk_losses).mean()
            epoch_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Don't clip gradients for diagnostics
            optimizer.step()

        # HEAVY DIAGNOSTICS every epoch
        if epoch % 1 == 0:
            print("\n" + "="*80)
            heavy_diagnostics(model, territorial, node_coords, forcing, doy, device, epoch)
            print("="*80)

        # Quick validation
        model.eval()
        with torch.no_grad():
            val_state = HydroState.default_warm(n_nodes, device=device)
            Q_val, _ = model.simulate(
                forcing=forcing[val_sl][:30],  # Just 30 days for speed
                initial_state=val_state,
                graph=graph,
                node_coords=node_coords,
                territorial=territorial,
                withdrawals=withdrawals,
                day_of_year=doy[val_sl][:30],
            )

            # Get outlet discharge for physics check
            outlet_Q = Q_val[:, -1].mean().item()

            val_kges = []
            for i in range(min(3, n_stations)):  # Just check a few stations
                q_o = q_obs_tensor[val_sl, i][:30].cpu()
                q_s = Q_val[:, station_mask][:, i].cpu()

                valid = ~torch.isnan(q_o)
                if valid.sum() < 10:
                    continue

                kge_val = compute_kge(q_o[valid], q_s[valid])
                val_kges.append(float(kge_val))

            mean_val_kge = np.mean(val_kges) if val_kges else -999
            mean_train_kge = np.mean(epoch_kges) if epoch_kges else -999
            mean_loss = np.mean(epoch_losses) if epoch_losses else 999

            print(f"\nEpoch {epoch:2d} SUMMARY:")
            print(f"  Loss: {mean_loss:.6f} | Train KGE: {mean_train_kge:.3f} | Val KGE: {mean_val_kge:.3f}")
            print(f"  Outlet discharge: {outlet_Q:.6f} m³/s")

            # Physics progress tracking
            if outlet_Q > 0.1:
                print(f"  🎉 PHYSICS BREAKTHROUGH: Discharge increased to {outlet_Q:.3f} m³/s!")
            elif outlet_Q > 0.05:
                print(f"  📈 Physics improving: discharge = {outlet_Q:.4f} m³/s")
            else:
                print(f"  ⚠️  Physics still broken: discharge = {outlet_Q:.6f} m³/s")

    print(f"\n=== HEAVY DIAGNOSTICS COMPLETE ===")
    print(f"Check diagnostics above to identify the ROOT CAUSE of physics issues")


if __name__ == "__main__":
    main()