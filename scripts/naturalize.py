"""Counterfactual naturalization — compute streamflow without human withdrawals.

Usage:
    python scripts/naturalize.py --checkpoint checkpoints/nicolet/best.pt \\
                                  --config configs/nicolet.yaml \\
                                  --output results/nicolet_natural.nc
"""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    import torch

    from meandre.data.forcing import ForcingDataset
    from meandre.data.graph_builder import from_shapefile
    from meandre.data.territorial_loader import load_territorial
    from meandre.data.withdrawals_loader import load_withdrawals
    from meandre.model import YHydro
    from meandre.routing.withdrawals import WithdrawalData
    from meandre.utils.state import HydroState

    parser = argparse.ArgumentParser(description="Naturalized flow simulation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/naturalized.pt")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    device = torch.device("cpu")

    forcing = ForcingDataset(cfg.data.forcing_path, device=device).load()
    graph = from_shapefile(cfg.data.network_shp)
    territorial = load_territorial(cfg.data.territorial_path, device=device)
    actual_withdrawals = load_withdrawals(
        cfg.data.withdrawals_dir,
        n_timesteps=forcing.n_timesteps,
        n_reaches=cfg.basin.n_nodes,
        device=device,
    )

    model = YHydro.load(args.checkpoint, n_nodes=cfg.basin.n_nodes, **cfg.model)
    model.eval()

    initial_state = HydroState.zeros(cfg.basin.n_nodes, device=device)

    with torch.no_grad():
        # Anthropogenic scenario (actual withdrawals)
        Q_anthropic, _ = model.simulate(
            forcing.data, initial_state, graph,
            territorial=territorial,
            withdrawals=actual_withdrawals,
            day_of_year=forcing.day_of_year(),
            node_coords=torch.zeros(cfg.basin.n_nodes, 2),  # TODO: load real coords
        )

        # Naturalized scenario (withdrawals -> 0)
        Q_natural, _ = model.simulate(
            forcing.data, initial_state, graph,
            territorial=territorial,
            withdrawals=WithdrawalData.zeros_like(actual_withdrawals),
            day_of_year=forcing.day_of_year(),
            node_coords=torch.zeros(cfg.basin.n_nodes, 2),
        )

    impact = Q_anthropic - Q_natural  # (n_timesteps, n_reaches)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"Q_anthropic": Q_anthropic, "Q_natural": Q_natural, "impact": impact},
        out,
    )
    log.info(f"Saved naturalized results to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
