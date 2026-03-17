"""Inference / scenario generation.

Usage:
    python scripts/predict.py --checkpoint checkpoints/nicolet/best.pt \\
                               --config configs/nicolet.yaml \\
                               --output results/q_sim.pt
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
    from meandre.utils.state import HydroState

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/q_sim.pt")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    device = torch.device("cpu")

    forcing = ForcingDataset(cfg.data.forcing_path, device=device).load()
    graph = from_shapefile(cfg.data.network_shp)
    territorial = load_territorial(cfg.data.territorial_path, device=device)
    withdrawals = load_withdrawals(
        cfg.data.withdrawals_dir,
        n_timesteps=forcing.n_timesteps,
        n_reaches=cfg.basin.n_nodes,
        device=device,
    )

    model = YHydro.load(args.checkpoint, n_nodes=cfg.basin.n_nodes, **cfg.model)
    model.eval()

    initial_state = HydroState.zeros(cfg.basin.n_nodes, device=device)

    with torch.no_grad():
        Q_sim, final_state = model.simulate(
            forcing.data, initial_state, graph,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=forcing.day_of_year(),
            node_coords=torch.zeros(cfg.basin.n_nodes, 2),
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Q_sim": Q_sim}, out)
    log.info(f"Predictions saved to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
