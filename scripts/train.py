"""CLI training entry point.

Usage:
    python scripts/train.py config=configs/nicolet.yaml
    python scripts/train.py config=configs/nicolet.yaml training.lr=5e-4
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="nicolet")
def main(cfg: DictConfig) -> None:
    import torch

    from meandre.data.forcing import ForcingDataset
    from meandre.data.graph_builder import from_shapefile
    from meandre.data.observations import ObservationDataset
    from meandre.data.territorial_loader import load_territorial
    from meandre.data.withdrawals_loader import load_withdrawals
    from meandre.model import YHydro
    from meandre.training.loss import HydroLoss
    from meandre.training.trainer import Trainer, TrainingConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    # ---- Data ----
    forcing = ForcingDataset(cfg.data.forcing_path, device=device).load()
    obs = ObservationDataset(cfg.data.observations_path, station_node_map={}, device=device).load()
    graph = from_shapefile(cfg.data.network_shp)
    territorial = load_territorial(cfg.data.territorial_path, device=device)
    withdrawals = load_withdrawals(
        cfg.data.withdrawals_dir,
        n_timesteps=forcing.n_timesteps,
        n_reaches=cfg.basin.n_nodes,
        device=device,
    )

    # ---- Model ----
    model = YHydro(
        n_nodes=cfg.basin.n_nodes,
        **cfg.model,
    ).to(device)

    loss_fn = HydroLoss(**cfg.loss)

    train_cfg = TrainingConfig(**cfg.training)
    trainer = Trainer(
        model=model,
        train_dataset=None,  # TODO: wrap into dataset class
        val_dataset=None,
        config=train_cfg,
        mlflow_run_name=cfg.basin.name,
    )
    trainer.fit()

    # Save checkpoint
    ckpt_dir = Path(cfg.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / "best.pt")
    log.info(f"Checkpoint saved to {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
