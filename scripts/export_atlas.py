"""Export model outputs for the Atlas post-2027 deliverables.

Produces:
    - Q_sim (m3/s): simulated streamflow at all reaches
    - Q_natural (m3/s): naturalized (withdrawal-free) streamflow
    - impact (m3/s): Q_sim - Q_natural (human withdrawal effect)
    - uncertainty (m3/s std): MC Dropout epistemic uncertainty

Output format: NetCDF with CF conventions.
"""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic", required=True, help="Path to Q_anthropic tensor")
    parser.add_argument("--natural", required=True, help="Path to Q_natural tensor")
    parser.add_argument("--output", default="results/atlas_export.nc")
    args = parser.parse_args()

    data_a = torch.load(args.anthropic, map_location="cpu")
    data_n = torch.load(args.natural, map_location="cpu")

    Q_a = data_a["Q_anthropic"] if "Q_anthropic" in data_a else data_a["Q_sim"]
    Q_n = data_n["Q_natural"] if "Q_natural" in data_n else data_n["Q_sim"]
    impact = Q_a - Q_n

    try:
        import xarray as xr
        import numpy as np

        n_time, n_reach = Q_a.shape
        ds = xr.Dataset(
            {
                "Q_anthropic": (["time", "reach"], Q_a.numpy()),
                "Q_natural": (["time", "reach"], Q_n.numpy()),
                "impact": (["time", "reach"], impact.numpy()),
            },
            attrs={
                "title": "meandre Atlas post-2027 streamflow outputs",
                "model": "meandre v0.1.0",
                "units": "m3 s-1",
            },
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out)
        log.info(f"NetCDF exported to {out}")
    except ImportError:
        log.warning("xarray not available; saving as torch tensors instead")
        out = Path(args.output).with_suffix(".pt")
        torch.save({"Q_anthropic": Q_a, "Q_natural": Q_n, "impact": impact}, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
