"""Patch the KGE=0.904 checkpoint for f_vert warm-start.

Replaces the 3 fc_out columns that changed from the old parameterization
(slope_factor, krec, k_interflow) to the new one (f_vert_1, f_vert_3, f_vert_2)
with literature-default values, while preserving the 33 learned columns.

Usage:
    python notebooks/slso/patch_fvert_checkpoint.py
"""

import math
import torch
from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams

CKPT_IN = "notebooks/slso/checkpoints/best_phaseB_kge0904_epoch50.pt"
CKPT_OUT = "notebooks/slso/checkpoints/best-fvert-warmstart.pt"

# Positions of the 3 changed parameters in the 36-dim fc_out output
# Old: slope_factor (22), krec (23), k_interflow (29)
# New: f_vert_1 (22), f_vert_3 (23), f_vert_2 (29)
FVERT_POSITIONS = [22, 23, 29]

# Literature defaults for f_vert (from SpatialFieldNetwork.init_from_literature)
FVERT_DEFAULTS = {
    "f_vert_1": 0.50,  # 50% vertical drainage from layer 1
    "f_vert_3": 0.70,  # 70% vertical (recharge) from layer 3
    "f_vert_2": 0.60,  # 60% vertical from layer 2
}

# f_vert uses bounded constraint [0, 1], so raw = inv_bounded(value, 0, 1)
# inv_bounded(x, lo, hi) = logit((x - lo) / (hi - lo)) = log(x / (1 - x))
# For x=0.5: raw = log(0.5/0.5) = 0.0
# For x=0.6: raw = log(0.6/0.4) = 0.405
# For x=0.7: raw = log(0.7/0.3) = 0.847


def inv_bounded(value, lo, hi):
    return math.log((value - lo) / (hi - value))


def main():
    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    fc_out_w = sd["spatial_encoder.fc_out.weight"].clone()  # (36, 256)
    fc_out_b = sd["spatial_encoder.fc_out.bias"].clone()      # (36,)

    print(f"Checkpoint loaded: {CKPT_IN}")
    print(f"fc_out.weight shape: {fc_out_w.shape}")
    print(f"fc_out.bias shape: {fc_out_b.shape}")
    print(f"\nReplacing positions {FVERT_POSITIONS} with f_vert literature defaults:")

    # Weight shrink factor (same as init_from_literature)
    weight_shrink = 0.01

    for pos, (name, default_val) in zip(FVERT_POSITIONS, FVERT_DEFAULTS.items()):
        raw_bias = inv_bounded(default_val, 0.0, 1.0)
        old_bias = fc_out_b[pos].item()
        old_w_norm = fc_out_w[pos].norm().item()

        # Set bias to literature default
        fc_out_b[pos] = raw_bias
        # Shrink weight row (same as init_from_literature: small weights = uniform init)
        fc_out_w[pos] = fc_out_w[pos] * weight_shrink

        print(f"  pos {pos} ({name}): bias {old_bias:.4f} -> {raw_bias:.4f}, "
              f"weight ||w|| {old_w_norm:.4f} -> {fc_out_w[pos].norm().item():.6f}")

    sd["spatial_encoder.fc_out.weight"] = fc_out_w
    sd["spatial_encoder.fc_out.bias"] = fc_out_b

    # Update init_kwargs to match current model
    if "init_kwargs" in ckpt:
        ckpt["init_kwargs"]["n_territorial"] = 17  # updated from 16

    torch.save(ckpt, CKPT_OUT)
    print(f"\nPatched checkpoint saved: {CKPT_OUT}")
    print(f"  33 learned parameters preserved")
    print(f"  3 f_vert parameters reset to literature defaults")


if __name__ == "__main__":
    main()