import torch
ck = torch.load(r".runs/slso/checkpoints/best-phenology-no-gru.pt", map_location="cpu", weights_only=False)
sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
n_gru = sum(1 for k in sd if "temporal_encoder" in k)
print(f"temporal_encoder keys in checkpoint : {n_gru}")
ik = ck.get("init_kwargs", {})
print(f"use_temporal = {ik.get('use_temporal')}")
print(f"use_phenology_modulator = {ik.get('use_phenology_modulator')}")
print("PhenologyModulator coefs apprises :")
for k, v in sd.items():
    if "phenology_modulator" in k:
        print(f"  {k} = {float(v):.4f}")
