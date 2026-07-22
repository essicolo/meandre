"""Demande évaporative du module ET appris (banc et_bench, MLP gelé supervisé MOD16).
Factorisé depuis etl_run.py pour réutilisation par joint.py (conjoint) et les runs mono.
compute_demand(...) -> (T, N) mm/j, fenêtres traînantes 8 j / 90 j (voir design_et_appris.md).
"""
import numpy as np
import torch
import torch.nn as nn

ETB = "D:/meandre-data/quebec/checkpoints-etbench"


def load_mlp(f_static, device):
    norm = torch.load(f"{ETB}/norm.pt", weights_only=False)
    mlp = nn.Sequential(nn.Linear(12 + f_static + 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()).to(device)
    sd = torch.load(f"{ETB}/mlp.pt", weights_only=True)
    mlp.load_state_dict({k.replace("head.", ""): v for k, v in sd.items()})
    mlp.eval()
    return mlp, norm


def compute_demand(forcing, day_of_year, node_coords, territorial, device):
    """forcing (T, N, >=6) sur device ; retourne demande (T, N) mm/j (no_grad, module gelé)."""
    n_nodes = forcing.shape[1]
    mlp, norm = load_mlp(territorial.n_features, device)
    H_HIST, H_COMP = norm["h_hist"], norm["h_comp"]
    with torch.no_grad():
        T = forcing.shape[0]
        mean, std = norm["mean"].to(device), norm["std"].to(device)
        C = torch.cat([torch.zeros(1, n_nodes, 6, device=device), forcing[:, :, :6].cumsum(0)], dim=0)
        t_ar = torch.arange(T, device=device)
        lo8 = torch.clamp(t_ar - (H_COMP - 1), min=0)
        a8 = (C[t_ar + 1] - C[lo8]) / (t_ar + 1 - lo8).reshape(-1, 1, 1)
        hi90, lo90 = torch.clamp(t_ar - (H_COMP - 1), min=1), torch.clamp(t_ar - (H_COMP - 1) - H_HIST, min=0)
        a90 = (C[hi90] - C[lo90]) / torch.clamp(hi90 - lo90, min=1).reshape(-1, 1, 1)
        sc = torch.stack([torch.sin(2 * np.pi * day_of_year / 365.25),
                          torch.cos(2 * np.pi * day_of_year / 365.25)], dim=1)
        lat_col = 0 if 40 < float(node_coords[:, 0].mean()) < 62 else 1
        lat = node_coords[:, lat_col].float() / 50.0
        stat = torch.cat([territorial.data, lat[:, None]], dim=1)
        demand = torch.empty(T, n_nodes, device=device)
        for lo in range(0, T, 365):
            hi = min(lo + 365, T)
            a8n = (a8[lo:hi] - mean) / std
            a90n = (a90[lo:hi] - mean) / std
            scb = sc[lo:hi, None, :].expand(hi - lo, n_nodes, 2)
            x = torch.cat([a8n, a90n, stat[None, :, :-1].expand(hi - lo, -1, -1), scb,
                           stat[None, :, -1:].expand(hi - lo, -1, -1)], dim=2)
            demand[lo:hi] = mlp(x.reshape(-1, x.shape[-1])).reshape(hi - lo, n_nodes)
    return demand
