"""Visualisation PhenologyModulator APPRIS (post-v4) vs INIT vs hardcoded.

Trois panneaux :
  (1) shape(GDD) appris vs init
  (2) Trajectoire K_c_eff(jour) 2018-2019 : appris vs init vs hardcoded
  (3) Tableau coefficients appris

Sert de figure-clé section identifiabilité du POC.

Lancer :
  uv run python .runs/slso/viz_phenology_learned.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from meandre.temporal.phenology_modulator import PhenologyModulator
from meandre.temporal.indices import compute_all_indices

OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path(".runs/slso/checkpoints/best-phenology-modulator.pt")


def load_learned_modulator() -> PhenologyModulator:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    # Extract phenology_modulator submodule weights
    pm_sd = {k.replace("phenology_modulator.", ""): v
             for k, v in sd.items() if k.startswith("phenology_modulator.")}
    if not pm_sd:
        raise RuntimeError("Pas de phenology_modulator dans checkpoint")
    m = PhenologyModulator()
    m.load_state_dict(pm_sd, strict=False)
    m.eval()
    return m


def main():
    m_init = PhenologyModulator(); m_init.eval()
    m_learned = load_learned_modulator()
    print(f"Init     : {m_init.extra_repr()}")
    print(f"Appris   : {m_learned.extra_repr()}")

    # GDD réel SLSO 2018-2019
    ds = xr.open_dataset(".runs/slso/data/forcing.nc")
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    mask = (dt_all >= pd.Timestamp("2018-01-01")) & (dt_all <= pd.Timestamp("2019-12-31"))
    fc_t = torch.from_numpy(fc_all[mask])
    dates_t = dt_all[mask]
    doy_t = torch.tensor(dates_t.dayofyear.values, dtype=torch.long)
    idx = compute_all_indices(fc_t, doy_t)
    gdd_cum = idx["gdd_cum"]
    swe_proxy = idx["swe_proxy"]
    T_air = 0.5 * (fc_t[..., 1] + fc_t[..., 2])

    annual_gdd = gdd_cum[364].numpy()
    node_typical = int(np.argsort(annual_gdd)[len(annual_gdd)//2])
    gdd_typ = gdd_cum[:, node_typical]
    T_typ = T_air[:, node_typical].numpy()
    swe_typ = swe_proxy[:, node_typical].numpy()

    K_c_b = torch.full((len(gdd_typ),), 0.85)
    with torch.no_grad():
        Kc_init = m_init(gdd_typ, K_c_b).numpy() * np.exp(-swe_typ / 10)
        Kc_lrn = m_learned(gdd_typ, K_c_b).numpy() * np.exp(-swe_typ / 10)
    hard = (1.0 / (1.0 + np.exp(-(T_typ - 5.0) / 2.0))) * np.exp(-swe_typ / 10.0)
    Kc_hard = 0.85 * (0.3 + 0.7 * hard)

    gdd_range = torch.linspace(0, 2200, 220)
    with torch.no_grad():
        shape_init = m_init.shape(gdd_range).numpy()
        shape_lrn = m_learned.shape(gdd_range).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(gdd_range.numpy(), shape_init, ls="--", lw=1.5, label="Init littérature")
    ax.plot(gdd_range.numpy(), shape_lrn, lw=2, label="Appris (post-v4)")
    ax.axvline(m_init.gdd_emerg.item(), ls=":", c="gray", alpha=0.5)
    ax.axvline(m_learned.gdd_emerg.item(), ls=":", alpha=0.7)
    ax.set_xlabel("GDD cumulé (°C·j)"); ax.set_ylabel("shape(GDD) ∈ [0,1]")
    ax.set_title("Forme phénologique : init vs apprise"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates_t, Kc_hard, lw=1, alpha=0.6, label="Hardcoded (T_air)")
    ax.plot(dates_t, Kc_init, ls="--", lw=1.2, label="GDD — init littérature")
    ax.plot(dates_t, Kc_lrn, lw=1.8, label="GDD — APPRIS")
    ax.set_xlabel("Date"); ax.set_ylabel("K_c effectif (forêt, K_c_base=0.85)")
    ax.set_title(f"Trajectoire K_c — nœud SLSO médian (annual GDD={annual_gdd[node_typical]:.0f})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle("PhenologyModulator — coefficients appris vs init", fontsize=12)
    out = OUT / "viz_phenology_learned.png"
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nPNG : {out}")

    # Table delta
    print("\nCoefficient            init      appris     Δ%")
    for name in ("gdd_emerg", "gdd_mid", "k_c_min", "k_c_max_factor"):
        v0 = getattr(m_init, name).item(); v1 = getattr(m_learned, name).item()
        print(f"  {name:18s}  {v0:8.3f}  {v1:8.3f}  {(v1-v0)/v0*100:+6.1f}%")


if __name__ == "__main__":
    main()
