"""Visualisation du PhenologyModulator avant entraînement.

Quatre panneaux :
  (1) shape(GDD) : courbe phénologique pure ∈ [0, 1]
  (2) K_c_eff(GDD) pour plusieurs K_c_base (typiques SLSO)
  (3) Trajectoire temporelle K_c_eff(jour) sur 2 ans, station moyenne SLSO
  (4) Comparaison avec phenology hardcoded existant (σ(T_air-5) × exp(-SWE/10))

Sert à valider visuellement que l'init du modulateur est sensée pour la
forêt boréale Québec avant d'investir dans un entraînement.

Lancer :
  uv run python .runs/slso/viz_phenology.py
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

from meandre.temporal.phenology_modulator import PhenologyModulator
from meandre.temporal.indices import compute_all_indices

OUT = Path(".reports/slso"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Modulateur à l'init littérature
    m = PhenologyModulator()
    m.eval()
    print(f"PhenologyModulator init : {m.extra_repr()}", flush=True)

    # ── Panel (1) : shape(GDD) seule ─────────────────────────────────
    gdd_range = torch.linspace(0, 2200, 220)
    with torch.no_grad():
        shape = m.shape(gdd_range).numpy()
    GDD_E = m.gdd_emerg.item()
    GDD_M = m.gdd_mid.item()
    GDD_S = m.gdd_mid.item() + m.senesc_offset.item()

    # ── Panel (2) : K_c_eff(GDD) pour K_c_base typiques SLSO ─────────
    # Lit les K_c_base réels d'un forward backbone existant
    print("Chargement K_c_base SLSO depuis cache_backbone.npz...", flush=True)
    cache = dict(np.load(".runs/slso/data/cache_backbone.npz", allow_pickle=True))
    sp = cache["spatial_params"]                                              # (n_st, 36)
    # K_c est l'index ... à trouver dans SpatialParams. Pour SLSO, K_c est
    # généralement vers 0.7-1.0 en forêt boréale. Comme on ne connait pas
    # l'index exact ici, on prend des valeurs représentatives.
    K_c_typical = np.array([0.5, 0.7, 0.85, 1.0, 1.15])                       # gammes typiques

    # ── Panel (3) : trajectoire temporelle sur 2 ans ─────────────────
    # Calcule GDD réel sur une station SLSO depuis le forçage
    print("Compute GDD réel SLSO 2018-2019...", flush=True)
    ds = xr.open_dataset(".runs/slso/data/forcing.nc")
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    mask = (dt_all >= pd.Timestamp("2018-01-01")) & (dt_all <= pd.Timestamp("2019-12-31"))
    fc_t = torch.from_numpy(fc_all[mask])                                     # (T, N, 6)
    dates_t = dt_all[mask]
    doy_t = torch.tensor(dates_t.dayofyear.values, dtype=torch.long)
    idx = compute_all_indices(fc_t, doy_t)
    gdd_cum = idx["gdd_cum"]                                                  # (T, N)
    swe_proxy = idx["swe_proxy"]                                              # (T, N)
    T_min = fc_t[..., 1]
    T_max = fc_t[..., 2]
    T_air = 0.5 * (T_min + T_max)

    # Station type "moyenne SLSO" : nœud médian par GDD annuel
    annual_gdd = gdd_cum[365 - 1].numpy()                                     # GDD en fin d'année 1
    node_typical = int(np.median(np.argsort(annual_gdd)[len(annual_gdd)//2:len(annual_gdd)//2+1]))
    gdd_typical = gdd_cum[:, node_typical].numpy()
    T_typical = T_air[:, node_typical].numpy()
    swe_typical = swe_proxy[:, node_typical].numpy()

    # Modulateur appris : K_c_eff(t) pour K_c_base = 0.85 (forêt typique)
    K_c_base_t = torch.full((len(gdd_typical),), 0.85)
    with torch.no_grad():
        K_c_eff_traj = m(gdd_cum[:, node_typical], K_c_base_t).numpy() * np.exp(-swe_typical / 10)

    # Hardcoded actuel pour comparaison
    phenology_hardcoded = (1.0 / (1.0 + np.exp(-(T_typical - 5.0) / 2.0))) * np.exp(-swe_typical / 10.0)
    season_modulator_hardcoded = 0.3 + 0.7 * phenology_hardcoded
    K_c_eff_hardcoded = 0.85 * season_modulator_hardcoded

    # ── Figure 4 panneaux ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) shape(GDD)
    ax = axes[0, 0]
    ax.plot(gdd_range.numpy(), shape, color="steelblue", lw=2)
    ax.axvline(GDD_E, ls="--", c="green", alpha=0.7, label=f"GDD_emerg={GDD_E:.0f}")
    ax.axvline(GDD_M, ls="--", c="orange", alpha=0.7, label=f"GDD_mid={GDD_M:.0f}")
    ax.axvline(GDD_S, ls="--", c="red", alpha=0.7, label=f"Sénescence={GDD_S:.0f}")
    ax.set_xlabel("GDD cumulé (°C·j)")
    ax.set_ylabel("shape(GDD) ∈ [0, 1]")
    ax.set_title("Forme phénologique apprise")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (2) K_c_eff vs GDD pour différents K_c_base
    ax = axes[0, 1]
    cmap = plt.get_cmap("viridis")
    for i, kc in enumerate(K_c_typical):
        K_c_b_t = torch.full((len(gdd_range),), kc)
        with torch.no_grad():
            K_c_eff = m(gdd_range, K_c_b_t).numpy()
        ax.plot(gdd_range.numpy(), K_c_eff, color=cmap(i/len(K_c_typical)),
                lw=2, label=f"K_c_base={kc:.2f}")
    ax.axhline(m.k_c_min.item(), ls=":", c="k", alpha=0.5, label=f"K_c_min={m.k_c_min.item():.2f}")
    ax.set_xlabel("GDD cumulé (°C·j)")
    ax.set_ylabel("K_c_eff")
    ax.set_title("K_c effectif vs GDD pour différents K_c_base")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (3) Trajectoire temporelle 2 ans
    ax = axes[1, 0]
    ax.plot(dates_t, K_c_eff_traj, color="steelblue", lw=1.5, label="PhenologyModulator (GDD)")
    ax.plot(dates_t, K_c_eff_hardcoded, color="firebrick", lw=1, alpha=0.7,
            label="Hardcoded actuel (T_air)")
    ax.set_xlabel("Date")
    ax.set_ylabel("K_c_eff (forêt K_c_base=0.85)")
    ax.set_title(f"Trajectoire K_c_eff — nœud SLSO médian (annual GDD = {annual_gdd[node_typical]:.0f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Format dates en mois
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # (4) GDD trajectory + T_air pour contexte
    ax = axes[1, 1]
    ax.plot(dates_t, gdd_typical, color="darkgreen", lw=1.5, label="GDD cumulé")
    ax.set_xlabel("Date")
    ax.set_ylabel("GDD cumulé (°C·j)", color="darkgreen")
    ax.tick_params(axis="y", labelcolor="darkgreen")
    ax2 = ax.twinx()
    ax2.plot(dates_t, T_typical, color="firebrick", lw=0.8, alpha=0.6, label="T_air")
    ax2.set_ylabel("T_air (°C)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax.axhline(GDD_E, ls="--", c="green", alpha=0.5)
    ax.axhline(GDD_M, ls="--", c="orange", alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_title("GDD réel + T_air SLSO (contexte climat)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"PhenologyModulator — INIT littérature (avant entraînement) — "
                 f"4 params apprenables", fontsize=13)
    out_png = OUT / "viz_phenology_init.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nPNG : {out_png}", flush=True)

    # Stats imprimées
    print(f"\nStats GDD station typique 2018 :")
    print(f"  GDD final année 1 : {gdd_cum[364, node_typical].item():.0f} °C·j")
    print(f"  GDD final année 2 : {gdd_cum[-1, node_typical].item():.0f} °C·j")
    print(f"  Reset OK : GDD[365] = {gdd_cum[365, node_typical].item():.1f}")
    # Date d'émergence prédite (GDD passe au-dessus de GDD_emerg)
    emerg_idx = (gdd_cum[:365, node_typical] >= GDD_E).nonzero()
    if len(emerg_idx) > 0:
        emerg_date = dates_t[int(emerg_idx[0].item())]
        print(f"  Date émergence prédite (year 1) : {emerg_date.date()}")
    print(f"\nK_c_eff range trajectoire 2 ans : "
          f"[{K_c_eff_traj.min():.3f}, {K_c_eff_traj.max():.3f}]")
    print(f"K_c_eff hardcoded existing range : "
          f"[{K_c_eff_hardcoded.min():.3f}, {K_c_eff_hardcoded.max():.3f}]")


if __name__ == "__main__":
    main()
