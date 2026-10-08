"""Conductivite de texture par troncon, pour BORNER le champ spatial (2026-10-08, R349).

La pedologie SIIGSOL (parts sable, limon, argile par profondeur, en coordonnees ilr dans la
table `territorial_siigsol` des bases) contraste la plaine argileuse que les fractions PHYSITEL
du parquet territorial ne voient pas : argile mediane 0,17 en Monteregie contre 0,06 en
Outaouais, soit 0,55 contre 1,9 m/j de conductivite de Saxton et Rawls. Libre entre 0,0003 et
55 m/j, le champ les avait apprises a l'envers. Ici : une conductivite de texture par troncon
et par couche, SIIGSOL d'abord (0-30 cm, 30-100 cm, 100-200 cm), PHYSITEL en repli, et des
bornes a un facteur declare autour.
"""
from __future__ import annotations

import numpy as np

from meandre.data.pedotransfert import saxton_rawls

BANDES = {1: ("0_5cm", "5_15cm", "15_30cm"), 2: ("30_60cm", "60_100cm"), 3: ("100_200cm",)}


def _parts_siigsol(con, bandes):
    """Parts (sable, limon, argile) par noeud, moyenne des coordonnees ilr des bandes, NaN si absent."""
    import nuee
    cols = [f"texture_ilr_{i}_{b}" for b in bandes for i in (1, 2)]
    df = con.execute(f"select node_idx, {', '.join(cols)} from territorial_siigsol order by node_idx").fetchdf()
    z1 = df[[c for c in cols if "_ilr_1_" in c]].mean(axis=1).to_numpy()
    z2 = df[[c for c in cols if "_ilr_2_" in c]].mean(axis=1).to_numpy()
    ok = np.isfinite(z1) & np.isfinite(z2)
    parts = np.full((len(df), 3), np.nan)
    if ok.any():
        parts[ok] = np.asarray(nuee.ilr_inv(np.stack([z1[ok], z2[ok]], axis=1)))
    return df.node_idx.to_numpy(), parts


def texture_conductivity(basin_db: str, n_nodes: int, raw_parquet: str | None = None, region: str | None = None) -> dict:
    """Conductivite de texture (m/j) par couche et par noeud, et la source de chaque noeud.

    Retourne {"K_sat_1": ks, "K_sat_2": ks, "K_sat_3": ks, "source": tableau de "siigsol" |
    "physitel" | "aucune"}. Un noeud sans aucune texture garde NaN : l'appelant lui laisse les
    bornes uniques.
    """
    import duckdb
    import pandas as pd
    out = {f"K_sat_{i}": np.full(n_nodes, np.nan) for i in (1, 2, 3)}
    source = np.array(["aucune"] * n_nodes, dtype=object)
    con = duckdb.connect(basin_db, read_only=True)
    try:
        tables = [t[0] for t in con.execute("show tables").fetchall()]
        if "territorial_siigsol" in tables:
            for couche, bandes in BANDES.items():
                idx, parts = _parts_siigsol(con, bandes)
                ok = np.isfinite(parts).all(axis=1) & (idx < n_nodes)
                ks = np.asarray(saxton_rawls(parts[ok, 0], parts[ok, 2])["k_sat"])
                out[f"K_sat_{couche}"][idx[ok]] = ks
                source[idx[ok]] = "siigsol"
    finally:
        con.close()
    if raw_parquet and region:
        r = pd.read_parquet(raw_parquet)
        r = r[r.region.str.lower() == region.lower()]
        if len(r) == n_nodes:
            ks = np.asarray(saxton_rawls(r.f_sand.to_numpy(float), r.f_clay.to_numpy(float))["k_sat"])
            for couche in (1, 2, 3):
                manque = ~np.isfinite(out[f"K_sat_{couche}"]) & np.isfinite(ks)
                out[f"K_sat_{couche}"][manque] = ks[manque]
            source[(source == "aucune") & np.isfinite(ks)] = "physitel"
    out["source"] = source
    return out


def bounds_from_texture(ks: np.ndarray, factor: float, lo_default: float, hi_default: float):
    """Bornes (lo, hi) par noeud : ks / factor et ks * factor, bornes uniques la ou ks manque."""
    lo = np.where(np.isfinite(ks), ks / factor, lo_default)
    hi = np.where(np.isfinite(ks), ks * factor, hi_default)
    return lo.astype(np.float32), hi.astype(np.float32)
