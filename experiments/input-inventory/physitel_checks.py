"""Ce que le modèle reçoit réellement de chaque chargeur de l'adaptateur PHYSITEL, par territoire.

Appelé par inventory.py. Trois tableaux, dans le dossier de l'inventaire :
- chargeurs.csv : par territoire, part des tronçons servie par chaque source et chaque repli ;
- attributs.csv : par territoire et par attribut du champ, médiane brute, part nulle, et accord
  d'ordre entre le brut provincial et la table de la base (corrélation 1 attendue) ;
- frontieres.csv : par attribut, écart entre tronçons voisins de part et d'autre d'une frontière
  de territoire, rapporté au même écart entre voisins d'un même territoire.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meandre.adapters import physitel
from meandre.adapters.physitel.hydrotel_calib import load_wetlands

NEIGHBOUR_KM = 5.0


def nodes_of(data, territory):
    con = duckdb.connect(str(data / "quebec" / f"{territory}.duckdb"), read_only=True)
    nodes = con.execute("select node_idx, node_id, lon, lat from nodes order by node_idx").df()
    local = con.execute("select * from territorial order by node_idx").df()
    con.close()
    return nodes, local


def loader_row(territory, nodes, aux):
    node_ids = nodes.node_id.to_numpy()
    n = len(node_ids)
    row = {"territoire": territory, "troncons": n}

    texture = physitel.texture_conductivity(territory, node_ids, n)
    source = pd.Series(texture["source"])
    row["texture_siigsol"] = (source == "siigsol").mean()
    row["texture_physitel"] = (source == "physitel").mean()
    row["texture_aucune"] = (source == "aucune").mean()
    row["k_sat_1_mediane_m_j"] = float(np.nanmedian(texture["K_sat_1"]))

    drainage = physitel.drainage_class(territory, node_ids, str(aux / "slc-drainage-troncons.parquet"))
    row["drainage_classe"] = np.isfinite(drainage).mean()
    row["plafond_uniforme_2mm"] = 1.0 - np.isfinite(drainage).mean()

    impervious = physitel.measured_impervious(territory, node_ids, str(aux / "impermeable-worldcover-troncons.parquet"))
    row["impermeable_mesure"] = np.isfinite(impervious).mean()

    deposits = physitel.deposit_thickness(territory, node_ids)
    row["epaisseur_depots"] = np.isfinite(deposits).mean() if deposits is not None else 0.0

    wetlands = load_wetlands(physitel.project_dir(territory), node_ids)
    wet_area = wetlands["wet_a_raw"].numpy() if wetlands else np.zeros(n)
    row["milieux_humides_charges"] = bool(wetlands)
    row["troncons_avec_milieu_humide"] = (wet_area > 0).mean()

    area, coefficient, exponent = physitel.lake_geometry(territory, node_ids, n)
    row["lacs_routes"] = int(np.isfinite(area).sum())

    try:
        physitel.land_cover(territory, node_ids, "cpu")
        row["occupation_projet"] = True
    except Exception as error:
        row["occupation_projet"] = f"{type(error).__name__}"
    return row


def attribute_rows(territory, local, raw, stats):
    rows = []
    own = raw[raw.region == territory]
    same_count = len(own) == len(local)
    for column in stats.index:
        values = own[column].to_numpy()
        if same_count and np.nanstd(values) > 0 and local[column].std() > 0:
            order = float(np.corrcoef(values, local[column].to_numpy())[0, 1])
        else:
            order = np.nan
        rows.append({
            "territoire": territory,
            "attribut": column,
            "lignes_brut": len(own),
            "lignes_base": len(local),
            "mediane_brute": float(np.nanmedian(values)) if len(values) else np.nan,
            "part_nulle": float(np.mean(values == 0)) if len(values) else np.nan,
            "part_manquante": float(np.mean(~np.isfinite(values))) if len(values) else np.nan,
            "accord_ordre": order,
        })
    return rows


def boundary_rows(frames, stats):
    """Écart absolu moyen (en écarts-types provinciaux) entre un tronçon et son plus proche voisin
    à moins de NEIGHBOUR_KM, séparément pour les voisins d'un autre territoire et du même."""
    table = pd.concat(frames, ignore_index=True)
    xy = np.c_[table.lon * 111.0 * np.cos(np.radians(table.lat)), table.lat * 111.0]
    tree = cKDTree(xy)
    distance, index = tree.query(xy, k=8)
    rows = []
    for column in stats.index:
        z = ((table[column] - stats.loc[column, "mean"]) / (stats.loc[column, "std"] + 1e-9)).to_numpy()
        cross, within = [], []
        pair_gaps = {}
        for k in range(1, 8):
            near = distance[:, k] < NEIGHBOUR_KM
            other = index[:, k]
            gap = np.abs(z - z[other])
            different = table.region.to_numpy() != table.region.to_numpy()[other]
            cross.append(gap[near & different])
            within.append(gap[near & ~different])
            for a, b, g in zip(table.region.to_numpy()[near & different], table.region.to_numpy()[other][near & different], gap[near & different]):
                pair_gaps.setdefault(tuple(sorted((a, b))), []).append(g)
        cross, within = np.concatenate(cross), np.concatenate(within)
        worst = max(pair_gaps.items(), key=lambda item: np.mean(item[1]) if len(item[1]) > 20 else 0)
        rows.append({
            "attribut": column,
            "ecart_frontiere": float(np.mean(cross)),
            "ecart_interieur": float(np.mean(within)),
            "rapport": float(np.mean(cross) / max(np.mean(within), 1e-6)),
            "pire_frontiere": "-".join(worst[0]),
            "ecart_pire_frontiere": float(np.mean(worst[1])),
            "paires_frontiere": len(cross),
        })
    return rows


def forcing_row(data, territory):
    path = data / "quebec" / f"forcing-{territory}-casr-brut.nc"
    with xr.open_dataset(path) as ds:
        part = ds["forcing"].isel(time=slice(0, 3653))
        means = part.mean(dim=("time", "node")).to_pandas()
    row = {"territoire": territory}
    row.update({f"{name}_moyen": float(value) for name, value in means.items()})
    return row


def run(territories, data, aux, output):
    raw = pd.read_parquet(physitel.QC_RAW)
    stats = pd.read_csv(physitel.QC_STATS, index_col=0)
    loaders, attributes, frames, forcings = [], [], [], []
    for territory in territories:
        nodes, local = nodes_of(data, territory)
        loaders.append(loader_row(territory, nodes, aux))
        attributes.extend(attribute_rows(territory, local, raw, stats))
        own = raw[raw.region == territory].reset_index(drop=True)
        if len(own) == len(nodes):
            frames.append(pd.concat([nodes[["lon", "lat"]], own], axis=1))
        forcings.append(forcing_row(data, territory))
        print(f"[controles] {territory} lu", flush=True)

    loaders = pd.DataFrame(loaders).merge(pd.DataFrame(forcings), on="territoire")
    attributes = pd.DataFrame(attributes)
    boundaries = pd.DataFrame(boundary_rows(frames, stats)).sort_values("rapport", ascending=False)
    loaders.to_csv(output / "chargeurs.csv", index=False)
    attributes.to_csv(output / "attributs.csv", index=False)
    boundaries.to_csv(output / "frontieres.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(loaders.round(3).to_string(index=False))
    print("\nattributs dont l'ordre ne concorde pas, ou nuls sur plus de 90 % d'un territoire")
    flagged = attributes[(attributes.accord_ordre < 0.999) | (attributes.part_nulle > 0.9) | (attributes.lignes_brut != attributes.lignes_base)]
    print(flagged.round(3).to_string(index=False))
    print("\nsauts aux frontières (rapport de l'écart entre voisins de deux territoires à l'écart entre voisins d'un même territoire)")
    print(boundaries.round(2).to_string(index=False))
