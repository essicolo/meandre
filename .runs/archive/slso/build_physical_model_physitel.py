# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: meandre
#     language: python
#     name: meandre
# ---

# %% [markdown]
# # Construction du modèle physique — SLSO
#
# Ce script importe le projet PHYSITEL dans un cache DuckDB utilisé par
# `slso.py` pour l'entraînement et `diagnostics.qmd` pour l'analyse.
#
# **À exécuter une seule fois**, ou lorsque les données PHYSITEL changent.

# %%
import os
import subprocess
from pathlib import Path

os.chdir(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())

import tomllib

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)

BASIN_DB    = Path(cfg["paths"]["basin_db"])
PHYSITEL_DIR = Path("notebooks/slso")   # racine du projet HYDROTEL (contient physitel/ et etat/)

print(f"PHYSITEL : {PHYSITEL_DIR}")
print(f"DuckDB  : {BASIN_DB}")

# %% [markdown]
# ## 1. Import PHYSITEL → DuckDB

# %%
from meandre.data.basin_cache import BasinCache

# Supprime l'ancien cache si présent
if BASIN_DB.exists():
    BASIN_DB.unlink()
    print(f"Ancien cache supprimé: {BASIN_DB}")

BASIN_DB.parent.mkdir(parents=True, exist_ok=True)

cache = BasinCache.from_hydrotel(
    project_dir=PHYSITEL_DIR,
    path=BASIN_DB,
)
print(f"Cache créé: {BASIN_DB}")

# %% [markdown]
# ## 2. Vérification du contenu

# %%
import duckdb

con = duckdb.connect(str(BASIN_DB), read_only=True)

print("=== Tables ===")
tables = con.execute("SHOW TABLES").fetchall()
for t in tables:
    n_rows = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
    cols = con.execute(f"DESCRIBE {t[0]}").fetchdf()
    print(f"\n{t[0]} ({n_rows} lignes) :")
    for _, row in cols.iterrows():
        print(f"  {row['column_name']:25s} {row['column_type']}")

con.close()

# %% [markdown]
# ## 3. Validation du réseau hydrographique

# %%
import numpy as np
import torch

hydro = cache.load(device=torch.device("cpu"))
graph = hydro["graph"]
coords = hydro["node_coords"].numpy()
lon, lat = coords[:, 0], coords[:, 1]
n_nodes = hydro["n_nodes"]

print(f"Nœuds : {n_nodes}")
print(f"Arêtes: {graph.n_edges}")
print(f"Lacs  : {graph.is_lake.sum().item()}")

# Vérifier les arêtes longues (possibles erreurs de topologie)
edge_src = graph.edge_index[0].numpy()
edge_dst = graph.edge_index[1].numpy()

dx = lon[edge_src] - lon[edge_dst]
dy = lat[edge_src] - lat[edge_dst]
dist_deg = np.sqrt(dx**2 + dy**2)
dist_km = dist_deg * 111.0  # approximation à cette latitude

print(f"\nDistance des arêtes (km):")
print(f"  Médiane : {np.median(dist_km):.2f}")
print(f"  P95     : {np.percentile(dist_km, 95):.2f}")
print(f"  Max     : {np.max(dist_km):.2f}")

LONG_EDGE_KM = 15.0
long_edges = np.where(dist_km > LONG_EDGE_KM)[0]
if len(long_edges) > 0:
    print(f"\n⚠ {len(long_edges)} arêtes > {LONG_EDGE_KM} km:")
    for i in long_edges[:20]:
        s, d = edge_src[i], edge_dst[i]
        print(f"  {s} → {d}  ({dist_km[i]:.1f} km)  "
              f"({lon[s]:.3f},{lat[s]:.3f}) → ({lon[d]:.3f},{lat[d]:.3f})")
    if len(long_edges) > 20:
        print(f"  ... et {len(long_edges) - 20} de plus")
else:
    print(f"\n✓ Aucune arête > {LONG_EDGE_KM} km")

# Vérifier la connectivité (composantes connexes)
from collections import defaultdict, deque

adj = defaultdict(set)
for s, d in zip(edge_src, edge_dst):
    adj[int(s)].add(int(d))
    adj[int(d)].add(int(s))

visited = set()
components = []
for node in range(n_nodes):
    if node in visited:
        continue
    comp = set()
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        for nb in adj[n]:
            if nb not in visited:
                queue.append(nb)
    components.append(comp)

print(f"\nComposantes connexes: {len(components)}")
if len(components) > 1:
    sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Tailles: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
    print(f"  ⚠ Le réseau n'est pas entièrement connecté")
else:
    print(f"  ✓ Réseau entièrement connecté ({n_nodes} nœuds)")

# %% [markdown]
# ## 4. Import des observations hydrométriques

# %%
STATIONS_FILE = Path(r"C:\Users\parse01\documents-locaux\rqh-local\rqh_2026-04\data\07_stations\stations_concatenees.nc")

cache.import_observations(STATIONS_FILE, basin_prefix="SLSO")

obs = cache.load_observations(
    date_start=cfg["temporal"]["date_start"],
    date_end=cfg["temporal"]["date_end"],
    min_valid_days=365,
)

n_stations = obs["n_stations"]
print(f"Stations retenues: {n_stations}")
print(f"Période: {cfg['temporal']['date_start']} – {cfg['temporal']['date_end']}")

# %% [markdown]
# ## 5. Import des prélèvements et rejets

# %%
WITHDRAWALS_FILE = Path("notebooks/io-eau-meandre.parquet")

if WITHDRAWALS_FILE.exists():
    # Vérifier si déjà importé
    _con = duckdb.connect(str(BASIN_DB), read_only=True)
    _has_wd = "withdrawals" in [
        r[0] for r in _con.execute("SHOW TABLES").fetchall()
    ]
    _con.close()

    if not _has_wd:
        n_imported = cache.import_withdrawals(
            WITHDRAWALS_FILE, site_col="site_id",
        )
        print(f"Prélèvements importés: {n_imported} lignes")
    else:
        _con = duckdb.connect(str(BASIN_DB), read_only=True)
        n = _con.execute("SELECT COUNT(DISTINCT node_idx) FROM withdrawals").fetchone()[0]
        _con.close()
        print(f"Prélèvements déjà importés ({n} nœuds)")
else:
    print(f"⚠ Fichier non trouvé: {WITHDRAWALS_FILE}")
    print("  Les prélèvements ne seront pas disponibles.")

# %% [markdown]
# ## Résumé
#
# Le DuckDB contient maintenant :
# - Le réseau hydrographique (nœuds, arêtes, lacs)
# - Les caractéristiques territoriales (occupation du sol, pente, élévation, ...)
# - L'état initial (cold state)
# - Les observations hydrométriques
# - Les prélèvements/rejets (si disponibles)
#
# Prochaine étape : `slso.py` pour l'entraînement.