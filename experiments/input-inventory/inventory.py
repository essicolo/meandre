"""Inventaire des entrées du modèle provincial, territoire par territoire, sans simulation.

Pour chaque donnée : part des tronçons couverte par ce que le modèle charge, et, quand la source
complète existe à côté, part qu'elle couvrirait. Lit les bases DuckDB, les tables dérivées et les
projets PHYSITEL ; n'importe que le module des chemins.

    python experiments/input-inventory/inventory.py

Sortie : <MEANDRE_DATA>/analysis/input-inventory/inventory.csv et le tableau imprimé.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meandre.utils import paths

TERRITORIES = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont", "outm", "outv", "sagu", "slno", "slso"]
DATA = Path(paths.DATA_ROOT)
AUX = Path(paths.DERIVED_ROOT) / "auxiliaires"
PLATFORMS = Path(paths.PLATFORMS_ROOT)
OUTPUT = DATA / "analysis" / "input-inventory" / "inventory.csv"


def by_reach(path, value, region="region", reach="troncon", one_based=True):
    """Part des tronçons couverts, par territoire, d'une table dérivée (valeur non nulle) ; vide si la table manque."""
    if not Path(path).exists():
        return pd.Series(dtype=float)
    t = pd.read_parquet(path)
    t = t.assign(_t=t[region].str.lower(), _ok=value(t))
    return t.groupby("_t")._ok.sum()


siigsol = by_reach(AUX / "siigsol-troncons.parquet", lambda t: t.texture_ilr_1_0_5cm.notna())
slc = by_reach(AUX / "slc-drainage-troncons.parquet", lambda t: t.couv_drainage > 0)
hier = by_reach(AUX / "drainage-hierarchique-troncons.parquet", lambda t: t.drainage.notna())
deposits = by_reach(AUX / "epaisseur-depots-troncons.parquet", lambda t: t.iloc[:, -1].notna()) if (AUX / "epaisseur-depots-troncons.parquet").exists() else pd.Series(dtype=float)
irda = by_reach(AUX / "irda-troncons.parquet", lambda t: t.filter(like="drainage").iloc[:, 0].notna())
imp = pd.read_parquet(AUX / "impermeable-worldcover-troncons.parquet")
imp = imp.assign(_t=imp.territory.str.lower(), _ok=imp.impervious.notna()).groupby("_t")._ok.sum()
wells = pd.read_parquet(AUX / "rsesq-puits.parquet").region.str.lower().value_counts()
hl = pd.read_parquet(DATA / "quebec" / "lacs_hydrolakes.parquet").region.str.lower().value_counts()

rows = []
for t in TERRITORIES:
    con = duckdb.connect(str(DATA / "quebec" / f"{t}.duckdb"), read_only=True)
    tables = set(con.execute("show tables").df().name)
    n = con.execute("select count(*) from nodes").fetchone()[0]

    def count(sql, table):
        return con.execute(sql).fetchone()[0] if table in tables else 0

    loaded_siigsol = count("select count(*) from territorial_siigsol where texture_ilr_1_0_5cm is not null", "territorial_siigsol")
    raw_cols = [c for c in con.execute("select * from territorial limit 0").df().columns if c.endswith("_raw")]
    row = {
        "territoire": t,
        "troncons": n,
        "stations": count("select count(distinct station_id) from stations", "stations"),
        "siigsol_charge": loaded_siigsol / n,
        "siigsol_disponible": siigsol.get(t, 0) / n,
        "drainage_pedopaysages": slc.get(t, 0) / n,
        "drainage_hierarchique_disponible": hier.get(t, 0) / n,
        "epaisseur_depots": deposits.get(t, 0) / n,
        "irda": irda.get(t, 0) / n,
        "impermeable_worldcover": imp.get(t, 0) / n,
        "lacs_hydrolakes": int(hl.get(t, 0)),
        "occupation_brute": len(raw_cols),
        "prelevements_lignes": count("select count(*) from withdrawals", "withdrawals"),
        "mod16_jours": count("select count(distinct date) from modis_et", "modis_et"),
        "grace_mois": count("select count(*) from grace_tws", "grace_tws"),
        "neige_sites": count("select count(*) from snow_sites", "snow_sites"),
        "neisim": (AUX / f"neisim-{t}.npz").exists(),
        "puits": int(wells.get(t, 0)),
    }
    con.close()
    project = PLATFORMS / "LN24HA" / f"{t.upper()}_LN24HA_2020"
    row["hgm"] = (project / "hgm").is_dir()
    row["milieux_humides"] = any(project.rglob("milieux_humides_isoles.csv")) if project.exists() else False
    forcing = DATA / "quebec" / f"forcing-{t}-casr-brut.nc"
    if forcing.exists():
        with xr.open_dataset(forcing) as ds:
            row["vent_moyen_m_s"] = float(ds["forcing"].sel(var="u2").isel(time=slice(0, 3650)).mean())
    else:
        row["vent_moyen_m_s"] = np.nan
    rows.append(row)

df = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(df.round(2).to_string(index=False))
print(f"écrit : {OUTPUT}")


# CONTRÔLES DES CHARGEURS (2026-11-13). Les colonnes ci-dessus disent si une source existe ; celles-ci
# appellent les fonctions mêmes de l'adaptateur et disent ce que le modèle reçoit réellement,
# tronçon par tronçon : source de la texture, repli, milieux humides, ordre des attributs.
import physitel_checks  # noqa: E402

physitel_checks.run(TERRITORIES, DATA, AUX, OUTPUT.parent)
