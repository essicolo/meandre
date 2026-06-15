"""Ingest remote sensing data into the SLSO DuckDB.

Downloads and ingests all four remote sensing products into the basin DB :

    Source      Variable           Cadence    Contraindre
    -----------------------------------------------------------
    MOD16A2     ETR (mm/day)        8-day      K_c, K_sat, f_vert
    MOD10A1     Snow cover (0-1)    daily      C_f, T_melt, T_snow
    MOD13A2     NDVI (ndvi)         16-day     K_c saisonnier, LAI
    GRACE-FO    ΔStorage (mm/mois)  monthly    k_gw, bilan total

MODIS : Planetary Computer STAC (gratuit, aucun compte requis).
GRACE : NASA Earthdata — requiert EARTHDATA_LOGIN + EARTHDATA_PASSWORD
        dans les variables d'environnement, ou ~/.netrc.

Usage
-----
    # Tout ingérer
    python .runs/slso/ingest_remote_sensing.py

    # Avec une config spécifique
    python .runs/slso/ingest_remote_sensing.py .runs/slso/config/slso-kendall-gal-v2.toml

    # Seulement certains produits
    python .runs/slso/ingest_remote_sensing.py --products et snow
    python .runs/slso/ingest_remote_sensing.py --products et snow ndvi grace

Les scripts sont idempotents : relancer n'ajoute que les nouvelles lignes.
"""
import os
import sys
from pathlib import Path

# Windows : stdout redirigé vers fichier est en cp1252, qui ne sait pas encoder
# les flèches/puces unicode des messages → forcer utf-8 (même fix que les autres
# scripts du repo).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.chdir(Path(__file__).resolve().parents[2])

import argparse
import tomllib
import torch
import numpy as np

from meandre.utils.paths import run_dir_from_config, resolve_run_path

parser = argparse.ArgumentParser(description="Ingest remote sensing into DuckDB")
parser.add_argument("config", nargs="?",
                    default=".runs/slso/config/slso.toml",
                    help="Path to TOML config")
parser.add_argument("--products", nargs="+",
                    choices=["et", "snow", "ndvi", "grace"],
                    default=["et", "snow", "ndvi", "grace"],
                    help="Which products to ingest")
parser.add_argument("--force", action="store_true",
                    help="Re-fetch even if already ingested")
args = parser.parse_args()

CFG_PATH = Path(args.config)
with open(CFG_PATH, "rb") as f:
    cfg = tomllib.load(f)

RUN_DIR = run_dir_from_config(CFG_PATH)

def _p(key: str) -> Path:
    return resolve_run_path(cfg["paths"][key], RUN_DIR)

BASIN_DB = _p("basin_db")
DATE_START = cfg["temporal"]["date_start"]
DATE_END = cfg["temporal"]["date_end"]

print(f"Config   : {CFG_PATH}")
print(f"Basin DB : {BASIN_DB}")
print(f"Period   : {DATE_START} → {DATE_END}")
print(f"Products : {args.products}")

from meandre.data.basin_cache import BasinCache

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
node_coords = hydro["node_coords"].cpu().numpy()
n_nodes = node_coords.shape[0]
lons, lats = node_coords[:, 0], node_coords[:, 1]

bbox = (float(lons.min()) - 0.1, float(lats.min()) - 0.1,
        float(lons.max()) + 0.1, float(lats.max()) + 0.1)

print(f"Nodes    : {n_nodes}")
print(f"Bbox     : ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})")

from meandre.data.modis_loader import (
    fetch_modis_et, fetch_modis_snow, fetch_modis_ndvi
)

node_indices = np.arange(n_nodes)


def _confirm_overwrite(name: str, has_fn) -> bool:
    if args.force:
        return True
    if has_fn():
        ans = input(f"\n{name} déjà présent. Re-fetcher et écraser? [y/N] ").strip().lower()
        return ans == "y"
    return True


# ── MOD16A2 ETR ───────────────────────────────────────────────────────────────
if "et" in args.products and _confirm_overwrite("modis_et", cache.has_modis_et):
    print("\n── MOD16A2 ETR ─────────────────────────────────────────────────────")
    df_et = fetch_modis_et(bbox, DATE_START, DATE_END, node_coords, node_indices)
    if df_et.empty:
        print("  Avertissement : aucune donnée ETR récupérée")
    else:
        n = cache.import_modis_et(df_et)
        good = df_et["quality_ok"].sum()
        print(f"  Ingéré : {n:,} lignes, {df_et['date'].nunique()} composites, "
              f"{good:,} bonne qualité ({good/len(df_et):.1%})")
        print(f"  ETR   : {df_et['etr_mm_day'].dropna().min():.3f} – "
              f"{df_et['etr_mm_day'].dropna().max():.3f} mm/jour")

# ── MOD10A1 snow ──────────────────────────────────────────────────────────────
if "snow" in args.products and _confirm_overwrite("modis_snow", cache.has_modis_snow):
    print("\n── MOD10A1 snow cover ──────────────────────────────────────────────")
    df_snow = fetch_modis_snow(bbox, DATE_START, DATE_END, node_coords, node_indices)
    if df_snow.empty:
        print("  Avertissement : aucune donnée snow récupérée")
    else:
        n = cache.import_modis_snow(df_snow)
        good = df_snow["quality_ok"].sum()
        print(f"  Ingéré : {n:,} lignes, {df_snow['date'].nunique()} jours, "
              f"{good:,} bonne qualité ({good/len(df_snow):.1%})")
        print(f"  Snow  : {df_snow['snow_frac'].dropna().min():.3f} – "
              f"{df_snow['snow_frac'].dropna().max():.3f} (fraction)")

# ── MOD13A2 NDVI ──────────────────────────────────────────────────────────────
if "ndvi" in args.products and _confirm_overwrite("modis_ndvi", cache.has_modis_ndvi):
    print("\n── MOD13A2 NDVI ────────────────────────────────────────────────────")
    df_ndvi = fetch_modis_ndvi(bbox, DATE_START, DATE_END, node_coords, node_indices)
    if df_ndvi.empty:
        print("  Avertissement : aucune donnée NDVI récupérée")
    else:
        n = cache.import_modis_ndvi(df_ndvi)
        good = df_ndvi["quality_ok"].sum()
        print(f"  Ingéré : {n:,} lignes, {df_ndvi['date'].nunique()} composites, "
              f"{good:,} bonne qualité ({good/len(df_ndvi):.1%})")
        print(f"  NDVI  : {df_ndvi['ndvi'].dropna().min():.3f} – "
              f"{df_ndvi['ndvi'].dropna().max():.3f}")

# ── GRACE-FO TWS ──────────────────────────────────────────────────────────────
if "grace" in args.products and _confirm_overwrite("grace_tws", cache.has_grace_tws):
    print("\n── GRACE/GRACE-FO TWS ──────────────────────────────────────────────")

    # ── Avertissement : enregistrer le token de façon permanente ─────────────
    import os as _os, sys as _sys

    def _token_is_permanent() -> bool:
        """Return True if EARTHDATA_TOKEN is already saved permanently."""
        if _sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
                winreg.QueryValueEx(key, "EARTHDATA_TOKEN")
                winreg.CloseKey(key)
                return True
            except (FileNotFoundError, OSError):
                return False
        else:
            import subprocess, pathlib
            rc_files = [
                pathlib.Path.home() / ".zshrc",
                pathlib.Path.home() / ".bashrc",
                pathlib.Path.home() / ".profile",
                pathlib.Path.home() / ".bash_profile",
            ]
            for f in rc_files:
                if f.exists() and "EARTHDATA_TOKEN" in f.read_text(errors="ignore"):
                    return True
            netrc = pathlib.Path.home() / ".netrc"
            if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="ignore"):
                return True
            return False

    if _os.environ.get("EARTHDATA_TOKEN") and not _token_is_permanent():
        print("""
  ⚠  EARTHDATA_TOKEN est défini pour cette session seulement.
     Pour ne pas avoir à le redéfinir à chaque fois, enregistrez-le
     de façon permanente (remplacez <token> par votre valeur) :

     Windows — PowerShell (permanent pour l'utilisateur courant) :
       [System.Environment]::SetEnvironmentVariable(
           "EARTHDATA_TOKEN", "<token>", "User")

     Windows — invite de commandes :
       setx EARTHDATA_TOKEN "<token>"

     macOS / Linux (~/.zshrc ou ~/.bashrc) :
       echo 'export EARTHDATA_TOKEN="<token>"' >> ~/.zshrc
       source ~/.zshrc

     Alternative — ~/.netrc (username/password) :
       machine urs.earthdata.nasa.gov
       login    votre-username-earthdata
       password votre-mot-de-passe
""")
    try:
        from meandre.data.grace_loader import fetch_grace_tws
        df_grace = fetch_grace_tws(bbox, DATE_START, DATE_END)
        if df_grace.empty:
            print("  Avertissement : aucune donnée GRACE récupérée")
            print("  (Vérifier EARTHDATA_LOGIN et EARTHDATA_PASSWORD)")
        else:
            n = cache.import_grace_tws(df_grace)
            print(f"  Ingéré : {n:,} mois")
            print(f"  TWS   : {df_grace['tws_mm'].min():.1f} – "
                  f"{df_grace['tws_mm'].max():.1f} mm")
    except ImportError as e:
        print(f"  earthaccess non installé — GRACE ignoré ({e})")
        print("  Installer avec: pip install earthaccess")

print("\n── Résumé ──────────────────────────────────────────────────────────────")
for name, has_fn in [("modis_et", cache.has_modis_et),
                     ("modis_snow", cache.has_modis_snow),
                     ("modis_ndvi", cache.has_modis_ndvi),
                     ("grace_tws", cache.has_grace_tws)]:
    status = "✓" if has_fn() else "✗"
    print(f"  {status}  {name}")

print("\nTerminé. Activer dans la config avec w_nll_et / w_nll_swe > 0.")
