"""GRACE multi-mascon : combine plusieurs solutions de mascon (JPL, CSR, et GSFC à
venir) pour obtenir une anomalie de stockage TWS robuste ET son incertitude
HONNÊTE = l'écart entre produits. Au lieu de choisir un mascon arbitraire, on
utilise l'enveloppe : moyenne = obs, std inter-produits = sigma de la contrainte
GRACE. Ce sigma alimente directement tws_anomaly_loss (au lieu du 25 mm fixe) et
la tête probabiliste.

Produits :
  - JPL Mascon RL06.3 (PO.DAAC / earthaccess) — déjà dans grace_loader.fetch_grace_tws.
  - CSR Mascon RL0603 (UT Austin, téléchargement direct public) — ci-dessous.
  - GSFC Mascon (format par-élément, TODO).

  python -m meandre.data.grace_multimascon   # auto-test sur le bbox SLSO
"""
from __future__ import annotations
import logging, os, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

# CSR RL0603 mascon grillé (0.25°), inclut GRACE + GRACE-FO. Public, sans auth.
CSR_MASCON_URL = ("https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/"
                  "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc")
CM_TO_MM = 10.0


def fetch_grace_csr(bbox, date_start, date_end, cache_dir="D:/meandre-data/grace"):
    """Télécharge (cache) le mascon CSR grillé, sous-échantillonne le bbox, retourne
    un df mensuel [date, tws_mm]. lon en 0-360 ou -180-180 géré."""
    os.makedirs(cache_dir, exist_ok=True)
    fp = Path(cache_dir) / "CSR_RL0603_mascons.nc"
    if not fp.exists() or fp.stat().st_size < 1_000_000:
        print(f"  [CSR] téléchargement {CSR_MASCON_URL} -> {fp} (gros fichier, une fois)…")
        urllib.request.urlretrieve(CSR_MASCON_URL, fp)
    import xarray as xr
    # CSR n'inscrit pas les units de time (valeurs = jours depuis 2002-01-01).
    ds = xr.open_dataset(fp, decode_times=False)
    var = next((v for v in ("lwe_thickness", "LWE_thickness", "lwe") if v in ds), None)
    if var is None:
        raise ValueError(f"[CSR] variable TWS introuvable ({list(ds.data_vars)[:5]})")
    latd = [d for d in ds[var].dims if "lat" in d.lower()][0]
    lond = [d for d in ds[var].dims if "lon" in d.lower()][0]
    lon = ds[lond].values
    if lon.max() > 180:
        sub = ds[var].sel({latd: slice(bbox[1], bbox[3]), lond: slice(bbox[0] % 360, bbox[2] % 360)})
    else:
        sub = ds[var].sel({latd: slice(bbox[1], bbox[3]), lond: slice(bbox[0], bbox[2])})
    tc = next((c for c in ("time", "TIME") if c in ds.coords), None)
    units = ds[tc].attrs.get("units") or "days since 2002-01-01"
    ref = units.split("since")[-1].strip()
    t = pd.to_datetime(ref) + pd.to_timedelta(ds[tc].values, unit="D")
    vals = np.nanmean(sub.values.reshape(sub.shape[0], -1), axis=1) * CM_TO_MM
    ds.close()
    df = pd.DataFrame({"date": pd.to_datetime(t).to_period("M").to_timestamp(), "tws_mm": vals})
    m = (df.date >= pd.Timestamp(date_start)) & (df.date <= pd.Timestamp(date_end))
    return df[m].reset_index(drop=True)


def fetch_grace_multimascon(bbox, date_start, date_end):
    """Combine JPL + CSR (+ GSFC à venir). Retourne df mensuel [date, tws_mm (moyenne
    des produits, RE-CENTRÉE sur baseline commune), tws_sigma (écart inter-produits,
    plancher 10 mm), n_products, quality_ok]. tws_sigma = incertitude honnête."""
    from meandre.data.grace_loader import fetch_grace_tws
    prods = {}
    try:
        jpl = fetch_grace_tws(bbox, date_start, date_end)[["date", "tws_mm"]]
        prods["JPL"] = jpl.set_index("date")["tws_mm"]
    except Exception as e:
        logger.warning(f"[multimascon] JPL échec : {e}")
    try:
        csr = fetch_grace_csr(bbox, date_start, date_end)
        prods["CSR"] = csr.set_index("date")["tws_mm"]
    except Exception as e:
        logger.warning(f"[multimascon] CSR échec : {e}")
    if not prods:
        return pd.DataFrame(columns=["date", "tws_mm", "tws_sigma", "n_products", "quality_ok"])
    wide = pd.DataFrame(prods)
    # Re-centrer chaque produit sur une baseline commune (chacun a sa période de réf).
    common = wide.index[(wide.index >= "2004-01-01") & (wide.index <= "2009-12-31")]
    if len(common) > 12:
        wide = wide - wide.loc[wide.index.isin(common)].mean()
    out = pd.DataFrame({
        "date": wide.index,
        "tws_mm": wide.mean(axis=1).values,
        "tws_sigma": np.clip(wide.std(axis=1).values, 10.0, None),   # plancher 10 mm
        "n_products": wide.notna().sum(axis=1).values,
    })
    out["quality_ok"] = np.isfinite(out["tws_mm"]) & (out["n_products"] >= 1)
    return out.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bbox = (-73.0, 44.5, -69.6, 47.7)   # SLSO
    df = fetch_grace_multimascon(bbox, "2002-01-01", "2024-12-31")
    print(df.head(12).to_string(index=False))
    print(f"\n{len(df)} mois | sigma médian {df['tws_sigma'].median():.1f} mm | "
          f"n_products médian {int(df['n_products'].median())}")
