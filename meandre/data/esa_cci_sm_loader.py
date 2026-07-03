"""ESA CCI Soil Moisture loader (via Copernicus CDS, accès cdsapi existant).

Humidité du sol de SURFACE satellite (m³/m³), produit COMBINÉ actif+passif qui
fusionne SMAP, SMOS, ASCAT, AMSR2. Remplace SMAP (NASA, auth indisponible) par
l'équivalent européen accessible avec ~/.cdsapirc. Contraint la partition
ruissellement/infiltration et le tarissement estival (étiage).

Dataset CDS : "satellite-soil-moisture" (global 0.25°, CDR+ICDR 1978-présent).
On télécharge le bbox-global puis on sous-échantillonne le bassin → moyenne bassin.

  python -m meandre.data.esa_cci_sm_loader   # test 1 mois sur SLSO
"""
from __future__ import annotations
import os, zipfile, glob, logging
from pathlib import Path
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)
DATASET = "satellite-soil-moisture"


def fetch_esa_cci_sm(bbox, years, aggregation="month_average",
                     cache_dir="D:/meandre-data/esa_cci_sm"):
    """Télécharge l'humidité du sol de surface ESA CCI (combiné) pour `years`,
    sous-échantillonne le bbox, retourne df [date, sm_surface (m³/m³), n_obs].
    aggregation : "month_average" (léger) ou "day_average" (lourd, ingestion finale).
    """
    import cdsapi
    os.makedirs(cache_dir, exist_ok=True)
    c = cdsapi.Client()
    rows = []
    for yr in years:
        zf = Path(cache_dir) / f"esacci_sm_{aggregation}_{yr}.zip"
        if not zf.exists() or zf.stat().st_size < 1000:
            req = {
                "variable": ["surface_soil_moisture_volumetric"],
                "type_of_sensor": ["combined"],
                "time_aggregation": [aggregation],
                "year": [str(yr)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": ([f"{d:02d}" for d in range(1, 32)] if aggregation == "daily" else ["01"]),
                "type_of_record": ["cdr"],
                "version": ["v202505"],
            }
            print(f"  [ESA CCI SM] requête CDS {yr} ({aggregation})…")
            c.retrieve(DATASET, req, str(zf))
        # dézippe + lit les netcdf
        with zipfile.ZipFile(zf) as z:
            z.extractall(Path(cache_dir) / f"_{yr}")
        import xarray as xr
        for nc in sorted(glob.glob(str(Path(cache_dir) / f"_{yr}" / "*.nc"))):
            ds = xr.open_dataset(nc)
            var = next((v for v in ("sm", "soil_moisture", "volumetric_surface_soil_moisture") if v in ds), None)
            if var is None:
                ds.close(); continue
            latd = [d for d in ds[var].dims if "lat" in d.lower()][0]
            lond = [d for d in ds[var].dims if "lon" in d.lower()][0]
            la = ds[latd].values
            sl_lat = slice(bbox[3], bbox[1]) if la[0] > la[-1] else slice(bbox[1], bbox[3])
            sub = ds[var].sel({latd: sl_lat, lond: slice(bbox[0], bbox[2])})
            t = pd.to_datetime(ds["time"].values)
            arr = sub.values.reshape(sub.shape[0], -1)
            mean = np.nanmean(arr, axis=1); nobs = np.sum(np.isfinite(arr), axis=1)
            for i in range(len(t)):
                rows.append({"date": pd.Timestamp(t[i]).to_period("M").to_timestamp(),
                             "sm_surface": float(mean[i]), "n_obs": int(nobs[i])})
            ds.close()
    df = pd.DataFrame(rows)
    if len(df):
        df = df[np.isfinite(df.sm_surface) & (df.n_obs > 0)].sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_esa_cci_sm((-73.0, 44.5, -69.6, 47.7), [2020], aggregation="month_average")
    print(df.to_string(index=False))
    if len(df):
        print(f"\nSM surface m³/m³ : {df.sm_surface.min():.3f}..{df.sm_surface.max():.3f}, "
              f"min juillet-aout (étiage) attendu")
