"""Aires de drainage aux jauges : HydroSHEDS vs D8 actuel vs rapporté (juste).

Échantillonne les DEUX rasters d'accumulation (D8 Copernicus actuel et
HydroSHEDS conditionné) au même snap métrique près de chaque jauge, et compare
à l'aire rapportée. Si HydroSHEDS reproduit mieux, le réseau open-data actuel
est faux et la reconstruction se justifie.

  python .runs/slso-od/compare_network_areas.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math
import numpy as np
import duckdb
import rasterio
import pyflwdir

HS_DIR = "D:/meandre-data/hydrosheds/slso_dir_3s.tif"
D8_ACC = "D:/meandre-data/geo_cache/slso-od/acc.npy"
D8_GRID = "D:/meandre-data/geo_cache/slso-od/dem_routing.tif"
BASIN_DB = ".runs/slso-od/data/basin.duckdb"
SNAP_M = 600.0  # rayon de snap métrique identique pour les deux

con = duckdb.connect(BASIN_DB, read_only=True)
st = con.execute("""
    SELECT station_id, lon, lat, drainage_area_km2 AS area_report
    FROM stations
    WHERE drainage_area_km2 IS NOT NULL AND drainage_area_km2 > 0
""").fetchdf()
con.close()
print(f"{len(st)} jauges avec aire rapportée\n", flush=True)


def cell_area_km2(res_deg, lat):
    dlat = res_deg * 111_000.0
    dlon = res_deg * 111_000.0 * math.cos(math.radians(lat))
    return dlat * dlon / 1e6


class AccGrid:
    def __init__(self, area_arr, transform, res_deg):
        self.a = area_arr            # aire amont en km² par cellule
        self.tr = transform
        self.res_deg = res_deg
    def area_at(self, lon, lat):
        c, r = ~self.tr * (lon, lat)
        r0, c0 = int(r), int(c)
        px = max(1, int(SNAP_M / (self.res_deg * 111_000.0)))
        H, W = self.a.shape
        r1, r2 = max(0, r0 - px), min(H, r0 + px + 1)
        c1, c2 = max(0, c0 - px), min(W, c0 + px + 1)
        sub = self.a[r1:r2, c1:c2]
        return float(np.nanmax(sub)) if sub.size else float("nan")


# D8 : acc en cellules → km².
acc_d8 = np.load(D8_ACC)
with rasterio.open(D8_GRID) as s:
    tr_d8 = s.transform; res_d8 = abs(s.transform.a)
lat_mid = float(st["lat"].mean())
d8 = AccGrid(acc_d8 * cell_area_km2(res_d8, lat_mid), tr_d8, res_d8)

# HydroSHEDS : aire amont km² via pyflwdir.
with rasterio.open(HS_DIR) as s:
    d8dir = s.read(1); tr_hs = s.transform; res_hs = abs(s.transform.a)
flw = pyflwdir.from_array(d8dir, ftype="d8", transform=tr_hs, latlon=True, cache=True)
hs = AccGrid(np.asarray(flw.upstream_area(unit="km2")), tr_hs, res_hs)

print(f"{'station':>10} {'rapporté':>9} {'D8':>9} {'HydroSHEDS':>11} {'errD8%':>8} {'errHS%':>8}", flush=True)
eD8, eHS = [], []
for _, x in st.iterrows():
    ad8 = d8.area_at(x.lon, x.lat)
    ahs = hs.area_at(x.lon, x.lat)
    e8 = 100 * (ad8 - x.area_report) / x.area_report
    eh = 100 * (ahs - x.area_report) / x.area_report
    eD8.append(abs(e8)); eHS.append(abs(eh))
print(f"\nErreur absolue MÉDIANE  |  D8 = {np.median(eD8):.1f}%   HydroSHEDS = {np.median(eHS):.1f}%", flush=True)
print(f"Erreur absolue MOYENNE  |  D8 = {np.mean(eD8):.1f}%   HydroSHEDS = {np.mean(eHS):.1f}%", flush=True)
print(f"Médiane |err| <20% : D8 {np.mean(np.array(eD8)<20)*100:.0f}%   HS {np.mean(np.array(eHS)<20)*100:.0f}% des jauges", flush=True)
print("DONE", flush=True)
