"""Recale le canal P d'un forçage régional sur la cible Budyko-Fu (lame obs + ET_Fu(PET Oudin)).
Remplace le 'lame + 450' du builder v1 (ETR boréale inadaptée au sud agricole / à la Côte-Nord).
  python .runs/quebec/rescale_forcing_budyko.py MONT [...]
Sortie : forcing-<reg>-budyko.nc (P rescalé, autres canaux inchangés).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb
from meandre.data.basin_cache import BasinCache
_LAMBDA = 2.45
def re_ext(doy, lat_deg):
    lat = np.radians(lat_deg)
    dr = 1 + 0.033*np.cos(2*np.pi*doy/365.25)
    dec = 0.409*np.sin(2*np.pi*doy/365.25 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat)*np.tan(dec), -1, 1))
    return 37.586*dr*(ws*np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.sin(ws))
for reg in [a.lower() for a in sys.argv[1:]]:
    nc_p = f"D:/meandre-data/quebec/forcing-{reg}.nc"; db = f"D:/meandre-data/quebec/{reg}.duckdb"
    d = xr.open_dataset(nc_p); F = d["forcing"].values.copy(); V = list(d["var"].values.astype(str))
    t = pd.to_datetime(d["time"].values); d.close()
    lat = BasinCache(db).load(device="cpu")["node_coords"][:, 1].numpy().mean()
    Tm = (F[:, :, V.index("Tmin")] + F[:, :, V.index("Tmax")]).mean(axis=1)/2.0
    pet = float(np.clip(re_ext(t.dayofyear.values, lat)/_LAMBDA*(Tm+5.0)/100.0, 0, None).mean()*365.25)
    c = duckdb.connect(db, read_only=True)
    st = c.execute("""SELECT s.drainage_area_km2 a, AVG(o.discharge) q FROM stations s
                      JOIN observations o ON s.station_id=o.station_id WHERE o.date<='2021-12-31'
                      GROUP BY s.station_id, s.drainage_area_km2""").fetchdf(); c.close()
    lame = float((st.dropna().q*31_557_600.0/(st.dropna().a*1e6)*1000.0).median())
    P = lame + 450.0
    for _ in range(80):
        phi = pet/P; et = P*(1+phi-(1+phi**2.6)**(1/2.6)); P2 = lame+et
        if abs(P2-P) < 0.1: break
        P = P2
    cur = F[:, :, 0].mean()*365.25
    F[:, :, 0] = (F[:, :, 0]*(P/cur)).astype(np.float32)
    out = f"D:/meandre-data/quebec/forcing-{reg}-budyko.nc"
    if os.path.exists(out): os.remove(out)
    xr.Dataset({"forcing": (("time", "node", "var"), F)},
               coords={"time": t, "node": np.arange(F.shape[1]), "var": V}).to_netcdf(out)
    print(f"[{reg}] lame {lame:.0f} + ET_Fu {et:.0f} -> P {cur:.0f}->{P:.0f} mm/an : {out}")
