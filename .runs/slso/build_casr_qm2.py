"""Prétraitement CaSR par QUANTILE MAPPING sur la distribution (FORME) de quebec.zarr + VOLUME CaSR préservé.
CaSR a le meilleur TIMING (plafond r 0.76) mais une distribution biaisée (crachin
94% jours pluvieux, sur-volume, queue haute ×2). quebec.zarr a une distribution
hydrologiquement saine mais un timing médiocre (krigeage). Le QM remappe, PAR NŒUD,
la précip de CaSR sur la CDF de quebec.zarr en préservant la SÉQUENCE (donc le
timing de CaSR). Résultat : timing CaSR + distribution quebec.zarr.

qm_p(t) = interp( casr_p(t), sort(casr_p), sort(qz_p) )  — mapping quantile-à-quantile.
Seule la PRÉCIP (canal 0) est remappée ; T/Rn/u2/ea de CaSR (réanalyse, bons) gardés.
Sortie : forcing-casr-qm.nc.  CPU seulement.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr

CASR = ".runs/slso/data/forcing-casr-riox-intens.nc"   # CaSR 7 canaux (P,Tmin,Tmax,Rn,u2,ea,DT_eff)
QZ = ".runs/slso/data/forcing.nc"                       # quebec.zarr (P au canal 0)
OUT = ".runs/slso/data/forcing-casr-qm2.nc"

dc = xr.open_dataset(CASR); dz = xr.open_dataset(QZ)
tc = pd.to_datetime(dc["time"].values).normalize()
tz = pd.to_datetime(dz["time"].values).normalize()
VARS = list(dc["var"].values.astype(str))
Fc = dc["forcing"].values.copy()                       # (T,N,7)
Pc = Fc[:, :, 0]                                        # CaSR precip
Pz = dz["forcing"].values[:, :, 0]                      # quebec.zarr precip
dc.close(); dz.close()
# aligner les périodes (même grille de nœuds PHYSITEL, mêmes dates attendues)
assert Pc.shape == Pz.shape, f"shapes CaSR {Pc.shape} vs QZ {Pz.shape}"
assert (tc == tz).all(), "dates CaSR != quebec.zarr"
T, N = Pc.shape
print(f"QM : {N} nœuds, {T} jours")
print(f"AVANT : CaSR jours pluvieux {(Pc>0.1).mean()*100:.0f}%, P {Pc.mean()*365.25:.0f} mm/an "
      f"| QZ jours pluvieux {(Pz>0.1).mean()*100:.0f}%, P {Pz.mean()*365.25:.0f} mm/an")

# quantile mapping par nœud (préserve la séquence temporelle = timing CaSR)
Pqm = np.empty_like(Pc)
for n in range(N):
    cs = np.sort(Pc[:, n]); zs = np.sort(Pz[:, n])
    q = np.interp(Pc[:, n], cs, zs)                     # forme QZ, même rang (timing CaSR)
    # QM-v2 : restaurer le VOLUME propre de CaSR à ce nœud (forme QZ, volume CaSR)
    sq = q.sum()
    Pqm[:, n] = q * (Pc[:, n].sum() / sq) if sq > 0 else q
Pqm = np.clip(Pqm, 0.0, None).astype(np.float32)

# vérif : timing préservé (corrélation de rang CaSR vs QM ~1), distribution = QZ
from scipy.stats import spearmanr
_s = spearmanr(Pc[:, 0], Pqm[:, 0]).correlation
print(f"APRÈS : QM jours pluvieux {(Pqm>0.1).mean()*100:.0f}%, P {Pqm.mean()*365.25:.0f} mm/an")
print(f"  corrélation de RANG CaSR↔QM (nœud 0) : {_s:.3f} (≈1 = timing CaSR préservé)")
print(f"  distribution QM vs QZ : moyenne {Pqm.mean():.3f} vs {Pz.mean():.3f} mm/j (≈ = distribution QZ acquise)")

Fc[:, :, 0] = Pqm
out = xr.Dataset({"forcing": (("time", "node", "var"), Fc.astype(np.float32))},
                 coords={"time": tc.values, "node": np.arange(N), "var": VARS})
if os.path.exists(OUT): os.remove(OUT)
out.to_netcdf(OUT, engine="h5netcdf"); out.close()
print(f"[ok] écrit {OUT}")
