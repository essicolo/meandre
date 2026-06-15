"""Diagnostic état theta par couche : valeurs simulées, saturation relative,
moyennes saisonnières. Confirme le pattern L1 saturée / L2-L3 sèches."""
import torch, numpy as np, pandas as pd, xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.routing.withdrawals import WithdrawalData

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = ".runs/slso/checkpoints/best-kendall-gal-v3-phase2-boxcox-nll.pt"
init = torch.load(CK, map_location="cpu", weights_only=False)["init_kwargs"]
model = HydroModel(**init).to(dev); model.load(CK); model.eval()

cache = BasinCache(".runs/slso/data/slso.duckdb"); h = cache.load(device=dev)
nc, terr = h["node_coords"], h["territorial"]
n_nodes = h["n_nodes"]

# Forcing (proxy cache existant)
ds = xr.open_dataset(".runs/slso/data/forcing.nc")
dates = pd.to_datetime(ds["time"].values)
forcing = torch.from_numpy(ds["forcing"].values.astype(np.float32)).to(dev)
ds.close()
doy = torch.tensor(dates.dayofyear.values, dtype=torch.long, device=dev)

with torch.no_grad():
    sp = model.spatial_encoder(nc, terr.to_tensor())
    _, _, diag = model.simulate(
        forcing=forcing, initial_state=HydroState.zeros(n_nodes, device=dev),
        graph=h["graph"], node_coords=nc, territorial=terr,
        withdrawals=WithdrawalData.zeros(forcing.shape[0], n_nodes, device=dev),
        day_of_year=doy, return_diagnostics=True,
    )

# Params par couche (moyenne-bassin)
def m(x): return float(x.detach().mean())
po = [m(sp.porosity_1), m(sp.porosity_2), m(sp.porosity_3)]
fc = [m(sp.theta_fc_1), m(sp.theta_fc_2), m(sp.theta_fc_3)]
wp = [m(sp.theta_wp_1), m(sp.theta_wp_2), m(sp.theta_wp_3)]

theta = [diag.theta1.cpu().numpy(), diag.theta2.cpu().numpy(), diag.theta3.cpu().numpy()]  # (T,N)
season = pd.Series(dates).dt.month.map(
    {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}).values

print(f"{'='*72}\nÉTAT THETA SIMULÉ (moyenne-bassin) — porosité / capacité champ / point flétr.\n{'='*72}")
for i in range(3):
    L = i + 1
    tm = np.nanmean(theta[i])  # moyenne globale
    sat = (tm - wp[i]) / (po[i] - wp[i] + 1e-9)  # saturation relative [0=wp, 1=porosité]
    print(f"\nCouche {L}:  θ_moy={tm:.3f}   [wp={wp[i]:.3f}  fc={fc[i]:.3f}  porosité={po[i]:.3f}]   "
          f"saturation relative={sat:.2f}")
    # moyennes saisonnières + saturation
    print("   saison ", "  ".join(f"{s:>6s}" for s in ["DJF","MAM","JJA","SON"]))
    th_s = [np.nanmean(theta[i][season==s]) for s in ["DJF","MAM","JJA","SON"]]
    sa_s = [(t - wp[i])/(po[i]-wp[i]+1e-9) for t in th_s]
    print("   θ      ", "  ".join(f"{t:6.3f}" for t in th_s))
    print("   sat.   ", "  ".join(f"{s:6.2f}" for s in sa_s))
