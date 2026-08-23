"""Corrélations spatiales et temporelles des forçages CaSR vs la grille SIMAT (quebec.zarr).
Diagnostic RAPIDE avant toute reconstruction météo : où CaSR (brut et corrigé) diverge-t-il
de la grille krigée du ministère (référence timing/volume aux jauges) ?
  TEMPOREL : corrélation Pearson par nœud (P journalier), aux lags -1/0/+1 (timing),
             par saison (DJF/MAM/JJA/SON), médiane sur les nœuds.
  SPATIAL  : corrélation entre nœuds pour chaque jour humide (moyenne SIMAT > 1 mm),
             médiane sur les jours — mesure si les ORAGES sont au bon endroit.
  VOLUME   : mm/an, fraction de jours pluvieux (>0.1), ratio par saison.
CPU seulement.  python .runs/slso/_cmp_simat.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, xarray as xr

REF = ".runs/slso/data/forcing.nc"  # quebec.zarr = grille SIMAT/GCQ aux noeuds
CANDS = {"casr-brut": ".runs/slso/data/forcing-casr-riox-intens.nc",
         "casr-corr": "D:/meandre-data/slso/forcing-casr-corr.nc",
         "casr-merge": "D:/meandre-data/slso/forcing-casr-merge.nc"}
CANDS = {k: v for k, v in CANDS.items() if os.path.exists(v)}

def load_p(path):
    ds = xr.open_dataset(path)
    t = pd.to_datetime(ds["time"].values).normalize()
    p = ds["forcing"].values[:, :, 0].astype(np.float64)
    ds.close()
    return pd.DatetimeIndex(t), p

tr, Pr = load_p(REF)
print(f"SIMAT (quebec.zarr) : {Pr.shape[1]} noeuds, {tr[0].date()} -> {tr[-1].date()}, {Pr.mean()*365.25:.0f} mm/an, jours pluvieux {(Pr>0.1).mean()*100:.0f}%")

def seasons(idx):
    m = idx.month
    return {"DJF": np.isin(m, [12, 1, 2]), "MAM": np.isin(m, [3, 4, 5]),
            "JJA": np.isin(m, [6, 7, 8]), "SON": np.isin(m, [9, 10, 11])}

def corr_cols(a, b):
    """Pearson colonne par colonne (T, N) sans boucle python sur N."""
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / den

for name, path in CANDS.items():
    tc, Pc = load_p(path)
    common = tr.intersection(tc)
    ir = tr.get_indexer(common); ic = tc.get_indexer(common)
    R = Pr[ir]; C = Pc[ic]
    print(f"\n=== {name} vs SIMAT ({len(common)} jours communs {common[0].date()} -> {common[-1].date()}) ===")
    print(f"volume : {C.mean()*365.25:.0f} vs {R.mean()*365.25:.0f} mm/an (ratio {C.mean()/R.mean():.2f}) | jours pluvieux {(C>0.1).mean()*100:.0f}% vs {(R>0.1).mean()*100:.0f}%")
    # temporel par noeud, lags -1/0/+1 (lag>0 = candidat EN RETARD sur SIMAT)
    for lag in (-1, 0, 1):
        cc = corr_cols(R[max(0, lag):len(R)-max(0, -lag)], C[max(0, -lag):len(C)-max(0, lag)])
        print(f"corr temporelle par noeud, lag {lag:+d} : mediane {np.nanmedian(cc):.3f} (q25 {np.nanpercentile(cc, 25):.3f}, q75 {np.nanpercentile(cc, 75):.3f})")
    for sn, sm in seasons(common).items():
        cc = corr_cols(R[sm], C[sm])
        print(f"  {sn} : corr mediane {np.nanmedian(cc):.3f} | volume ratio {C[sm].mean()/max(R[sm].mean(), 1e-9):.2f}")
    # spatial : corr entre noeuds, jours humides SIMAT (lame moyenne > 1 mm)
    wet = R.mean(1) > 1.0
    Rw = R[wet]; Cw = C[wet]
    a = Rw - Rw.mean(1, keepdims=True); b = Cw - Cw.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = (a * b).sum(1) / den
    print(f"corr SPATIALE par jour humide ({wet.sum()} jours) : mediane {np.nanmedian(sc):.3f} (q25 {np.nanpercentile(sc, 25):.3f}, q75 {np.nanpercentile(sc, 75):.3f})")
    # gros evenements : les 20 plus grosses lames SIMAT, le candidat les voit-il le meme jour ?
    top = np.argsort(R.mean(1))[-20:]
    hits = sum(1 for i in top if C[max(0, i-1):i+2].mean(1).max() >= 0.5 * R[i].mean())
    print(f"top-20 orages SIMAT : {hits}/20 vus par {name} a +/-1 jour (>=50% de la lame)")
