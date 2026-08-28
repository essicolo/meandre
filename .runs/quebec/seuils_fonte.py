"""Que fait vraiment le seuil de fonte calibre d'Hydrotel sur le forcage ?

Le calage d'Hydrotel porte un SEUIL DE FONTE PAR CLASSE D'OCCUPATION, lu dans
degre_jour_modifie.csv. Sous conifere il vaut +3.35 degres sur sagu/outv/abit et +2.26 sur
gasp/mont/slso. La neige sous couvert resineux ne fond donc PAS tant que l'air n'a pas
depasse ce seuil, quelle que soit la radiation. On mesure ici combien de jours d'hiver le
franchissent, donc a quel point ce seuil verrouille le manteau, et on le compare au seuil
physique (0 degre) et au prior de litterature que meandre utilise quand il apprend (-0.5).
"""
import numpy as np
import xarray as xr

SEUILS = {"sagu": (3.3518, 0.4034, -2.5450, 7.1968, 8.5248, 10.0978),
          "outv": (3.3518, 0.4034, -2.5450, 7.1968, 8.5248, 10.0978),
          "abit": (3.3518, 0.4034, -2.5450, 7.1968, 8.5248, 10.0978),
          "gasp": (2.2641, 1.9156, 1.5671, 4.5233, 9.0466, 18.0932),
          "mont": (2.2641, 1.9156, 1.5671, 4.5233, 9.0466, 18.0932),
          "slso": (2.2641, 1.9156, 1.5671, 4.5233, 9.0466, 18.0932)}

for reg, (sc, sf, sd, tc, tf, td) in SEUILS.items():
    try:
        ds = xr.open_dataset(f"D:/meandre-data/quebec/forcing-{reg}-hyb.nc")
    except Exception as e:
        print(f"[{reg}] {e}")
        continue
    fo = ds["forcing"]
    iv = list(ds["var"].values)
    tmin = fo[:, :, iv.index("Tmin")].mean("node").values
    tmax = fo[:, :, iv.index("Tmax")].mean("node").values
    tm = 0.5 * (tmin + tmax)
    mois = ds["time"].dt.month.values
    hiver = np.isin(mois, [12, 1, 2, 3])
    print("")
    print(f"[{reg}] seuils conif {sc:+.2f} feu {sf:+.2f} dec {sd:+.2f} | "
          f"taux {tc:.1f}/{tf:.1f}/{td:.1f} mm/j/degC")
    print("    seuil                  | dec-mars   avril     mai   (% de jours ou l'air depasse)")
    for nom, s_ in (("conifere Hydrotel", sc), ("feuillu Hydrotel", sf),
                    ("decouvert Hydrotel", sd), ("physique 0", 0.0),
                    ("meandre prior -0.5", -0.5)):
        a = float((tm[hiver] > s_).mean() * 100)
        b = float((tm[mois == 4] > s_).mean() * 100)
        c = float((tm[mois == 5] > s_).mean() * 100)
        print(f"    {nom:>22} | {a:7.1f}% {b:6.1f}% {c:6.1f}%")
