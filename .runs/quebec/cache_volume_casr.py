"""Cache des deux mesures de volume et de temperature du forcage, pour la presentation.

QUESTION D'ESSI, 2026-09-15 : un collegue avance que CaSR n'a pas vraiment de probleme de
volume d'eau, plutot un probleme de calendrier de fonte. Les operations de restriction de
volume sont-elles alors superflues ?

Deux mesures, aucune simulation.

1. Le FACTEUR de recalage reellement applique a la precipitation, region par region. Le
   recalage multiplie tout le canal P par une constante, cible = ecoulement observe aux
   jauges + evapotranspiration de Fu sur l'evapotranspiration potentielle d'Oudin. On
   rapporte le facteur et le NOMBRE DE JAUGES qui portent la cible, puisque c'est la
   fiabilite de la cible, et non celle de CaSR, qui est en cause la ou le facteur est
   grand.

2. Le BIAIS DE TEMPERATURE contre les stations d'Environnement Canada, par mois, et la
   date de franchissement d'un cumul de degres-jours, qui est l'indicateur direct du
   calendrier de fonte.

    .venv/Scripts/python.exe .runs/quebec/cache_volume_casr.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

SORTIE = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
NOMS = {"outv": "Outaouais aval", "gasp": "Gaspésie", "sagu": "Saguenay",
        "mont": "Montérégie", "slno": "Saint-Laurent nord-ouest",
        "slso": "Saint-Laurent sud-ouest", "abit": "Abitibi", "cnda": "Côte-Nord A",
        "cndb": "Côte-Nord B", "cndc": "Côte-Nord C", "cndd": "Côte-Nord D",
        "cnde": "Côte-Nord E", "labi": "Labrador", "outm": "Outaouais moyen",
        "vaud": "Vaudreuil"}


def main():
    os.makedirs(SORTIE, exist_ok=True)
    lignes = []
    for reg in REGIONS:
        brut = _paths.data_path("quebec", f"forcing-{reg}.nc")
        cal = _paths.data_path("quebec", f"forcing-{reg}-budyko.nc")
        base = _paths.data_path("quebec", f"{reg}.duckdb")
        if not (os.path.exists(brut) and os.path.exists(cal)):
            continue
        pa = float(xr.open_dataset(brut).forcing[:, :, 0].mean()) * 365.25
        pb = float(xr.open_dataset(cal).forcing[:, :, 0].mean()) * 365.25
        cx = duckdb.connect(base, read_only=True)
        st = cx.execute("""select s.station_id, s.drainage_area_km2 a, avg(o.discharge) q
            from stations s join observations o on s.station_id = o.station_id
            where o.date <= '2021-12-31' group by 1, 2""").fetchdf()
        cx.close()
        st = st.dropna()
        lignes.append({"region": reg, "nom": NOMS.get(reg, reg.upper()),
                       "p_casr": pa, "p_cible": pb, "facteur": pb / pa,
                       "jauges": len(st),
                       "aire_mediane_km2": float(st.a.median()) if len(st) else np.nan})
    d = pd.DataFrame(lignes)
    f = f"{SORTIE}/facteur-volume-casr.csv"
    d.to_csv(f, index=False)
    print(f"{f} : {len(d)} régions, facteur médian {d.facteur.median():.3f}")
    print(d[["nom", "p_casr", "p_cible", "facteur", "jauges"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
