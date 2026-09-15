"""La cible de correction du volume est-elle gonflee par l'eau ajoutee par les activites ?

La correction de volume multiplie la precipitation de CaSR par le facteur qui ferme le
bilan P = Q + ET(P), ou Q est l'ecoulement OBSERVE aux stations. Or l'observation d'une
station situee en aval de rejets porte de l'eau qui n'est pas tombee en pluie sur son
bassin. La cible est alors trop haute, et la correction ajoute de la precipitation pour
expliquer de l'eau d'origine anthropique.

Mesure, sans simulation : pour chaque station, on cumule le bilan anthropique net de tous
les troncons de son bassin versant, en remontant le graphe, et on le retire de son debit
observe. On compare l'ecoulement specifique median avant et apres, puisque c'est la
mediane des stations qui fixe la cible.

    .venv/Scripts/python.exe .runs/quebec/cible_budyko_anthropique.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
SORTIE = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")


def amont(n_nodes, src, dst):
    """Ensembles de noeuds en amont de chaque noeud, par remontee dans l'ordre topologique."""
    enfants = [[] for _ in range(n_nodes)]
    for a, b in zip(src, dst):
        enfants[b].append(a)
    return enfants


def main():
    lignes = []
    for reg in REGIONS:
        base = _paths.data_path("quebec", f"{reg}.duckdb")
        if not os.path.exists(base):
            continue
        cx = duckdb.connect(base, read_only=True)
        nd = cx.sql("select node_idx, topo_order from nodes order by node_idx").fetchdf()
        ed = cx.sql("select src, dst from edges").fetchdf()
        try:
            w = cx.sql("""select node_idx, avg(net_surface + net_gw) net
                from withdrawals group by 1""").fetchdf()
        except Exception:
            w = pd.DataFrame(columns=["node_idx", "net"])
        st = cx.sql("""select s.station_id, s.node_idx, s.drainage_area_km2 a,
            avg(o.discharge) q from stations s join observations o
            on s.station_id = o.station_id where o.date <= '2021-12-31'
            group by 1, 2, 3""").fetchdf()
        cx.close()
        st = st.dropna(subset=["a", "q", "node_idx"])
        if st.empty:
            continue
        n = len(nd)
        net = np.zeros(n)
        if len(w):
            net[w.node_idx.values.astype(int)] = w.net.values
        # Cumul amont : on parcourt les noeuds du plus amont au plus aval.
        enf = amont(n, ed.src.values.astype(int), ed.dst.values.astype(int))
        ordre = nd.sort_values("topo_order").node_idx.values.astype(int)
        cum = net.copy()
        for j in ordre:
            for k in enf[j]:
                cum[j] += cum[k]
        st["net_amont"] = cum[st.node_idx.values.astype(int)]
        st["ecoul"] = st.q * 31_557_600.0 / (st.a * 1e6) * 1000.0
        st["ecoul_nat"] = (st.q - st.net_amont) * 31_557_600.0 / (st.a * 1e6) * 1000.0
        lignes.append({"region": reg, "jauges": len(st),
                       "net_regional": float(net.sum()),
                       "ecoul_med": float(st.ecoul.median()),
                       "ecoul_med_nat": float(st.ecoul_nat.median()),
                       "part_jauges_touchees": float((st.net_amont.abs() > 0.01).mean())})
    d = pd.DataFrame(lignes)
    d["ecart_mm"] = d.ecoul_med_nat - d.ecoul_med
    d["ecart_pct"] = 100.0 * d.ecart_mm / d.ecoul_med
    print(f"{'région':>7} {'jauges':>7} {'net m³/s':>9} {'écoul. mm/an':>13} "
          f"{'sans apport':>12} {'écart':>8} {'jauges touchées':>16}")
    for r in d.itertuples():
        print(f"{r.region:>7} {r.jauges:7d} {r.net_regional:+9.2f} {r.ecoul_med:13.0f} "
              f"{r.ecoul_med_nat:12.0f} {r.ecart_pct:+7.1f}% {100 * r.part_jauges_touchees:15.0f}%")
    if os.path.isdir(SORTIE):
        d.to_csv(f"{SORTIE}/cible-budyko-anthropique.csv", index=False)
        print(f"cache : {SORTIE}/cible-budyko-anthropique.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
