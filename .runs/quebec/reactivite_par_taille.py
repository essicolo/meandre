"""Le pas journalier est-il la limite sur les petits bassins ?

Question d'Essi, 2026-09-15 : une frequence adaptative aurait-elle un sens pour les petits
bassins reactifs ? Avant d'en discuter, il faut savoir si le pas d'un jour est deja trop
grossier pour eux. On le mesure sur l'OBSERVE seul, sans le modele, donc la reponse ne
depend d'aucun choix de modelisation.

Deux indicateurs par station, sur le debit observe journalier :
  la duree de MONTEE des crues annuelles, du creux precedent au sommet. Si elle vaut un
    jour, le pas journalier ne voit qu'un seul point de la montee et ne peut pas la decrire ;
  l'autocorrelation a un jour des variations, qui dit si la serie journaliere est lisse ou
    si elle saute d'un jour a l'autre.

    .venv/Scripts/python.exe .runs/quebec/reactivite_par_taille.py
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


def main():
    racine = _paths.DATA_ROOT
    lignes = []
    for reg in REGIONS:
        base = f"{racine}/quebec/{reg}.duckdb"
        if not os.path.exists(base):
            continue
        cx = duckdb.connect(base, read_only=True)
        st = cx.sql("""select s.station_id, s.drainage_area_km2 a from stations s
            where s.drainage_area_km2 is not null""").fetchdf()
        for r in st.itertuples():
            o = cx.sql(f"""select date, discharge from observations
                where station_id = '{r.station_id}' and discharge is not null
                order by date""").fetchdf()
            if len(o) < 2000:
                continue
            q = o.discharge.values.astype(float)
            t = pd.DatetimeIndex(o.date)
            if np.nanmean(q) <= 0:
                continue
            # Duree de montee du plus fort evenement de chaque annee.
            durees = []
            for an in np.unique(t.year):
                m = np.asarray(t.year == an)
                if m.sum() < 300:
                    continue
                qa = q[m]
                i = int(np.nanargmax(qa))
                j = i
                while j > 0 and qa[j - 1] < qa[j]:
                    j -= 1
                if i > j:
                    durees.append(i - j)
            if len(durees) < 5:
                continue
            dq = np.diff(q)
            ac = float(np.corrcoef(dq[:-1], dq[1:])[0, 1]) if len(dq) > 100 else np.nan
            lignes.append({"region": reg, "station": str(r.station_id), "aire_km2": float(r.a),
                           "montee_j": float(np.median(durees)),
                           "part_montee_1j": float(np.mean(np.asarray(durees) <= 1)),
                           "autocorr_variation": ac})
        cx.close()
    d = pd.DataFrame(lignes)
    if d.empty:
        print("aucune station")
        return 1
    print(f"{len(d)} stations, débit observé journalier, 2001-2024\n")
    print(f"{'aire drainée':>16} {'n':>4} {'montée médiane':>16} "
          f"{'part des crues montant en 1 j':>30} {'autocorr. des variations':>26}")
    for lo, hi in ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 1e9)):
        q = d[(d.aire_km2 >= lo) & (d.aire_km2 < hi)]
        if len(q) < 3:
            continue
        et = f"{lo:.0f}-{hi:.0f} km²" if hi < 1e9 else f"plus de {lo:.0f} km²"
        print(f"{et:>16} {len(q):4d} {q.montee_j.median():14.1f} j "
              f"{100 * q.part_montee_1j.median():29.0f}% {q.autocorr_variation.median():26.3f}")
    from scipy.stats import spearmanr
    la = np.log10(d.aire_km2.values)
    for col in ("montee_j", "part_montee_1j", "autocorr_variation"):
        rho, p = spearmanr(la, d[col].values, nan_policy="omit")
        print(f"\ncorrélation de rang avec le log de l'aire, {col} : rho {rho:+.3f} | p {p:.4f}")
    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/reactivite-par-taille.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
