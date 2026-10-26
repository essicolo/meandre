"""Compare des passes à poids gelés d'un territoire (sorties ETL_DUMP_Q et journal du pilote) :
perte, KGE médian, terme d'étiage, et décomposition de l'été (ligne de base, événements, moyenne,
simulé sur observé, médiane des étés et des stations).

    .venv/Scripts/python.exe experiments/plan_ml/passes-ete.py slso sol temoin fc055 fc055por075
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

spec = importlib.util.spec_from_file_location("es", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ete-sud.py"))
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)
DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"


def main(reg, prefixe, tags):
    lignes = []
    for tag in tags:
        txt = open(f"{DOSSIER}/{prefixe}-{reg}-{tag}.log", encoding="utf-8", errors="replace").read()
        kge = re.search(rf"HELD-OUT [0-9-]+ {reg}: n=\d+ \| médian ([0-9.]+)", txt)
        pt = re.search(rf"PERTE [0-9-]+ {reg} .*?: total ([0-9.]+).*?etiage ([0-9.]+)", txt)
        mois = re.search(r"simule/observe par mois : (.*)", txt)
        z = np.load(f"{DOSSIER}/q-{prefixe}-{reg}-{tag}.npz", allow_pickle=True)
        t = pd.DatetimeIndex(z["dates"])
        dec = np.array([es.decomposition(pd.Series(z["q_sim"][:, k], index=t), pd.Series(z["q_obs"][:, k], index=t)) for k in range(z["q_sim"].shape[1])])
        lignes.append(dict(variante=tag, perte=float(pt.group(1)) if pt else np.nan, etiage=float(pt.group(2)) if pt else np.nan, kge=float(kge.group(1)) if kge else np.nan, ete_base=np.nanmedian(dec[:, 0]), ete_evenements=np.nanmedian(dec[:, 1]), ete_moyenne=np.nanmedian(dec[:, 2]), mois=mois.group(1).strip() if mois else ""))
    L = pd.DataFrame(lignes).set_index("variante")
    L.to_csv(f"{DOSSIER}/passes-{prefixe}-{reg}.csv")
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(L.drop(columns="mois").round(4).to_string())
        print(L["mois"].to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
