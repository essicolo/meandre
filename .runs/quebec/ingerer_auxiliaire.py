"""Ingère une source auxiliaire au tronçon, pour les régions demandées.

La source est décrite par `<sources>/<nom>/source.toml`. Le résultat va dans
`<derives>/auxiliaires/<nom>-troncons.parquet`, avec la nature et la version de la source dans
les métadonnées du fichier.

    .venv/Scripts/python.exe .runs/quebec/ingerer_auxiliaire.py siigsol labi mont
"""
import json
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.auxiliary import load_source
from meandre.data.auxiliary.ingest import ingest
from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud"]


def main(nom, regions):
    source = load_source(nom)
    tables = []
    for reg in regions:
        t0 = time.time()
        t = ingest(source, reg)
        couv = t[[c for c in t.columns if c.startswith("couv_")]]
        print(f"{reg}: {len(t)} tronçons, {len(couv.columns)} variables | couverture médiane {couv.median().median():.2f}, maximum {couv.max().max():.3f} | {time.time() - t0:.0f} s", flush=True)
        tables.append(t)
    d = pd.concat(tables, ignore_index=True)
    sortie = f"{_paths.DERIVED_ROOT}/auxiliaires/{nom}-troncons.parquet"
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    table = pa.Table.from_pandas(d, preserve_index=False)
    meta = {"source": nom, "nature": source.nature, "version": str(source.meta["version"]), "regions": regions}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"meandre": json.dumps(meta).encode()})
    pq.write_table(table, sortie)
    print(f"\n{sortie} : {len(d)} tronçons, {len(d.columns)} colonnes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], [a.lower() for a in sys.argv[2:]] or REGIONS))
