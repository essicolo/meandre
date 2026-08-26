"""Ingere les prelevements et rejets provinciaux dans les bases regionales.

CONSTAT du 2026-08-21 : AUCUNE des 15 bases regionales n'a de table `withdrawals`.
Le passage a l'echelle du Quebec tourne donc SANS prelevements ni rejets, alors que
le banc SLSO en a 92 486 lignes. Sur SLSO, le terme net de surface vaut ~+7.9 m3/s a
l'echelle du bassin contre un debit moyen de 19.4 m3/s aux jauges : c'est du meme
ordre de grandeur que ce qu'on cherche a predire.

Ce que cela coute, au-dela du score : le champion a ete AJUSTE sans ces termes, donc
il les a absorbes dans ses parametres. Une conductivite, une recharge ou un facteur de
fonte compense silencieusement une prise d'eau ou un rejet. C'est exactement ce que
l'identifiabilite, promesse centrale du projet, doit empecher.

La donnee EXISTE et couvre la province : data/io-eau-meandre.parquet, 1 082 964 lignes,
2001-2024 au pas mensuel, lon -79.5 a -57.1, lat 45.0 a 53.8, avec IDTRONCON deja
apparie. Seule l'ingestion vers les bases regionales n'avait jamais ete faite.

  python .runs/quebec/ingest_withdrawals.py outv
  python .runs/quebec/ingest_withdrawals.py toutes
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import duckdb
import pandas as pd

from meandre.data.basin_cache import BasinCache
from meandre.data.hydrotel_calib import id_local
from meandre.utils import paths as _paths

# SOURCE : le derive d'io-eau, surchargeable par IO_EAU. La copie locale `data/` etait
# une COPIE DATEE ; on lit desormais le derive a la source par defaut, pour ne plus
# reingerer un instantane perime sans s'en apercevoir. La version du 2026-06-10 corrige
# un mauvais appariement majeur : l'emissaire de la rive sud de Montreal (32 m3/s) etait
# porte par un troncon de 13 km2, soit 1857 L/s/km2 -- cent fois le debit specifique
# d'un cours d'eau quebecois. Il revient sur MONT00002, l'axe principal (38 931 km2),
# ou il vaut 0.8 L/s/km2.
SOURCE = os.environ.get(
    "IO_EAU",
    "C:/Users/parse01/documents-locaux/GitHub/io-eau/data/derived/io-eau-meandre.parquet")
if not os.path.exists(SOURCE):
    SOURCE = "data/io-eau-meandre.parquet"
REGIONS = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont",
           "outm", "outv", "sagu", "slno", "slso", "vaud"]

cible = (REGIONS if (len(sys.argv) > 1 and sys.argv[1].lower() == "toutes")
         else [(sys.argv[1] if len(sys.argv) > 1 else "outv").lower()])

df = pd.read_parquet(SOURCE)
df["date"] = pd.to_datetime(df["date"])
print(f"[prelev] source : {len(df):,} lignes, {df.site_id.nunique():,} sites, "
      f"{df.date.min().date()} -> {df.date.max().date()}")

for reg in cible:
    db = _paths.data_path("quebec", f"{reg}.duckdb")
    if not os.path.exists(db):
        print(f"[prelev] {reg}: base absente")
        continue
    con = duckdb.connect(db, read_only=True)
    noeuds = con.execute("SELECT node_idx, node_id FROM nodes ORDER BY node_idx").df()
    con.close()
    # APPARIEMENT EXACT par l'identifiant de troncon, PAS par coordonnees. Le fichier
    # porte IDTRONCON au format provincial ("OUTV03387") et la base porte l'entier
    # local : la conversion est celle de hydrotel_calib, ecrite pour la dette #1 du
    # registre (trois numerotations de troncons). Un rattachement au noeud le plus
    # proche a 10 km ecartait 879 sites sur 1 661 et en placait d'autres a 6.7 km,
    # alors que l'appariement exact existe.
    sel = df[df.IDTRONCON.str.upper().str.startswith(reg.upper())].copy()
    if sel.empty:
        print(f"[prelev] {reg}: aucun point porte cet identifiant de region")
        continue
    sel["troncon"] = sel.IDTRONCON.map(lambda x: id_local(x)[1])
    corr = dict(zip(noeuds.node_id.astype(int), noeuds.node_idx.astype(int)))
    sel["node_idx"] = sel["troncon"].map(corr)
    perdus = int(sel.node_idx.isna().sum())
    sel = sel.dropna(subset=["node_idx"])
    sel["node_idx"] = sel["node_idx"].astype(int)
    if sel.empty:
        print(f"[prelev] {reg}: aucun troncon apparie ({perdus:,} lignes sans correspondance)")
        continue
    n = BasinCache(db).import_withdrawals(sel[["date", "node_idx", "net_withdrawal", "source"]],
                                          node_col="node_idx", site_col=None)
    net = float(sel["net_withdrawal"].sum()) / max(sel["date"].nunique(), 1)
    print(f"[prelev] {reg}: {sel.site_id.nunique() if 'site_id' in sel else 0:,} sites | "
          f"{n:,} lignes | {sel.node_idx.nunique()} noeuds | "
          f"net moyen {net:+.3f} m3/s | {perdus:,} lignes sans troncon correspondant")
