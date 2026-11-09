"""Empreintes des données d'entrée d'une machine, et comparaison de deux machines (2026-11-09, R519).

Le code se synchronise par git, les données non : chaque machine garde sa copie, et rien ne
vérifiait qu'elles étaient les mêmes. Avant de lancer sur une autre machine (Ubuntu, Narval,
RunPod), on compare ses empreintes à celles du poste, qui fait référence.

    python -m meandre.run.data_fingerprint empreintes-poste.json
    python -m meandre.run.data_fingerprint --compare empreintes-poste.json empreintes-narval.json

Les emplacements viennent de MEANDRE_DATA, MEANDRE_DERIVES et MEANDRE_PLATEFORMES, comme pour le
pilote. Les bases DuckDB sont comparées sur leur contenu, les autres fichiers sur leurs octets.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from meandre.domain.manifest import fingerprint, fingerprint_duckdb
from meandre.utils import paths

AUXILIARIES = ("depots-sigeom-troncons.parquet", "epaisseur-depots-troncons.parquet", "impermeable-worldcover-troncons.parquet",
               "rsesq-puits.parquet", "siigsol-troncons.parquet",
               "slc-drainage-troncons.parquet", "station-attachment.csv", "station-audit.csv")


def inputs() -> dict[str, str]:
    """Fichiers d'entrée du domaine provincial, par nom relatif stable d'une machine à l'autre."""
    q = os.path.join(paths.DATA_ROOT, "quebec")
    out = {}
    for pattern in ("*.duckdb", "forcing-*-casr-brut.nc", "*.parquet"):
        for f in glob.glob(os.path.join(q, pattern)):
            out[f"quebec/{os.path.basename(f)}"] = f
    aux = os.path.join(paths.DERIVED_ROOT, "auxiliaires")
    for n in AUXILIARIES:
        if os.path.exists(os.path.join(aux, n)):
            out[f"auxiliaires/{n}"] = os.path.join(aux, n)
    for f in glob.glob(os.path.join(aux, "neisim-*.npz")):
        out[f"auxiliaires/{os.path.basename(f)}"] = f
    for d in sorted(glob.glob(os.path.join(paths.PLATFORMS_ROOT, "LN24HA", "*_LN24HA_2020"))):
        out[f"plateformes/{os.path.basename(d)}"] = d
    out["paquet/territorial_stats_QC.csv"] = os.path.join(os.path.dirname(paths.__file__), "..", "adapters", "physitel", "data", "territorial_stats_QC.csv")
    return out


def platform_fingerprint(project: str) -> str:
    """Empreinte des fichiers d'une plateforme que meandre lit (reseau, occupation, sols, physiographie,
    etats et configuration de simulation) ; les rasters, couches SIG, sorties de simulation d'Hydrotel
    et la meteo du projet n'entrent pas, ni l'ordre de tri propre au systeme."""
    import hashlib
    rows = []
    for f in glob.glob(os.path.join(project, "**", "*"), recursive=True):
        rel = os.path.relpath(f, project).replace("\\", "/")
        top = rel.split("/")[0]
        keep = (top in ("physitel", "physio", "etat") and rel.rsplit(".", 1)[-1].lower() in ("trl", "csv", "cla", "sol", "def")) or (rel.startswith("simulation/simulation/") and rel.count("/") == 2 and rel.endswith(".csv"))
        if not (keep and os.path.isfile(f)) or rel.endswith("simulation/output.csv"):
            continue
        if rel.endswith("simulation/simulation.csv"):
            # meandre n'en lit que l'indicateur des milieux humides isoles ; dates et chemins des
            # simulations d'Hydrotel changent d'une machine a l'autre sans effet sur le modele.
            flag = [ln.strip() for ln in open(f, encoding="latin-1") if ln.upper().startswith("MILIEUX HUMIDES ISOLES")]
            rows.append(f"{rel}|{'/'.join(flag)}")
        else:
            rows.append(f"{rel}|{fingerprint(f)}")
    h = hashlib.sha1()
    for r in sorted(rows):
        h.update(r.encode())
    return h.hexdigest()[:16]


def compute() -> dict[str, str]:
    res = {}
    for name, p in sorted(inputs().items()):
        if os.path.isdir(p):
            res[name] = platform_fingerprint(p)
        elif p.endswith(".duckdb"):
            res[name] = fingerprint_duckdb(p)
        else:
            res[name] = fingerprint(p)
    return res


def compare(a: dict, b: dict) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            out.append(f"absent de la reference : {k}")
        elif k not in b:
            out.append(f"absent de l'autre machine : {k}")
        elif a[k] != b[k]:
            out.append(f"DIFFERENT : {k}")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        a, b = (json.load(open(f, encoding="utf-8")) for f in sys.argv[2:4])
        gaps = compare(a, b)
        print("\n".join(gaps) if gaps else f"identiques : {len(a)} fichiers")
        sys.exit(1 if any(g.startswith("DIFFERENT") for g in gaps) else 0)
    res = compute()
    json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(f"{len(res)} empreintes ecrites dans {sys.argv[1]}")
