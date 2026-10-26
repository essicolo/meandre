"""Télécharge la fiche technique (detail.asp) de chaque barrage listé.

Les pages brutes sont mises en cache dans barrages/cache/ (windows-1252, telles
que servies) : le parsing est fait à part par 03_parse.py, sans re-télécharger.
"""
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import requests

BASE = "https://www.cehq.gouv.qc.ca/barrages/"
HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")
LISTE = DATA / "liste.csv"
UA = "io-eau/1.0 (recherche hydrologique; contact via depot)"

_tl = local()


def session():
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
        _tl.s.headers["User-Agent"] = UA
    return _tl.s


def fetch(no):
    dest = CACHE / f"{no}.html"
    if dest.exists() and dest.stat().st_size > 5000:
        return "cache"
    for essai in range(4):
        try:
            r = session().get(BASE + "detail.asp", params={"no_mef_lieu": no}, timeout=90)
            if r.status_code == 200 and len(r.content) > 5000:
                dest.write_bytes(r.content)
                return "ok"
            time.sleep(2 * (essai + 1))
        except Exception:
            time.sleep(3 * (essai + 1))
    return "echec"


def main():
    CACHE.mkdir(exist_ok=True)
    nos = [r["no_barrage"] for r in csv.DictReader(LISTE.open(encoding="utf-8"))]
    # les barrages que la recherche du site n'indexe pas mais dont la fiche
    # existe (voir 01c_sondage_numeros.py)
    orph = DATA / "numeros-orphelins.csv"
    if orph.exists():
        vus = set(nos)
        nos += [r["no_barrage"] for r in csv.DictReader(orph.open(encoding="utf-8"))
                if r["no_barrage"] not in vus]
    print(f"{len(nos)} fiches à récupérer")
    n = {"ok": 0, "cache": 0, "echec": 0}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, statut in enumerate(ex.map(fetch, nos), 1):
            n[statut] += 1
            if i % 200 == 0:
                print(f"  {i}/{len(nos)}  {n}", flush=True)
    print(f"terminé : {n}")
    if n["echec"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
