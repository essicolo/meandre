"""Troisième contrôle d'exhaustivité : sonder les numéros manquants.

Les numéros de barrage historiques forment un bloc dense (X0000462 à X0008036).
On interroge directement `detail.asp` sur chaque numéro de ce bloc qui n'est pas
dans la liste : une fiche inexistante renvoie une erreur 500, une fiche
existante une page complète. Un seul numéro absent de la liste mais servi par
le site suffirait à prouver que l'énumération par MRC est incomplète.

Le second bloc (X2000000 et plus) est trop clairsemé pour être sondé de la même
façon : ces numéros appartiennent à un registre de lieux plus large que les
seuls barrages.

Sortie : diagnostic à l'écran et barrages/data/numeros-orphelins.csv
"""
import csv
import json
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
BLOC = (462, 8036)

_tl = local()


def session():
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
        _tl.s.headers["User-Agent"] = "io-eau/1.0 (recherche hydrologique; contact via depot)"
    return _tl.s


def existe(no):
    """True / False / None (le site n'a pas tranché).

    Le serveur finit par refuser toute requête quand on le sollicite trop :
    on recule franchement plutôt que d'insister, sinon il ne revient pas.
    """
    for essai in range(4):
        try:
            r = session().get(BASE + "detail.asp", params={"no_mef_lieu": no}, timeout=60)
            if r.status_code == 500:
                return False
            if r.status_code == 200 and b"Fiche technique" in r.content:
                return True
        except Exception:
            pass
        time.sleep(5 * (essai + 1))
    return None                       # indetermine


def main():
    connus = {r["no_barrage"] for r in
              csv.DictReader((DATA / "liste.csv").open(encoding="utf-8"))}
    # journal des sondages déjà tranchés : le site tombe régulièrement, la
    # reprise ne doit pas tout refaire. Les indéterminés sont resondés.
    journal = DATA / "sondage-numeros.json"
    deja = json.loads(journal.read_text()) if journal.exists() else {}
    a_sonder = [f"X{i:07d}" for i in range(BLOC[0], BLOC[1] + 1)
                if f"X{i:07d}" not in connus and deja.get(f"X{i:07d}") is None]
    print(f"{len(a_sonder)} numéros à sonder dans le bloc "
          f"X{BLOC[0]:07d}-X{BLOC[1]:07d} ({len(deja)} déjà tranchés)")

    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (no, res) in enumerate(zip(a_sonder, ex.map(existe, a_sonder)), 1):
                deja[no] = res
                if res is True:
                    print(f"  ORPHELIN {no}", flush=True)
                if i % 200 == 0:
                    journal.write_text(json.dumps(deja))
                    print(f"  {i}/{len(a_sonder)} sondés", flush=True)
    finally:
        journal.write_text(json.dumps(deja))

    orphelins = sorted(k for k, v in deja.items() if v is True)
    indetermines = [k for k, v in deja.items() if v is None]
    dest = DATA / "numeros-orphelins.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no_barrage"])
        w.writerows([[x] for x in orphelins])
    print(f"\nfiches servies par le site mais absentes de la liste : {len(orphelins)}")
    print(f"numéros indéterminés (le site n'a pas répondu) : {len(indetermines)}")


if __name__ == "__main__":
    main()
