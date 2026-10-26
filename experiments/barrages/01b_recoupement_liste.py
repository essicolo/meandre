"""Contrôle d'exhaustivité de la liste par une seconde partition du territoire.

01_liste.py énumère les barrages MRC par MRC. Rien ne garantit a priori que
cette partition couvre tout le répertoire (une fiche sans MRC renseignée y
serait invisible). On recommence donc l'énumération par municipalité, qui est
une partition indépendante, et on compare les deux ensembles.

Sortie : barrages/data/liste-municipalites.csv et un diagnostic à l'écran.
"""
import csv
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

import importlib.util

_spec = importlib.util.spec_from_file_location("liste", Path(__file__).parent / "01_liste.py")
_liste = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_liste)

BASE = _liste.BASE
ENC = _liste.ENC
HERE = Path(__file__).parent
# Les donnees vivent hors du depot : elles pesent une centaine de mega et se
# regenerent depuis le site du CEHQ (02_fiches.py saute ce qui est en cache).
DATA = Path("D:/meandre-data/barrages/data")
CACHE = Path("D:/meandre-data/barrages/cache")


def options(html, champ):
    m = re.search(rf'<select[^>]*name="{champ}"[^>]*>(.*?)</select>', html, re.S | re.I)
    vals = re.findall(r'<option[^>]*value="([^"]*)"', m.group(1))
    return [_liste.unescape(v) for v in vals if v.strip()]


def main():
    s = requests.Session()
    s.headers["User-Agent"] = "io-eau/1.0 (recherche hydrologique; contact via depot)"
    for essai in range(5):
        try:
            home = _liste.decode(s.get(BASE + "default.asp", timeout=120))
            break
        except Exception as e:
            print(f"  reprise page d'accueil : {e}")
            time.sleep(10)
    else:
        raise SystemExit("le site ne repond pas")
    muns = options(home, "municipalite")
    print(f"{len(muns)} municipalités à interroger")

    vus, out = set(), []
    for i, mun in enumerate(muns, 1):
        data = {"nom_barrage": "", "lac_cours_eau": "", "municipalite": mun, "mrc": "",
                "no_barrage": "", "region": "", "contenance1": "on", "contenance2": "on",
                "contenance3": "on", "cpvalide": "", "code_postal": ""}
        body = urlencode(data, encoding=ENC, errors="replace").encode("ascii")
        for essai in range(3):
            try:
                r = s.post(BASE + "ListeBarrages.asp?Tri=No", data=body,
                           headers={"Content-Type": "application/x-www-form-urlencoded"},
                           timeout=120)
                rows = _liste.parse_liste(_liste.decode(r), mun)
                break
            except Exception as e:
                print(f"  reprise {mun}: {e}")
                time.sleep(5)
        else:
            print(f"  ECHEC {mun}")
            continue
        for x in rows:
            if x["no_barrage"] not in vus:
                vus.add(x["no_barrage"])
                out.append(x)
        if i % 50 == 0:
            print(f"[{i:3d}/{len(muns)}] total {len(out)}", flush=True)
        time.sleep(0.2)

    dest = DATA / "liste-municipalites.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    par_mrc = {r["no_barrage"] for r in
               csv.DictReader((DATA / "liste.csv").open(encoding="utf-8"))}
    print(f"\npartition par MRC          : {len(par_mrc)}")
    print(f"partition par municipalité : {len(vus)}")
    print(f"vus seulement par MRC          : {len(par_mrc - vus)}")
    print(f"vus seulement par municipalité : {len(vus - par_mrc)}")
    for no in sorted(vus - par_mrc)[:20]:
        print("   ", no)


if __name__ == "__main__":
    main()
