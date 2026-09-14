"""Énumère tous les barrages du répertoire du CEHQ.

Le formulaire de recherche (default.asp) refuse une requête sans critère : on
boucle donc sur la liste des MRC, qui partitionne exhaustivement le territoire.
Chaque reponse est la liste complète de la MRC (pas de pagination).

Sortie : barrages/data/liste.csv
"""
import re
import time
import csv
from pathlib import Path

import requests
from urllib.parse import urlencode

BASE = "https://www.cehq.gouv.qc.ca/barrages/"
OUT = DATA / "liste.csv"
ENC = "cp1252"


def decode(r):
    return r.content.decode(ENC, errors="replace")


def options_mrc(html):
    m = re.search(r'<select[^>]*name="mrc"[^>]*>(.*?)</select>', html, re.S | re.I)
    vals = re.findall(r'<option[^>]*value="([^"]*)"', m.group(1))
    return [unescape(v) for v in vals if v.strip()]


def unescape(s):
    for a, b in (("&#039;", "'"), ("&amp;", "&"), ("&mdash;", "—"), ("&#8212;", "—"), ("&nbsp;", " "), ("&#34;", '"')):
        s = s.replace(a, b)
    return s.strip()


def parse_liste(html, mrc):
    rows = []
    # une ligne = un <tr> contenant un lien detail.asp
    for tr in re.findall(r"(?is)<tr>(.*?)</tr>", html):
        if "detail.asp?no_mef_lieu=" not in tr:
            continue
        no = re.search(r"no_mef_lieu=([^\"'&>]+)", tr).group(1)
        cells = [unescape(re.sub(r"(?s)<[^>]+>", " ", c)) for c in re.findall(r"(?is)<td[^>]*>(.*?)</td>", tr)]
        cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
        cells += [""] * (6 - len(cells))
        rows.append({
            "no_barrage": no,
            "nom_barrage": cells[1],
            "municipalite": cells[2],
            "mrc_liste": cells[3],
            "lac_cours_eau": cells[4],
            "categorie": cells[5],
            "mrc_requete": mrc,
        })
    return rows


def main():
    s = requests.Session()
    s.headers["User-Agent"] = "io-eau/1.0 (recherche hydrologique; contact via depot)"
    home = decode(s.get(BASE + "default.asp", timeout=60))
    mrcs = options_mrc(home)
    print(f"{len(mrcs)} MRC à interroger")

    seen, out = set(), []
    for i, mrc in enumerate(mrcs, 1):
        data = {"nom_barrage": "", "lac_cours_eau": "", "municipalite": "", "mrc": mrc,
                "no_barrage": "", "region": "", "contenance1": "on", "contenance2": "on",
                "contenance3": "on", "cpvalide": "", "code_postal": ""}
        for essai in range(3):
            try:
                # Le serveur ASP classique attend le formulaire en windows-1252 :
                # poste en UTF-8, toute MRC accentuee ne renvoie aucune ligne.
                body = urlencode(data, encoding=ENC, errors="replace")
                r = s.post(BASE + "ListeBarrages.asp?Tri=No", data=body.encode("ascii"),
                           headers={"Content-Type": "application/x-www-form-urlencoded"},
                           timeout=120)
                rows = parse_liste(decode(r), mrc)
                break
            except Exception as e:
                print(f"  reprise {mrc}: {e}")
                time.sleep(5)
        else:
            print(f"  ECHEC {mrc}")
            continue
        neufs = [x for x in rows if x["no_barrage"] not in seen]
        seen.update(x["no_barrage"] for x in rows)
        out.extend(neufs)
        print(f"[{i:3d}/{len(mrcs)}] {mrc:45s} {len(rows):4d} lignes, {len(neufs):4d} nouveaux (total {len(out)})")
        time.sleep(0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\n{len(out)} barrages ecrits dans {OUT}")


if __name__ == "__main__":
    main()
