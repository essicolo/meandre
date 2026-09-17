"""Télécharge des jeux de Données Québec dans `sources/`, avec leur source.toml.

Un dossier par jeu, le format vectoriel le plus commode, les métadonnées en PDF, et les index
de téléchargement des produits découpés en feuillets. Le source.toml porte la provenance, la
licence et la nature de la donnée ; les empreintes des fichiers vont dans empreintes.txt.

    .venv/Scripts/python.exe .runs/quebec/telecharger_donnees_quebec.py [jeu ...]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

CURL = shutil.which("curl")
API = "https://www.donneesquebec.ca/recherche/api/3/action/package_show?"
# Un dossier de source par jeu, avec la nature de la donnée et les formats à prendre.
JEUX = {
    "eau-souterraines-sih-index": ("hydrogeologie-sih", "observe", ("GPKG", "FGDB")),
    "rsesq": ("eaux-souterraines-rsesq", "observe", ("GPKG", "SQLITE", "JSON", "FGDB")),
    "projets-d-acquisition-de-connaissances-sur-les-eaux-souterraines-paces": ("eaux-souterraines-paces", "interpole", ("GPKG", "SHP", "FGDB")),
    "bathymetries-lacs": ("bathymetrie-lacs", "observe", ("GPKG",)),
    "lits-d-ecoulements-potentiels-issus-du-lidar": ("lit-ecoulement-lidar", "interpole", ("CSV", "SHP")),
    "ecotones-riverains-issus-du-lidar": ("ecotone-riverain-lidar", "interpole", ("CSV", "SHP")),
    "crhq": ("crhq", "interpole", ("FGDB", "GPKG")),
    "rscq_grilles_climatiques_version_3": ("grilles-climatiques-gcq", "interpole", ("CSV", "ZIP")),
    "milieux-humides-potentiels": ("milieux-humides-potentiels", "predit", ("GPKG",)),
    "milieux-humides-du-quebec": ("milieux-humides-detaille", "interpole", ("FGDB", "GPKG")),
    "prelevements-eau": ("prelevements-eau-melccfp", "observe", ("GPKG", "CSV", "XLSX")),
    "pressions-municipales-rejets-d-eaux-usees": ("rejets-municipaux", "observe", ("CSV",)),
    "pressions-industrielles-rejets-d-eaux-usees": ("rejets-industriels", "observe", ("CSV",)),
    "carte-des-depots-de-surface-du-nord-quebecois": ("depot-surface-nord", "interpole", ("GPKG", "SQLITE", "FGDB")),
    "resultats-d-inventaire-et-carte-ecoforestiere": ("ecoforestier", "interpole", ("GPKG",)),
}
DOCS = ("PDF", "XLSX")


def empreinte(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def telecharger(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return os.path.getsize(destination), True
    # Certains chemins du catalogue portent des espaces : ils doivent être encodés.
    morceaux = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit(morceaux._replace(path=urllib.parse.quote(morceaux.path)))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partiel = destination + ".partiel"
    # Le serveur de diffusion du ministère ferme la connexion ouverte par urllib, quel que
    # soit l'agent annoncé, alors que curl passe. On délègue donc à curl quand il existe.
    if CURL:
        r = subprocess.run([CURL, "-sSL", "--retry", "3", "--retry-delay", "5", "--max-time", "7200",
                            "-A", "meandre/1.0", "-o", partiel, url], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(partiel):
            raise RuntimeError(f"curl {r.returncode} : {(r.stderr or '').strip()[:120]}")
    else:
        requete = urllib.request.Request(url, headers={"User-Agent": "meandre/1.0"})
        for essai in range(3):
            try:
                with urllib.request.urlopen(requete, timeout=300) as r, open(partiel, "wb") as f:
                    while True:
                        bloc = r.read(1 << 20)
                        if not bloc:
                            break
                        f.write(bloc)
                break
            except Exception:
                if essai == 2:
                    raise
                time.sleep(5)
    os.replace(partiel, destination)
    return os.path.getsize(destination), False


def nom_fichier(ressource, url):
    base = os.path.basename(urllib.parse.urlparse(url).path)
    if not base or "." not in base:
        base = (ressource.get("name") or "ressource").replace(" ", "_")[:60] + "." + (ressource.get("format") or "bin").lower()
    return urllib.parse.unquote(base)


def un_jeu(nom_jeu):
    dossier, nature, formats = JEUX[nom_jeu]
    with urllib.request.urlopen(API + urllib.parse.urlencode({"id": nom_jeu}), timeout=120) as r:
        d = json.load(r)["result"]
    base = f"{_paths.SOURCES_ROOT}/{dossier}"
    os.makedirs(base, exist_ok=True)
    ressources = d.get("resources", [])
    garde = [x for x in ressources if (x.get("format") or "").upper() in formats]
    if not garde:
        print(f"{dossier} : aucun format parmi {formats}, ressources {(sorted({(x.get('format') or '').upper() for x in ressources}))}")
    garde += [x for x in ressources if (x.get("format") or "").upper() in DOCS]
    lignes = []
    for x in garde:
        url = x.get("url")
        if not url:
            continue
        cible = f"{base}/{nom_fichier(x, url)}"
        try:
            taille, deja = telecharger(url, cible)
        except Exception as e:
            print(f"  {os.path.basename(cible)[:50]:52s} ECHEC {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  {os.path.basename(cible)[:50]:52s} {taille / 1e6:9.1f} Mo{' (déjà là)' if deja else ''}", flush=True)
        lignes.append((os.path.basename(cible), taille, x.get("format"), url))
    with open(f"{base}/empreintes.txt", "w", encoding="utf-8") as f:
        for fichier, taille, _, url in lignes:
            f.write(f"{empreinte(f'{base}/{fichier}')}  {fichier}  {taille}  {url}\n")
    toml = [
        "[source]",
        f'nom = "{d["title"]}"',
        f'producteur = "{d["organization"]["title"]}"',
        f'url = "https://www.donneesquebec.ca/recherche/dataset/{nom_jeu}"',
        f'licence = "{d.get("license_title") or "?"}"',
        f'version = "{(d.get("metadata_modified") or "")[:10]}"',
        "telecharge_le = 2026-09-17",
        f'nature = "{nature}"',
        "",
        "# Empreintes et tailles des fichiers : empreintes.txt",
        "# La recette d'ingestion reste à écrire : type, variables, transformations.",
        "[ingestion]",
        'type = "a_definir"',
        "",
    ]
    for fichier, _, fmt, _u in lignes:
        toml += ["[[fichier]]", f'chemin = "{fichier}"', f'format = "{fmt}"', ""]
    with open(f"{base}/source.toml", "w", encoding="utf-8") as f:
        f.write("\n".join(toml))
    return sum(t for _, t, _, _ in lignes)


def main(jeux):
    total = 0
    for j in jeux:
        print(f"\n== {j}", flush=True)
        total += un_jeu(j)
    print(f"\ntotal téléchargé : {total / 1e9:.1f} Go", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(JEUX)))
