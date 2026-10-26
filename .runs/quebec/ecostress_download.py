"""Télécharge les scènes ECOSTRESS (ET PT-JPL et LST) au-dessus de la Montérégie.

Produits (collections actuelles LP DAAC, versions 002/003 confondues par la recherche) :
  ECO_L3T_JET  : evapotranspiration tuilée, fichier journalier *_ETdaily.tif (mm/j)
  ECO_L2T_LSTE : température de surface tuilée, *_LST.tif (K)

Le sous-ensemble demandé : saisons de croissance (1er juin au 30 septembre) 2019-2024,
la fenêtre de comparaison avec MOD16. Les fichiers sont déposés sous
D:/meandre-data/ecostress/{jet,lste}/, à plat, noms de fichiers uniques.

Reprise : un fichier déjà présent et non vide est sauté. Le journal dit, par année et
par produit, les granules trouvés, les fichiers écrits et le volume cumulé.

Usage :
    python .runs/quebec/ecostress_download.py            # tout (2019-2024, juin-septembre)
    python .runs/quebec/ecostress_download.py --essai    # une scène de chaque produit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import earthaccess
import requests

BBOX = (-74.6, 44.95, -72.3, 45.9)  # Montérégie large
ANNEES = range(2019, 2025)
RACINE = Path(r"D:\meandre-data\ecostress")

# suffixes de fichiers retenus par produit (le reste : couches non nécessaires)
PRODUITS = {
    "jet": ("ECO_L3T_JET", ("_ETdaily.tif", "_ETinst.tif", "_ETinstUncertainty.tif",
                            "_cloud.tif", "_water.tif")),
    "lste": ("ECO_L2T_LSTE", ("_LST.tif", "_LST_err.tif", "_QC.tif", "_cloud.tif",
                              "_water.tif", "_EmisWB.tif", "_view_zenith.tif")),
}


def telecharger(session, url: str, cible: Path, essais: int = 4) -> tuple[int, bool]:
    """Télécharge une URL si le fichier manque. Retourne (octets, écrit).

    LP DAAC renvoie parfois 502/503 : on réessaie avec un délai croissant.
    """
    if cible.exists() and cible.stat().st_size > 0:
        return cible.stat().st_size, False
    dernier = None
    for tentative in range(essais):
        try:
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                attendu = int(r.headers.get("Content-Length", 0))
                temporaire = cible.with_suffix(cible.suffix + ".part")
                n = 0
                with open(temporaire, "wb") as f:
                    for morceau in r.iter_content(chunk_size=1 << 20):
                        f.write(morceau)
                        n += len(morceau)
                if attendu and n != attendu:
                    temporaire.unlink(missing_ok=True)
                    raise IOError(f"taille reçue {n} != annoncée {attendu} pour {cible.name}")
                temporaire.replace(cible)
                return n, True
        except (requests.RequestException, IOError) as e:
            dernier = e
            time.sleep(2.0 * (tentative + 1))
    raise dernier if dernier is not None else RuntimeError(url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essai", action="store_true", help="une scène de chaque produit, 2020")
    args = ap.parse_args()

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        print("Échec de l'authentification Earthdata (.netrc).", flush=True)
        return 1
    session = earthaccess.get_requests_https_session()

    annees = [2020] if args.essai else list(ANNEES)
    total_octets = 0
    total_ecrits = 0
    echecs = []

    for cle, (produit, suffixes) in PRODUITS.items():
        dossier = RACINE / cle
        dossier.mkdir(parents=True, exist_ok=True)
        for annee in annees:
            granules = earthaccess.search_data(
                short_name=produit, bounding_box=BBOX,
                temporal=(f"{annee}-06-01", f"{annee}-09-30"), count=-1)
            vus = set()
            uniques = []
            for g in granules:
                gid = g["meta"].get("native-id", "")
                if gid not in vus:
                    vus.add(gid)
                    uniques.append(g)
            if args.essai:
                uniques = uniques[:1]
            print(f"[{cle}] {annee} : {len(uniques)} granules", flush=True)
            octets_annee = 0
            for i, g in enumerate(uniques, 1):
                liens = [l for l in g.data_links() if l.endswith(suffixes)]
                for url in liens:
                    cible = dossier / url.split("/")[-1]
                    try:
                        n, ecrit = telecharger(session, url, cible)
                    except Exception as e:
                        echecs.append((cible.name, str(e)))
                        print(f"  ÉCHEC {cible.name}: {e}", flush=True)
                        continue
                    octets_annee += n
                    total_ecrits += int(ecrit)
                if i % 25 == 0 or i == len(uniques):
                    print(f"  {cle} {annee} : {i}/{len(uniques)} granules, "
                          f"{octets_annee / 1e9:.2f} Go cette année", flush=True)
            total_octets += octets_annee
            print(f"[{cle}] {annee} terminé : {octets_annee / 1e9:.2f} Go", flush=True)

    print(f"\nTotal : {total_ecrits} fichiers écrits, {total_octets / 1e9:.2f} Go sous {RACINE}",
          flush=True)
    if echecs:
        print(f"{len(echecs)} échecs (à relancer, la reprise les reprendra) :", flush=True)
        for nom, err in echecs[:20]:
            print(f"  {nom}: {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
