"""Le REGIME hydrologique des stations, recolte a la source et jamais ingere jusqu'ici.

Remarque d'Essi (2026-08-29) : la riviere des Outaouais est pleine de barrages aux
operateurs diversifies, nos strategies de barrage sont infructueuses, et donc un mauvais
KGE sur une station NOTEE INFLUENCEE dans les donnees source est ATTENDU, pas une
anomalie. Rapporter une mediane provinciale sans separer les deux regimes melange ce que
le modele rate parce qu'il est faux avec ce qu'il rate parce qu'on ne lui a rien donne
pour le representer.

Le CEHQ publie l'information en clair dans l'en-tete de chaque fichier de debits :

    Bassin versant: 6768 km2       Regime: Non influence

Nos tables `observations` ne gardent que (station_id, date, discharge, remark,
reconstructed) : le regime n'y a jamais ete ingere, exactement comme le drapeau de
qualite ne l'etait pas avant R19. Ce script le recolte et l'ecrit a cote, pour que toute
statistique de score puisse etre stratifiee.

    python .runs/quebec/regime_stations.py            # toutes les plateformes
    python .runs/quebec/regime_stations.py sagu mont  # un sous-ensemble

Sortie : D:/meandre-data/quebec/regime-stations.csv (station_id, regime, bassin_km2).
"""
import io
import os
import re
import sys
import time
import urllib.request

import duckdb
import pandas as pd

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

BASE = "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier"
RACINE = os.environ.get("MEANDRE_QUEBEC", f"{_DATA_ROOT}/quebec")
SORTIE = f"{RACINE}/regime-stations.csv"
PLATS = [a.lower() for a in sys.argv[1:]] or [
    "outv", "gasp", "mont", "sagu", "slno", "abit", "slso",
    "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "vaud"]


def entete(sid: str) -> tuple[str | None, float | None]:
    """Regime et bassin versant declares, lus dans l'en-tete du fichier du CEHQ.

    On ne telecharge que les premieres lignes : le fichier complet fait plusieurs
    centaines de kilo-octets et l'en-tete tient dans les cinq premieres.
    """
    url = f"{BASE}/{sid}_Q.txt"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            brut = r.read(4096)
    except Exception:
        return None, None
    txt = brut.decode("latin-1", errors="replace")
    # `R.?gime` et non `R\W?gime` : en Unicode un e accentue EST un caractere de mot,
    # donc \W ne le capture pas et le motif rendait None sur les 174 stations, sans
    # lever la moindre erreur -- une colonne entierement vide qui a l'air d'une donnee.
    m = re.search(r"R.?gime\s*:\s*(.+?)\s*$", txt, re.MULTILINE)
    reg = m.group(1).strip() if m else None
    if reg:
        # "Non influence" et "Influence" apres normalisation des accents
        r2 = reg.lower().replace("\u00e9", "e")
        # TROIS libelles au CEHQ, pas deux : "Non influence", "Influence" et
        # "Naturel". Ma premiere normalisation n'en gerait que deux et laissait
        # 28 stations sous un libelle brut, invisible dans un compte par valeur.
        if r2.startswith("non") or r2.startswith("naturel"):
            reg = "naturel"
        elif "influence" in r2:
            reg = "influence"
    b = re.search(r"Bassin versant:\s*([0-9]+)", txt)
    return reg, (float(b.group(1)) if b else None)


def main():
    sids = []
    for p in PLATS:
        db = f"{RACINE}/{p}.duckdb"
        if not os.path.exists(db):
            continue
        c = duckdb.connect(db, read_only=True)
        try:
            for (s,) in c.execute("SELECT DISTINCT station_id FROM stations").fetchall():
                sids.append((p, str(s)))
        except Exception:
            pass
        c.close()
    vus, lignes = set(), []
    print(f"{len(sids)} stations sur {len(PLATS)} plateformes", flush=True)
    for i, (p, s) in enumerate(sids):
        if s in vus:
            continue
        vus.add(s)
        reg, bv = entete(s)
        lignes.append({"plateforme": p, "station_id": s, "regime": reg, "bassin_km2": bv})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sids)}", flush=True)
        time.sleep(0.15)   # on ne martele pas le serveur du ministere
    d = pd.DataFrame(lignes)
    d.to_csv(SORTIE, index=False, encoding="utf-8")
    print(f"\necrit : {SORTIE}")
    print(d.regime.value_counts(dropna=False).to_string())
    print("\npar plateforme :")
    print(pd.crosstab(d.plateforme, d.regime.fillna("inconnu")).to_string())


if __name__ == "__main__":
    main()
