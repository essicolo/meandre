"""Débits journaliers du CEHQ, AVEC leur drapeau de qualité.

Pourquoi ce module existe (registre R19, 2026-08-21). Nos tables `observations` ne
gardent que ``(station_id, date, discharge)``. Le drapeau que le CEHQ publie à côté de
chaque valeur n'a jamais été ingéré, alors qu'il change la nature de la comparaison :
sous couvert de glace la relation hauteur-débit ne tient plus, et l'organisme ESTIME
la valeur par interpolation entre jaugeages manuels. Une mesure indirecte le montrait
déjà (21.7 % des jours de février sont sur un segment parfaitement rectiligne contre
0.3 % en été), mais indirectement ; le drapeau le dit.

Conséquence pratique : de décembre à mars, une part du signal contre lequel on ajuste
le modèle est la reconstruction d'un hydrologue, pas une mesure. Hydrotel est calé sur
les mêmes séries, donc la comparaison entre modèles reste équitable ; ce qui ne l'est
pas, c'est de traiter cette part comme une vérité de terrain.

Lexique publié par le CEHQ dans l'en-tête de chaque fichier :
    E   la donnée est estimée
    J   un jaugeage a été exécuté à cette date
    MC  débit moyen converti
    MJ  moyenne journalière
    P   provisoire ; P* provisoire et non estimée
    PL  première lecture de niveau de la journée
    R   débit corrigé pour tenir compte de l'effet de REFOULEMENT (glace, embâcle)
    S   saisie manuelle
    Z   redistribution temporelle (défectuosité de l'appareil)
"""
from __future__ import annotations

_BASE = "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier"

# Les deux codes qui disqualifient une valeur comme MESURE. `E` est explicite. `R`
# signale un debit corrige pour refoulement, c'est-a-dire que la courbe de tarage a ete
# jugee inapplicable telle quelle -- typiquement sous glace ou embacle. Les autres codes
# decrivent la provenance (jaugeage, saisie manuelle, moyenne) sans mettre en doute la
# valeur elle-meme.
RECONSTRUCTED_FLAGS = ("E", "R")


def parse_cehq_text(texte: str):
    """Analyse un fichier `NNNNNN_Q.txt` du CEHQ.

    Retourne un DataFrame ``(station_id, date, discharge, remark)``. Les jours sans
    valeur sont CONSERVES avec ``discharge`` a NaN : leur absence est une information
    (lacune de la station) qu'on perdrait en les filtrant ici.
    """
    import numpy as np
    import pandas as pd

    lignes = texte.splitlines()
    debut = None
    for i, l in enumerate(lignes):
        if l.strip().startswith("Station") and "Date" in l and "Remarque" in l:
            debut = i + 1
            break
    if debut is None:
        raise ValueError("en-tete de tableau introuvable : format CEHQ inattendu")

    sid, dates, val, rem = [], [], [], []
    for l in lignes[debut:]:
        if not l.strip():
            continue
        m = l.split()
        if len(m) < 2 or "/" not in m[1]:
            continue
        sid.append(m[0])
        dates.append(m[1])
        # colonne debit absente = lacune ; presente = valeur, suivie ou non d'une remarque
        if len(m) >= 3:
            try:
                val.append(float(m[2]))
                rem.append(" ".join(m[3:]) if len(m) > 3 else "")
            except ValueError:
                # pas de debit, le 3e champ est deja la remarque
                val.append(np.nan)
                rem.append(" ".join(m[2:]))
        else:
            val.append(np.nan)
            rem.append("")
    return pd.DataFrame({"station_id": sid,
                         "date": pd.to_datetime(dates, format="%Y/%m/%d"),
                         "discharge": val, "remark": rem})


def is_reconstructed(remark) -> bool:
    """Vrai si la remarque porte un code qui fait de la valeur une RECONSTRUCTION.

    Les codes sont separes par des espaces et peuvent se cumuler. `P*` ne compte pas
    comme estime : l'etoile signifie justement provisoire et NON estimee.
    """
    if not isinstance(remark, str) or not remark:
        return False
    return any(c in RECONSTRUCTED_FLAGS for c in remark.replace("P*", "").split())


def fetch_station(station_id: str, timeout: float = 60.0) -> str:
    """Telecharge le fichier brut d'une station. Encodage latin-1 (publie tel quel)."""
    import urllib.request
    url = f"{_BASE}/{station_id}_Q.txt"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("latin-1")
