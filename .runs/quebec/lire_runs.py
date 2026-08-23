"""Lecture comparee de plusieurs journaux de run, dans l'ordre qui compte.

Motif (2026-08-22). La file en cours produit quatre journaux (aux-A, aux-B, res-C1,
res-C2) de plusieurs milliers de lignes chacun, et la question posee n'est PAS « lequel
a le meilleur score ». Elle est, dans cet ordre : la nappe se remplit-elle, la phase du
stockage se corrige-t-elle contre GRACE, avril et mai reviennent-ils vers 1.0, et
seulement ensuite ce que tout cela coute au KGE (R11 : une baisse n'est pas un echec,
c'est un arbitrage a expliciter).

Lire quatre journaux a la main dans cet ordre, c'est la garantie de regarder le score en
premier parce qu'il est plus facile a trouver. Ce script impose l'ordre.

    .venv/Scripts/python.exe .runs/quebec/lire_runs.py aux-A aux-B res-C1 res-C2
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from meandre.utils import paths as _paths

JOURNAL = f"{_paths.DATA_ROOT}/quebec"

# Reference : le champion, mesure dans ce meme protocole.
REF = {"mai": -6.0, "juin": -34.0, "mars": 90.0, "nappe": 0.0,
       "avril": 0.753, "mai_q": 1.138, "decembre": 1.235, "score": 0.7885}


def lire(tag):
    chemin = f"{JOURNAL}/log-{tag}.txt"
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8", errors="replace") as f:
        return f.read()


def dernier(motif, texte, groupe=1):
    m = re.findall(motif, texte)
    return m[-1] if m else None


def bloc(texte, debut, fin):
    i = texte.find(debut)
    if i < 0:
        return None
    j = texte.find(fin, i)
    return texte[i:j if j > 0 else None]


def rapporter(tag, texte):
    print(f"\n{'=' * 78}\n  {tag}\n{'=' * 78}")

    # 0. le run a-t-il seulement tourne jusqu'au bout ?
    if "DONE" not in texte and "HELD-OUT" not in texte:
        print("  INCOMPLET : ni HELD-OUT ni DONE dans le journal.")
        tb = bloc(texte, "Traceback", "\n[etl]")
        if tb:
            print("  " + tb.strip().splitlines()[-1])
        return

    # 1. LA NAPPE SE REMPLIT-ELLE ? Premier signe, avant tout score.
    st = bloc(texte, "STOCKS, niveau moyen par mois", "FLUX, lame moyenne")
    if st:
        for ligne in st.splitlines():
            if "nappe" in ligne or "sol L3" in ligne or "manteau" in ligne:
                print("  " + ligne.strip())
        print(f"  (reference champion : nappe 0 mm tous les mois)")
    else:
        print("  1. stocks : absents (ETL_STOCKS=1 non demande ?)")

    # 2. LA PHASE GRACE SE CORRIGE-T-ELLE ?
    gr = bloc(texte, "GRACE TWS (mm)", "\n\n")
    if gr:
        for ligne in gr.splitlines():
            if any(k in ligne for k in ("mois", "biais", "ecart-t", "residu", "PERTE")):
                print("  " + ligne.strip())
        print(f"  (reference champion : +44 mars, -47 mai, -45 juin ; residu 26.4 mm)")
    else:
        print("  2. audit GRACE : absent (ETL_AUX=1 non demande ?)")

    # 3. AVRIL ET MAI REVIENNENT-ILS VERS 1.0 ?
    r = dernier(r"simule/observe par mois : (.+)", texte)
    if r:
        print(f"\n  3. rapport simule/observe par mois :\n     {r.strip()}")
        print(f"     (champion : 04=0.753  05=1.138  12=1.235)")

    # 4. LE CHAMP krec A-T-IL APPRIS, OU EST-IL PLAT ?
    for motif in (r"\[etl\] krec[^\n]*", r"\[etl\] k_gw[^\n]*"):
        for ligne in re.findall(motif, texte)[:2]:
            print(f"  4. {ligne.strip()}")

    # 5. ET SEULEMENT MAINTENANT, LE SCORE.
    ho = dernier(r"HELD-OUT[^\n]*", texte)
    if ho:
        print(f"\n  5. {ho.strip()}")
        print(f"     (champion dans ce protocole : median 0.7885 au meilleur du run A)")
    comp = dernier(r"composantes ponderees \| ([^\n]*)", texte)
    if comp:
        print(f"     composantes a la derniere epoque : {comp.strip()}")
    aux = bloc(texte, "contraintes auxiliaires effectives", "\n[etl] modele")
    if aux:
        print("     " + "\n     ".join(l.strip() for l in aux.splitlines()[1:] if l.strip()))


def main():
    tags = sys.argv[1:] or ["aux-A", "aux-B", "res-C1", "res-C2"]
    print("LECTURE DANS L'ORDRE QUI COMPTE : nappe, phase GRACE, mois, champ, PUIS score.")
    print("Le score vient en dernier a dessein -- R11 : le debit seul prefere une physique")
    print("irrealiste, donc une baisse n'est pas un echec mais un arbitrage a expliciter.")
    for t in tags:
        txt = lire(t)
        if txt is None:
            print(f"\n  {t} : journal absent")
            continue
        rapporter(t, txt)


if __name__ == "__main__":
    main()
