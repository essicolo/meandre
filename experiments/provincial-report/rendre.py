"""Rend la presentation et le rapport provinciaux avec le bon interpreteur.

POURQUOI CE SCRIPT (2026-09-10). `quarto render` choisit tout seul un Python, et sur ce
poste il tombe sur celui du Windows Store, ou Jupyter n'est pas installe. Quarto n'echoue
alors PAS : il ecrit un HTML sans executer un seul bloc de code, en laissant le message
« Jupyter is not available » au milieu de son journal. On croit avoir rendu le document
et on a produit une coquille avec les figures de la veille. Ce script pose QUARTO_PYTHON
sur l'interpreteur du depot et VERIFIE que les blocs ont bien tourne.

    python .runs/quebec/rendre.py                 # presentation + rapport
    python .runs/quebec/rendre.py presentation
    python .runs/quebec/rendre.py rapport
"""
import os
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
DOSSIER = RACINE / ".reports" / "quebec"
CIBLES = {"presentation": ("presentation.qmd", "revealjs"),
          "rapport": ("rapport_provincial.qmd", "html")}


def rendre(nom):
    src, fmt = CIBLES[nom]
    env = dict(os.environ)
    env["QUARTO_PYTHON"] = sys.executable
    env["PYTHONIOENCODING"] = "utf-8"
    journal = DOSSIER / f"rendu-{nom}.log"
    print(f"[rendu] {src} vers {fmt} ...", flush=True)
    t0 = time.time()
    with open(journal, "w", encoding="utf-8", errors="replace") as f:
        # --no-execute-daemon : quarto garde un noyau jupyter vivant 300 s et le REUTILISE
        # au rendu suivant. Un module du depot modifie entre deux rendus reste alors celui
        # de la version precedente, deja importe. On paie quelques secondes de demarrage
        # pour la garantie que le document rendu correspond au code sur le disque.
        p = subprocess.run(["quarto", "render", src, "--to", fmt, "--no-execute-daemon"],
                           cwd=str(DOSSIER), env=env, stdout=f,
                           stderr=subprocess.STDOUT, shell=(os.name == "nt"))
    texte = journal.read_text(encoding="utf-8", errors="replace")
    # Le piege : quarto rend 0 meme quand il n'a execute aucun bloc.
    muet = "Jupyter is not available" in texte or "ModuleNotFoundError" in texte
    sortie = DOSSIER / (src.replace(".qmd", ".html"))
    ok = p.returncode == 0 and not muet and sortie.exists()
    print(f"[rendu] {nom} : {'ok' if ok else 'ECHEC'} en {time.time()-t0:.0f} s "
          f"-> {sortie.name} ({sortie.stat().st_size/1024:.0f} ko)" if sortie.exists()
          else f"[rendu] {nom} : ECHEC, aucune sortie")
    if not ok:
        print("".join(texte.splitlines(keepends=True)[-25:]))
        if muet:
            print("[rendu] CAUSE : quarto n'a execute aucun bloc de code "
                  "(Jupyter absent de l'interpreteur choisi).")
    return ok


def main():
    noms = sys.argv[1:] or list(CIBLES)
    inconnus = [n for n in noms if n not in CIBLES]
    if inconnus:
        raise SystemExit(f"cible inconnue : {inconnus}. Choix : {list(CIBLES)}")
    print(f"[rendu] interpreteur {sys.executable}")
    resultats = {n: rendre(n) for n in noms}
    if not all(resultats.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
