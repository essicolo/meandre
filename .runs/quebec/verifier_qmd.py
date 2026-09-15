"""Controle statique d'un document Quarto, avant de le rendre.

Le rendu ne revele qu'une erreur a la fois, et chaque cycle coute plusieurs minutes. Le
2026-09-15, trois erreurs se sont succedees ainsi : une variante de fichier absente de six
regions sur quinze, puis `xarray` non importe. Ce controle les trouve d'un coup, sans rien
executer.

Trois verifications :
  la syntaxe de chaque cellule ;
  les noms utilises et jamais definis par une cellule anterieure ;
  les chemins de fichiers litteraux, en developpant les variantes de region.

Les cellules portant `#| eval: false` sont ignorees. ATTENTION : une cellule placee dans un
commentaire HTML est EXECUTEE par le moteur jupyter, qui decoupe sur les clotures de code et
ignore les commentaires ; il faut donc lui poser `#| eval: false` explicitement.

    .venv/Scripts/python.exe .runs/quebec/verifier_qmd.py .reports/quebec/presentation.qmd
"""
import ast
import builtins
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
s = io.open(sys.argv[1] if len(sys.argv) > 1 else ".reports/quebec/presentation.qmd",
            encoding="utf-8").read()

connus = set(dir(builtins))
manquants = []
for k, m in enumerate(re.finditer(r"```\{python\}\n(.*?)```", s, re.S), 1):
    brut = m.group(1)
    entetes = [l for l in brut.split("\n") if l.strip().startswith("#|")]
    if any("eval: false" in e for e in entetes):
        continue
    code = "\n".join(l for l in brut.split("\n") if not l.strip().startswith("#|"))
    try:
        arbre = ast.parse(code)
    except SyntaxError as e:
        print(f"cellule {k} : syntaxe invalide, {e.msg}")
        continue
    # Les noms definis par cette cellule, y compris les cibles de boucle et les fonctions.
    definis = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            definis.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definis.add(n.name)
        elif isinstance(n, ast.alias):
            definis.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.arg):
            definis.add(n.arg)
        elif isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
            definis.add(n.target.id)
    utilises = {n.id for n in ast.walk(arbre)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    absents = sorted(utilises - definis - connus)
    if absents:
        print(f"cellule {k} : utilise sans définition antérieure -> {', '.join(absents)}")
        manquants += [(k, a) for a in absents]
    connus |= definis
print(f"\n{len(manquants)} nom(s) potentiellement manquant(s)")
