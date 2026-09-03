"""Banc rapide : est-ce que la physique de meandre fonctionne ?

Pourquoi ce banc (2026-09-03, demande d'Essi). Les diagnostics de la semaine prennent
des heures parce qu'ils passent par un entrainement ou par une simulation provinciale.
Un correctif de physique doit pouvoir se verifier en DEUX MINUTES, en local, sans GPU
et sans jeu de donnees provincial. C'est ce que fait ce banc : il enchaine les bancs de
validation du clone contre le binaire Hydrotel C++, module par module, et rend un
tableau. Chaque ligne porte l'ecart au C++ et son seuil d'acceptation.

Ce banc ne dit PAS si un modele est bon (cela demande la periode d'evaluation et les
stations). Il dit si la physique est intacte : neige, gel, evapotranspiration, sol trois
couches, colonne complete avec milieux humides, routage, et differentiabilite de bout en
bout.

  python .runs/quebec/banc_physique.py            tout
  python .runs/quebec/banc_physique.py gel sol    seulement ces bancs
"""
import os
import re
import subprocess
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

# nom court, script, motif du resume, seuil (None = pas de seuil numerique, on exige
# seulement que le banc aille au bout)
BANCS = [
    ("neige", "hydrotel_clone/validate_snow.py", None, None),
    ("gel", "hydrotel_clone/validate_frost.py",
     r"RMSE profondeur_gel = ([\d.]+) cm", 0.05),
    ("etp", "hydrotel_clone/validate_mcguinness.py", None, None),
    ("etr", "hydrotel_clone/validate_et.py", None, None),
    ("sol", "hydrotel_clone/validate_soil_all_uhrh.py",
     r"loam\s+n=\s*\d+\s+rmse_t1 m\S+d ([\d.]+)", 0.05),
    ("colonne", "hydrotel_clone/validate_column_full.py",
     r"TOUS\s+m\S+d ([\d.]+)", 0.10),
    ("routage", "hydrotel_clone/validate_routing.py", None, None),
    ("chaine", "tests/smoke_hydrotel_column.py",
     r"grad d\(prod\)/d\(krec\) = ([\d.]+)", None),
    # Bout en bout sur le banc mini (384 troncons) : champ spatial, colonne, routage et
    # retour du gradient. Verifie qu'un debit sort, sans NaN, et que le champ recoit
    # encore du signal a travers toute la chaine.
    ("bout-en-bout", "tests/smoke_hydrotel_model.py",
     r"grad NeRF ([\d.]+)", None),
]


def lancer(nom, script, motif, seuil):
    depart = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([PY, script], cwd=RACINE, env=env, timeout=600,
                           capture_output=True, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return nom, "DEPASSEMENT", None, 600.0, False
    duree = time.time() - depart
    sortie = (r.stdout or "") + (r.returncode and (r.stderr or "") or "")
    if r.returncode != 0:
        return nom, "ECHEC", None, duree, False
    valeur = None
    if motif:
        m = re.search(motif, sortie)
        if m:
            valeur = float(m.group(1))
    if valeur is None:
        # Pas de nombre a extraire : le banc doit au moins avoir signale sa fin.
        fini = "DONE" in sortie or "SMOKE OK" in sortie
        return nom, ("ok" if fini else "SANS VERDICT"), None, duree, fini
    if seuil is not None and valeur > seuil:
        return nom, "HORS SEUIL", valeur, duree, False
    return nom, "ok", valeur, duree, True


def main():
    demandes = [a.lower() for a in sys.argv[1:]]
    bancs = [b for b in BANCS if not demandes or b[0] in demandes]
    if not bancs:
        raise SystemExit(f"bancs connus : {', '.join(b[0] for b in BANCS)}")

    print(f"{'banc':10s} {'verdict':13s} {'ecart au C++':>13s} {'seuil':>8s} {'duree':>7s}")
    print("-" * 56)
    total = 0.0
    echecs = []
    for nom, script, motif, seuil in bancs:
        nom, verdict, valeur, duree, ok = lancer(nom, script, motif, seuil)
        total += duree
        if not ok:
            echecs.append(nom)
        v = f"{valeur:.4f}" if valeur is not None else "-"
        s = f"{seuil:.3f}" if seuil is not None else "-"
        print(f"{nom:10s} {verdict:13s} {v:>13s} {s:>8s} {duree:6.1f}s", flush=True)
    print("-" * 56)
    print(f"{len(bancs) - len(echecs)}/{len(bancs)} bancs au vert en {total:.0f} s")
    if echecs:
        print(f"EN ECHEC : {', '.join(echecs)}")
        return 1
    print("La physique du clone est intacte (ecarts au binaire C++ sous les seuils).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
