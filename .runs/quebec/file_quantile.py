"""Phase probabiliste : une tete de quantiles par region, sur le socle GELE.

POURQUOI EN PYTHON ET PAS EN SHELL (2026-09-10). La version shell a echoue deux fois
pour des raisons qui n'ont rien a voir avec le modele : `python` absent du PATH de Git
Bash, puis `bash` tape depuis cmd qui appelle le bash de WSL, lequel ne transmet aux
executables Windows que les variables listees dans WSLENV. Toutes les variables ETL_*
etaient donc jetees EN SILENCE et le pilote repartait sur ses defauts, region gasp et
douze epoques, en ecrivant son journal dans un dossier nomme `D:`. Ce script supprime
la question : il pose l'environnement dans le processus fils, sans shell.

POURQUOI PAS LES CONFIGURATIONS <region>-quantile.toml. Elles designent
checkpoints/best-<region>.pt, qui n'existe pas ; slso.py saute alors le depart a chaud
sans rien dire et gele une physique jamais entrainee. Elles ne portent ni la recette du
socle ni le forcage -budyko. Mesure sur six regions : le debit exporte s'ecarte de 27 a
62 % de celui de la flotte.

    python .runs/quebec/file_quantile.py                 # les 9 regions disponibles
    python .runs/quebec/file_quantile.py gasp sagu       # une selection
    python .runs/quebec/file_quantile.py --epoques 4
    python .runs/quebec/file_quantile.py --sans-controle  # sauter le controle prealable
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MEANDRE_DATA", "D:/meandre-data"))
PLATES = Path(os.environ.get("MEANDRE_PLATEFORMES",
                             "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel"))
FLOTTE = DATA / "quebec" / "flotte"
SORTIE = DATA / "quebec" / "rapport"
DEPART = os.environ.get("DEPART", "-etl-fdsA-v4")
# La plus petite region en tete : le controle repond plus vite.
REGIONS = ["cndc", "cndb", "abit", "sagu", "mont", "slso", "slno", "outv", "gasp"]


def lancer(reg, epoques, dump, journal):
    ck = FLOTTE / f"best-{reg}{DEPART}.pt"
    if not ck.exists():
        print(f"[quantile] {reg} SAUTE : {ck} absent.")
        return None
    env = dict(os.environ)
    env.update({
        "ETL_CONFIG": str(RACINE / ".runs/quebec/config/socle.toml"),
        "JOINT_FX_SUFFIX": "-budyko",
        "ETL_FORCE": "1",
        "ETL_REGION": reg,
        "ETL_COMPILE": os.environ.get("ETL_COMPILE", "0"),
        "ETL_MELT_DIR": str(PLATES / "LN24HA" / f"{reg.upper()}_LN24HA_2020"),
        "ETL_QUANTILE": "1",
        "ETL_WARM_FROM": str(ck),
        "ETL_TAG": "-q1ctrl" if epoques == 0 else "-q1",
        "ETL_EPOCHS": str(epoques),
        # TAUX D'APPRENTISSAGE PROPRE A LA PHASE QUANTILE (2026-09-11). Le socle
        # part a 5e-4 et decroit jusqu'a 5e-6 : c'est un reglage d'AFFINAGE du champ
        # physique. La tete de quantiles, elle, est neuve et ne porte que 1772 poids.
        # Mesure sur donnees synthetiques log-normales dont les quantiles vrais sont
        # connus : a 1e-2 elle les retrouve en 400 pas, couvertures 0,499 et 0,900 ;
        # au taux du socle elle reste bloquee sur son enveloppe de depart, d'ou les
        # couvertures de 0,13 et 0,40 observees sur la Cote-Nord.
        "ETL_LR": os.environ.get("ETL_LR", "1e-2"),
        "ETL_DUMP_Q": str(dump),
        "PYTHONIOENCODING": "utf-8",
    })
    t0 = time.time()
    with open(journal, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([sys.executable, "-u", str(RACINE / ".runs/quebec/etl_run.py")],
                           cwd=str(RACINE), env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def controle_prealable(reg):
    """Passe a ZERO epoque : le debit exporte doit etre celui de la flotte, au
    millionieme pres. Sinon la tete s'entrainerait sur un autre modele."""
    dump = SORTIE / f"ctrl-{reg}.npz"
    journal = SORTIE / f"log-ctrl-{reg}.txt"
    print(f"[quantile] CONTROLE PREALABLE : controle a zero epoque sur {reg}")
    r = lancer(reg, 0, dump, journal)
    if r is None:
        return False
    code, duree = r
    if code != 0 or not dump.exists():
        print(f"[quantile] CONTROLE PREALABLE : le pilote a echoue (code {code}). Vingt dernieres lignes :")
        print("".join(open(journal, encoding="utf-8", errors="replace").readlines()[-20:]))
        return False
    ctrl = RACINE / ".runs/quebec/controle_quantile.py"
    p = subprocess.run([sys.executable, str(ctrl), "--fichier", str(dump), "--region", reg],
                       cwd=str(RACINE))
    if p.returncode != 0:
        print(f"[quantile] CONTROLE PREALABLE REFUSE : le debit exporte ne reproduit pas celui de la "
              f"flotte. Voir {journal}.")
        return False
    print(f"[quantile] controle prealable passe en {duree/60:.1f} min.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*", default=None)
    ap.add_argument("--epoques", type=int, default=int(os.environ.get("ETL_EPOCHS", "8")))
    ap.add_argument("--sans-controle", action="store_true")
    a = ap.parse_args()
    regions = a.regions or REGIONS
    SORTIE.mkdir(parents=True, exist_ok=True)
    # Le pilote y depose son point de reprise ; le dossier est ignore par git, donc
    # absent d'un depot fraichement clone (echec observe sur la machine Ubuntu).
    (RACINE / '.runs' / 'quebec' / 'checkpoints').mkdir(parents=True, exist_ok=True)
    print(f"[quantile] interpreteur {sys.executable}")
    print(f"[quantile] {len(regions)} region(s), {a.epoques} epoques, depart {DEPART}")
    if not a.sans_controle and not controle_prealable(regions[0]):
        print("[quantile] file NON lancee.")
        raise SystemExit(6)
    for i, reg in enumerate(regions, 1):
        dump = SORTIE / f"quant-{reg}.npz"
        journal = SORTIE / f"log-quant-{reg}.txt"
        print(f"[quantile] {i}/{len(regions)} {reg} ...", flush=True)
        r = lancer(reg, a.epoques, dump, journal)
        if r is None:
            continue
        code, duree = r
        etat = "ok" if code == 0 else f"ECHEC (code {code})"
        print(f"[quantile] {reg} {etat} en {duree/60:.1f} min -> {dump.name}", flush=True)
    print("[quantile] file terminee. Controle :")
    print(f"  {sys.executable} .runs/quebec/controle_quantile.py")


if __name__ == "__main__":
    main()
