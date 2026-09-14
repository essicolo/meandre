"""Controle du millionieme sur les exports de quantiles.

La tete de quantiles s'entraine sur un socle GELE : sa mediane est, par construction,
le debit du modele deterministe de depart. Le debit de l'export doit donc etre celui de
l'export de la flotte, valeur par valeur. Toute autre situation signifie que la tete a
ete apprise sur un AUTRE modele, et l'export ne dit alors rien du modele presente.

Le piege que ce controle existe pour attraper (2026-09-10) : les KGE MEDIANS des deux
modeles coincidaient a 0,715 en Gaspesie et les debits moyens a 0,3 % pres, alors que
les series journalieres differaient de 42 %. Un controle porte sur le score aurait
conclu que tout allait bien.

    python .runs/quebec/controle_quantile.py
"""
import os
import numpy as np

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
REP = f"{DATA}/quebec/rapport"
FLOTTE = f"{DATA}/quebec/flotte"
TAGS = ("A-v4", "B", "A")

# SEUILS, REVISES LE 2026-09-10 APRES MESURE.
# Le premier seuil pose etait l'identite au millionieme. Il est INATTEIGNABLE par
# construction : la passe de controle repart de l'etat initial du cache, alors que
# l'export de la flotte porte l'etat laisse par son propre entrainement. Mesure sur la
# Cote-Nord C, meme point de reprise et meme recette : ecart relatif median 0,12 %, max
# 5,2 %, et le defaut est SAISONNIER -- 1,0 % en janvier et fevrier, 0,05 % en juin --
# ce qui est la signature d'un etat de manteau neigeux different, non d'un autre modele.
# Le defaut que ce controle doit attraper est d'une tout autre taille : une tete apprise
# sur une physique jamais entrainee donne 27 a 62 % d'ecart median. Les deux seuils
# ci-dessous separent les deux situations d'un facteur vingt au moins.
SEUIL_MEDIAN = 0.02
SEUIL_MAX = 0.25

# CONTROLE DE L'ENVELOPPE (2026-09-11). Le controle ci-dessus ne regarde que le DEBIT
# MEDIAN : un export dont le socle est le bon mais dont la tete a diverge le passait sans
# broncher, et il a pollue une figure du rapport avec un quatre-vingt-quinzieme centile a
# 2150 fois la mediane. On verifie donc aussi que l'enveloppe est d'un ordre de grandeur
# possible pour une riviere. Bornes volontairement larges : on n'exige pas la calibration,
# seulement que la figure soit lisible.
ENV_MIN, ENV_MAX = 0.02, 20.0


def enveloppe_plausible(z):
    """Rend (ok, q05/Q median, q95/Q median)."""
    taus = list(np.round(z["quantile_taus"] if "quantile_taus" in z.files else z["taus"], 2))
    off = z["q_quantiles"] if "q_quantiles" in z.files else z["offsets"]
    Q = z["q_sim"]
    pos = np.isfinite(Q) & (Q > 1e-6)
    if not pos.any():
        return False, np.nan, np.nan
    r05 = float(np.median((Q + off[:, :, taus.index(0.05)])[pos] / Q[pos]))
    r95 = float(np.median((Q + off[:, :, taus.index(0.95)])[pos] / Q[pos]))
    return (ENV_MIN < r05 < 1.0) and (1.0 < r95 < ENV_MAX), r05, r95


def controle(reg, fichier=None):
    z = np.load(fichier or f"{REP}/quant-{reg}.npz", allow_pickle=True)
    meilleur = None
    for tag in TAGS:
        f = f"{FLOTTE}/q-{reg}-{tag}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        if d["q_sim"].shape != z["q_sim"].shape:
            continue
        m = np.isfinite(z["q_sim"]) & np.isfinite(d["q_sim"])
        if not m.any():
            continue
        rel = np.abs(z["q_sim"][m] - d["q_sim"][m]) / np.clip(np.abs(d["q_sim"][m]), 1e-9, None)
        med, pire = float(np.median(rel)), float(rel.max())
        if meilleur is None or med < meilleur[2]:
            meilleur = (tag, pire, med)
    return meilleur


def accepte(med, pire):
    return med < SEUIL_MEDIAN and pire < SEUIL_MAX


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fichier", help="un export precis a controler (mode barrage)")
    ap.add_argument("--region", help="region de cet export")
    a = ap.parse_args()
    if a.fichier:
        # MODE BARRAGE : code de sortie 0 si le debit reproduit celui de la flotte.
        r = controle(a.region, a.fichier)
        if r is None:
            print(f"  {a.region} : aucun export deterministe comparable")
            raise SystemExit(1)
        tag, pire, med = r
        if accepte(med, pire):
            print(f"  {a.region} OK : ecart median {100*med:.2f} %, max {100*pire:.1f} % "
                  f"contre q-{a.region}-{tag} (seuils {100*SEUIL_MEDIAN:.0f} et {100*SEUIL_MAX:.0f} %)")
            raise SystemExit(0)
        print(f"  {a.region} REFUSE : ecart median {100*med:.0f} %, max {100*pire:.0f} % "
              f"contre q-{a.region}-{tag}")
        raise SystemExit(1)
    fichiers = sorted(f for f in os.listdir(REP) if f.startswith("quant-") and f.endswith(".npz"))
    if not fichiers:
        print("aucun export quant-<region>.npz")
        return
    passes = 0
    for f in fichiers:
        reg = f[6:-4]
        r = controle(reg)
        if r is None:
            print(f"  {reg:<6s} AUCUN export deterministe comparable")
            continue
        tag, pire, med = r
        z = np.load(f"{REP}/{f}", allow_pickle=True)
        env_ok, r05, r95 = enveloppe_plausible(z)
        if accepte(med, pire) and env_ok:
            passes += 1
            print(f"  {reg:<6s} OK      ecart median {100*med:.2f} %, max {100*pire:.1f} % contre q-{reg}-{tag} "
                  f"| enveloppe {r05:.2f} a {r95:.2f} fois la mediane")
        elif accepte(med, pire):
            print(f"  {reg:<6s} REFUSE  socle correct mais enveloppe invraisemblable : "
                  f"{r05:.2e} a {r95:.2e} fois la mediane")
        else:
            print(f"  {reg:<6s} REFUSE  ecart median {100*med:.0f} %, max {100*pire:.0f} % "
                  f"contre q-{reg}-{tag}")
    print(f"\n{passes}/{len(fichiers)} exports utilisables.")


if __name__ == "__main__":
    main()
