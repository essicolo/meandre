"""Ecoulement rapide sous averse : la colonne sait-elle produire un pic d'ete ?

Le pic simule vaut 0,59 du pic observe, et hors crue printaniere 0,48 a 0,55 selon la
region, alors que la pluie d'entree cumulee sur les quatre jours qui precedent l'evenement
n'a pas de biais mesurable, -0,07 en logarithme sur la Monteregie avec cinq jauges par
bassin. Le deficit est donc dans la transformation pluie-debit, non dans la lame recue.

Ce banc isole la question sans reseau, sans routage et sans entrainement : une colonne
BV3C2 seule recoit une averse d'un jour sur un sol a humidite donnee, et on lit la part
qui sort en ruissellement de surface. Trois leviers la gouvernent :

  K_sat de la premiere couche, qui reste au champ par la loi des ancrages, donc apprenable ;
  duree effective d'orage, qui active l'exces d'infiltration sous-journalier deja code dans
    le clone (`storm_hours`) et absent du forcage quebecois, lequel n'a que six canaux ;
  humidite anterieure du sol.

Reference : sous averse convective d'ete en foret boreale, le coefficient de ruissellement
evenementiel se situe entre 5 et 30 pour cent selon l'humidite anterieure. Une colonne qui
rend moins de 5 pour cent ne peut pas produire de pointe d'ete, quelle que soit la pluie.

    .venv/Scripts/python.exe .runs/quebec/banc_ruissellement.py
"""
import os

import torch

from hydrotel_clone.bv3c2 import BV3C2Clone, SOIL_TEXTURES, EPAISSEUR, KREC_DEFAULT, CIN_DEFAULT

AVERSES = (20.0, 40.0, 80.0)
HUMIDITES = (0.5, 0.7, 0.9)
KS_FACTEURS = (0.25, 1.0, 4.0)
ORAGES = (None, 6.0, 2.0)
TEXTURE = "silt_loam"
torch.set_default_dtype(torch.float64)


def params(ks_facteur):
    """Le constructeur officiel du clone pose toutes les cles derivees ; on ne modifie
    ensuite que la conductivite de la premiere couche, seul parametre reste au champ."""
    from hydrotel_clone.bv3c2 import make_params
    p = make_params(TEXTURE, TEXTURE, TEXTURE, slope=0.05, fsa=1.0, fse=0.0, fsi=0.0,
                    krec=KREC_DEFAULT, cin=CIN_DEFAULT, coef_recharge=0.0)
    z1, z2, z3 = EPAISSEUR
    for k, v in (("z1", z1), ("z2", z2), ("z3", z3)):
        p[k] = torch.full((1,), float(v))
    p = {k: (v if torch.is_tensor(v) and v.numel() == 1 else
             (torch.full((1,), float(v)) if not torch.is_tensor(v) else v[:1].clone()))
         for k, v in p.items()}
    p["ks1"] = p["ks1"] * ks_facteur
    return p


def part_surface(pluie_mm, theta_rel, ks_facteur, orage, n_substep=64):
    os.environ["MEANDRE_NSUBSTEP"] = str(n_substep)
    cl = BV3C2Clone()
    p = params(ks_facteur)
    tex = SOIL_TEXTURES[TEXTURE]
    t = tex["thetapf"] + theta_rel * (tex["thetas"] - tex["thetapf"])
    th = [torch.full((1,), float(t)) for _ in range(3)]
    sh = None if orage is None else torch.full((1,), float(orage))
    out = cl.forward(th[0], th[1], th[2], torch.full((1,), float(pluie_mm)),
                     torch.zeros(1), torch.zeros(1), torch.zeros(1), p, storm_hours=sh)
    surf, hypo = float(out[0].item()), float(out[1].item())
    return 100.0 * surf / max(pluie_mm, 1e-9), 100.0 * hypo / max(pluie_mm, 1e-9)


def main():
    print(f"Colonne BV3C2 seule, texture {TEXTURE}. Part de l'averse sortie en ruissellement")
    print("de SURFACE le jour meme, en pour cent. Attendu 5 a 30 % sous averse convective.\n")
    for orage in ORAGES:
        nom = "clone fidèle, pas journalier" if orage is None else f"orage de {orage:.0f} h"
        print(f"  {nom}")
        print(f"    {'averse':>9s} " + " ".join(f"{'K_sat x' + str(k):>13s}" for k in KS_FACTEURS))
        for hum in HUMIDITES:
            print(f"    humidité relative du sol {hum:.1f}")
            for pl in AVERSES:
                cases = []
                for kf in KS_FACTEURS:
                    s, _ = part_surface(pl, hum, kf, orage)
                    cases.append(f"{s:12.1f}%")
                print(f"    {pl:6.0f} mm " + " ".join(cases))
        print()


if __name__ == "__main__":
    main()
