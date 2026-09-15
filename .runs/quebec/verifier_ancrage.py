"""Controle de l'ancrage geometrique du routage, sans pytest.

Reprend les sept verifications du fichier de tests, pour un environnement ou pytest n'est
pas installe, comme celui de la grappe. Sort en code 0 si tout passe, 1 sinon.

    python .runs/quebec/verifier_ancrage.py
"""
import math
import os
import sys

# Lancer un script d'un sous-dossier met CE sous-dossier en tete du chemin d'import, pas
# la racine du depot : sans cette ligne, `import meandre` echoue sur la grappe.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from meandre.spatial.field_network import SpatialFieldNetwork

ECHECS = []


def verifie(nom, condition, detail=""):
    if condition:
        print(f"  [ok]    {nom}")
    else:
        print(f"  [ECHEC] {nom}  {detail}")
        ECHECS.append(nom)


def champ(n):
    torch.manual_seed(0)
    f = SpatialFieldNetwork(n_territorial=16, n_nodes=n)
    f.eval()
    return f


def entrees(n):
    torch.manual_seed(1)
    return torch.randn(n, 2) * 2 + torch.tensor([-72.0, 50.0]), torch.randn(n, 16)


def identiques(n):
    return torch.tensor([[-72.0, 50.0]]).repeat(n, 1), torch.zeros(n, 16)


def main():
    print("Contrôle de l'ancrage géométrique du temps de transfert Muskingum")

    f, (c, t) = champ(64), entrees(64)
    with torch.no_grad():
        a = f(c, t).K_musk_hours.clone()
    f.set_routing_anchor(None)
    with torch.no_grad():
        b = f(c, t).K_musk_hours
    verifie("sans ancre, comportement inchangé", bool(torch.equal(a, b)))

    f, (c, t) = champ(2), identiques(2)
    f.set_routing_anchor(torch.tensor([5.0, 10.0]))
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    r = float(k[1] / k[0])
    verifie("deux fois plus long, deux fois plus lent", abs(r - 2.0) < 1e-3, f"rapport {r:.4f}")

    f, (c, t) = champ(2), identiques(2)
    f.set_routing_anchor(torch.tensor([7.0, 7.0]), slope_frac=torch.tensor([0.005, 0.020]))
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    r = float(k[0] / k[1])
    verifie("quatre fois plus de pente, deux fois plus rapide", abs(r - 2.0) < 5e-3,
            f"rapport {r:.4f}")

    f, (c, t) = champ(1), entrees(1)
    f.set_routing_anchor(torch.tensor([7.1]))
    with torch.no_grad():
        k = float(f(c, t).K_musk_hours[0])
    verifie("tronçon médian de 7,1 km : temps physique", 0.3 < k < 6.0, f"{k:.2f} h")

    f, (c, t) = champ(64), entrees(64)
    L = torch.full((64,), 7.1)
    f.set_routing_anchor(L)
    anc = f._k_musk_anchor
    ok = True
    for g in (-50.0, 50.0):
        with torch.no_grad():
            f.fc_out.bias.fill_(g)
            k = f(c, t).K_musk_hours
        ok &= bool(torch.all(k >= 0.05) and torch.all(k <= 48.0))
        ok &= bool(torch.all(k <= anc * math.e ** 1.5 + 1e-4))
    verifie("la modulation reste bornée", ok)

    f, (c, t) = champ(64), entrees(64)
    torch.manual_seed(2)
    f.set_routing_anchor(torch.rand(64) * 20 + 0.5)
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    cv = float(k.std() / k.mean())
    verifie("l'ancre varie d'un tronçon à l'autre", cv > 0.3, f"coefficient de variation {cv:.2f}")

    f, (c, t) = champ(8), entrees(8)
    f.set_routing_anchor(torch.full((8,), 7.1))
    f(c, t).K_musk_hours.sum().backward()
    g = f.fc_out.weight.grad
    verifie("le gradient passe par la modulation",
            g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0)

    f, (c, t) = champ(4), entrees(4)
    with torch.no_grad():
        libre = f(c, t).K_musk_hours.clone()
    f.set_routing_anchor(torch.tensor([5.0, float("nan"), 5.0, float("nan")]))
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    verifie("un nœud NaN garde le paramètre borné libre",
            bool(torch.allclose(k[1], libre[1]) and torch.allclose(k[3], libre[3])
                 and not torch.allclose(k[0], libre[0]) and torch.isfinite(k).all()))

    f, (c, t) = champ(64), entrees(64)
    torch.manual_seed(3)
    f.set_routing_anchor(torch.full((64,), 5.0), slope_frac=torch.rand(64) * 0.10 + 0.02)
    m = float(f._k_musk_anchor.median())
    verifie("la pente de référence vaut la médiane des pentes fournies",
            abs(m - 5000.0 / (5.0 / 3.0) / 3600.0) < 0.05, f"ancre médiane {m:.3f} h")

    print(f"\n{'CONTRÔLE RÉUSSI' if not ECHECS else 'CONTRÔLE ÉCHOUÉ : ' + ', '.join(ECHECS)}")
    return 1 if ECHECS else 0


if __name__ == "__main__":
    sys.exit(main())
