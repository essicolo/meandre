"""Figure d'illustration du problème d'identifiabilité, par l'ombre d'un objet.

Idée d'Essi (2026-09-09) pour la présentation : une ombre carrée ne dit pas quel objet la
projette. Quatre solides très différents ont exactement la même empreinte au sol, donc la
même ombre sous un éclairage vertical. C'est la situation d'un modèle hydrologique calé
sur le seul débit : plusieurs jeux de paramètres reproduisent la même série observée.

La deuxième rangée dit la sortie du problème. Sous une seconde source de lumière, venue de
côté, les quatre ombres deviennent différentes : une observation supplémentaire, prise
sous un autre angle, sépare ce que la première confondait. C'est le rôle des contraintes
auxiliaires, évapotranspiration satellitaire, stockage gravimétrique, masse du couvert
nival.

    .venv/Scripts/python.exe .runs/quebec/figure_identifiabilite.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SORTIE = ".reports/quebec/images/identifiabilite.png"
GRIS = "0.55"


def _prisme(h=1.0, c=1.0):
    """Prisme droit à base carrée."""
    s = c / 2
    b = [(-s, -s), (s, -s), (s, s), (-s, s)]
    bas = [(x, y, 0.0) for x, y in b]
    haut = [(x, y, h) for x, y in b]
    faces = [bas, haut]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([bas[i], bas[j], haut[j], haut[i]])
    return faces


def _pyramide(h=1.0, c=1.0):
    """Pyramide à base carrée."""
    s = c / 2
    bas = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
    som = (0.0, 0.0, h)
    return [bas] + [[bas[i], bas[(i + 1) % 4], som] for i in range(4)]


def _tronc(h=1.0, c=1.0, r=0.35):
    """Tronc de pyramide : base carrée large, sommet carré étroit."""
    s, p = c / 2, c * r / 2
    bas = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
    haut = [(-p, -p, h), (p, -p, h), (p, p, h), (-p, p, h)]
    faces = [bas, haut]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([bas[i], bas[j], haut[j], haut[i]])
    return faces


def _escalier(h=1.0, c=1.0, n=4):
    """Socle en gradins : même empreinte, volume tout autre."""
    faces = []
    for k in range(n):
        s = (c / 2) * (1.0 - 0.18 * k)
        z0, z1 = h * k / n, h * (k + 1) / n
        b = [(-s, -s), (s, -s), (s, s), (-s, s)]
        bas = [(x, y, z0) for x, y in b]
        haut = [(x, y, z1) for x, y in b]
        faces += [bas, haut]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bas[i], bas[j], haut[j], haut[i]])
    return faces


OBJETS = [("prisme droit", _prisme()), ("pyramide", _pyramide()),
          ("tronc de pyramide", _tronc()), ("socle en gradins", _escalier())]


def _ombre_sol(c=1.0, z=0.0):
    s = c / 2
    return [[(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)]]


def _leve(faces, dz):
    """Souleve un solide pour degager son ombre au sol, qui doit rester visible."""
    return [[(x, y, z + dz) for x, y, z in f] for f in faces]


def _sol(c=3.0, z=0.0):
    s = c / 2
    return [[(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)]]


def _silhouette_mur(faces, x_mur=-1.15):
    """Ombre portée sur un mur vertical : enveloppe de la projection sur le plan (y, z)."""
    pts = np.array([p for f in faces for p in f])
    y, z = pts[:, 1], pts[:, 2]
    from scipy.spatial import ConvexHull
    h = ConvexHull(np.column_stack([y, z]))
    v = h.vertices
    return [[(x_mur, float(y[i]), float(z[i])) for i in list(v) + [v[0]]]]


def main():
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    fig = plt.figure(figsize=(15.5, 7.0))
    for col, (nom, faces) in enumerate(OBJETS):
        for rang in (0, 1):
            ax = fig.add_subplot(2, 4, rang * 4 + col + 1, projection="3d")
            DZ = 0.85 if rang == 0 else 0.0
            if rang == 0:
                # Pas de plan de sol : matplotlib trie les polygones par distance de leur
                # centre, et un grand plan masquait l'ombre qu'il devait porter. L'ombre
                # seule sur fond blanc se lit mieux et ne peut pas etre occultee.
                ax.add_collection3d(Poly3DCollection(_ombre_sol(), facecolor=GRIS,
                                                     edgecolor="none", alpha=.95, zorder=0))
            else:
                ax.add_collection3d(Poly3DCollection(_silhouette_mur(faces), facecolor=GRIS,
                                                     edgecolor="none", alpha=.9))
            ax.add_collection3d(Poly3DCollection(_leve(faces, DZ), facecolor="white",
                                                 edgecolor="0.25", linewidths=.9, alpha=1.0))
            if rang == 0:
                ax.quiver(0, 0, 2.55, 0, 0, -0.45, color="0.3", arrow_length_ratio=.4, lw=1.5)
            else:
                ax.quiver(-2.0, 0, 0.5, 0.55, 0, 0, color="0.3", arrow_length_ratio=.4, lw=1.5)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-0.05, 2.35)
            ax.set_box_aspect((1, 1, 1.1))
            ax.view_init(elev=22, azim=-58)
            ax.set_axis_off()

    for col, (nom, _f) in enumerate(OBJETS):
        fig.text(0.125 + col * 0.25, 0.885, nom, ha="center", fontsize=12.5)
    fig.text(0.5, 0.965, "Une seule observation ne détermine pas le modèle",
             ha="center", fontsize=17)
    fig.text(0.5, 0.928, "Lumière verticale : les quatre solides projettent la même ombre carrée",
             ha="center", fontsize=12, color="0.3")
    fig.text(0.5, 0.455, "Une seconde observation, prise sous un autre angle, les sépare",
             ha="center", fontsize=17)
    fig.text(0.5, 0.415, "Lumière latérale : les quatre ombres portées deviennent différentes",
             ha="center", fontsize=12, color="0.3")
    fig.subplots_adjust(left=.005, right=.995, top=.875, bottom=.01, hspace=.12, wspace=.0)
    fig.savefig(SORTIE, dpi=200, facecolor="white")
    print(f"{SORTIE} ecrit ({os.path.getsize(SORTIE)/1024:.0f} ko)")


if __name__ == "__main__":
    main()
