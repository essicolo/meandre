"""Tout poids qu'une configuration peut porter doit atteindre l'objet de perte.

Cinq poids ont été lus, imprimés et jamais posés : la recette équilibrée a tourné avec deux
de ses six termes sans qu'aucun journal ne le signale. Le pilote imprime ce qu'il DEMANDE,
et le bilan des composantes écarte les termes de valeur nulle, si bien qu'un terme oublié se
lit exactement comme un terme éteint volontairement.
"""
import ast
import glob
import inspect
import io
import os
import tomllib

from meandre.training.loss import HydroLoss

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTE = os.path.join(RACINE, ".runs", "quebec", "joint_data.py")
# Poses ailleurs que dans l'appel de construction, par le pilote ou la phase quantile.
HORS_APPEL = {"w_nappe", "w_quantile", "w_nll", "w_nll_et", "w_nll_swe", "w_mixture"}


def _poids_de_lappel():
    """Noms des poids passés à HydroLoss dans le chargeur de territoire."""
    arbre = ast.parse(io.open(PILOTE, encoding="utf-8").read())
    for n in ast.walk(arbre):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "HydroLoss":
            return {k.arg for k in n.keywords if k.arg and k.arg.startswith("w_")}
    raise AssertionError("aucun appel a HydroLoss dans " + PILOTE)


def test_les_poids_des_configurations_sont_tous_poses():
    poses = _poids_de_lappel()
    acceptes = set(inspect.signature(HydroLoss.__init__).parameters)
    manquants = {}
    for f in glob.glob(os.path.join(RACINE, ".runs", "quebec", "config", "*.toml")):
        section = tomllib.load(open(f, "rb")).get("loss", {})
        for cle, valeur in section.items():
            if not cle.startswith("w_") or cle not in acceptes or cle in HORS_APPEL:
                continue
            if cle not in poses:
                manquants.setdefault(cle, []).append(os.path.basename(f))
    assert not manquants, f"poids lus dans une configuration et jamais poses : {manquants}"


def test_les_termes_du_kge_decompose_sont_poses():
    poses = _poids_de_lappel()
    for cle in ("w_r", "w_beta", "w_gamma", "w_peak_ratio", "w_recession"):
        assert cle in poses, f"{cle} n'atteint pas l'objet de perte"


def test_le_seuil_de_pics_suit_les_deux_termes():
    """Le seuil doit être construit pour le terme en rapport comme pour celui en carré."""
    source = io.open(PILOTE, encoding="utf-8").read()
    debut = source.index("peak_threshold=")
    extrait = source[debut:debut + 200]
    assert "w_peak_ratio" in extrait, "le seuil n'est construit que pour le terme en carre"
