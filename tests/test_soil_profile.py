"""Le profil de sol déclaré doit reproduire le clone quand il décrit le clone.

C'est le garde-fou du chantier ouvert le 2026-09-20 : rendre les processus de sol déclaratifs
à la manière de Raven, sans perdre la fidélité au binaire C++ validée par UHRH à la décimale.
Deux exigences, et ce fichier les épingle. Sans déclaration, le chemin de code d'origine est
intact. Avec la déclaration qui décrit Hydrotel, la sortie est la MÊME au bit près.
"""
import pytest
import torch

from hydrotel_clone.bv3c2 import BV3C2Clone, make_params
from meandre.vertical import soil_processes as sp


def _entrees(n=4, device="cpu"):
    g = torch.Generator(device=device).manual_seed(11)
    p = make_params(device=device)
    theta = [torch.full((n,), v) for v in (0.30, 0.32, 0.34)]
    apport = torch.rand(n, generator=g) * 12.0
    etp = torch.rand(n, generator=g) * 3.0
    return p, theta, apport, etp


def _passe(p, theta, apport, etp, n_substep=32):
    col = BV3C2Clone(n_substep=n_substep)
    zero = torch.zeros_like(apport)
    return col.forward(theta[0], theta[1], theta[2], apport, etp, zero, zero, p)


def test_le_profil_hydrotel_reproduit_le_clone_au_bit_pres():
    p, theta, apport, etp = _entrees()
    avant = _passe(dict(p), theta, apport, etp)
    p2 = dict(p)
    p2["soil_profile"] = sp.hydrotel_profile(krec=p["krec"])
    apres = _passe(p2, theta, apport, etp)
    for a, b, nom in zip(avant[:4], apres[:4], ("surface", "hypodermique", "base", "recharge")):
        assert torch.equal(a, b), f"{nom} : la declaration ne reproduit pas le clone"


def test_sans_declaration_rien_ne_change():
    """Le champ absent laisse le chemin d'origine, ce qui protege les harnais de validation."""
    p, theta, apport, etp = _entrees()
    un = _passe(dict(p), theta, apport, etp)
    deux = _passe({**p, "soil_profile": None}, theta, apport, etp)
    assert all(torch.equal(a, b) for a, b in zip(un[:4], deux[:4]))


def test_une_couche_sans_processus_declare_ne_produit_aucun_flux():
    """Semantique de Raven : la declaration est complete, elle ne complete pas l'existant."""
    p, theta, apport, etp = _entrees()
    vide = sp.SoilProfile(layers=3, processes=())
    sortie = _passe({**p, "soil_profile": vide}, theta, apport, etp)
    assert torch.all(sortie[1] == 0.0), "aucun lateral declare : l'hypodermique doit etre nul"
    assert torch.all(sortie[2] == 0.0), "aucune percolation declaree : la base doit etre nulle"


def test_la_sortie_laterale_profonde_est_bien_une_forme_du_catalogue():
    """La loi ecrite a la main le 2026-09-19 est BASE_THRESH_POWER d'exposant un."""
    ctx = {"theta": torch.tensor([0.40]), "theta_fc": torch.tensor([0.30]),
           "thickness": torch.tensor([2.70]), "porosity": torch.tensor([0.50]),
           "conductivity": torch.tensor([1e-3]), "sin_slope": torch.tensor([0.04])}
    a_la_main = torch.clamp(ctx["theta"] - ctx["theta_fc"], min=0.0) * ctx["thickness"] / 48.0
    du_catalogue = sp.base_thresh_power(ctx, {"tau": 48.0})
    assert torch.allclose(a_la_main, du_catalogue)


def test_le_plafond_borne_le_flux():
    ctx = {"theta": torch.tensor([0.49]), "theta_fc": torch.tensor([0.30]),
           "thickness": torch.tensor([2.70]), "porosity": torch.tensor([0.50]),
           "conductivity": torch.tensor([1e-3]), "sin_slope": torch.tensor([0.04])}
    libre = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                           params={"tau": 48.0})
    borne = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                           params={"tau": 48.0}, ceiling=1e-6)
    assert float(borne.flux(ctx)) == pytest.approx(1e-6)
    assert float(libre.flux(ctx)) > 1e-6


def test_la_lecture_du_toml_convertit_les_unites():
    profil = sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER",
         "tau_days": 2.0, "ceiling_mm_per_day": 24.0}]})
    proc = profil.of(3, "percolation")[0]
    assert proc.params["tau"] == pytest.approx(48.0), "deux jours font quarante-huit heures"
    assert proc.ceiling == pytest.approx(1e-3), "24 mm/j font 1e-3 m/h"


def test_une_forme_inconnue_ou_mal_placee_est_refusee():
    with pytest.raises(ValueError, match="catalogue"):
        sp.SoilProcess(layer=1, kind="lateral", form="BASE_INVENTEE")
    with pytest.raises(ValueError, match="ne convient pas"):
        sp.SoilProcess(layer=3, kind="lateral", form="PERC_LINEAR", params={"krec": 1.0})
    with pytest.raises(ValueError, match="manquant"):
        sp.SoilProcess(layer=3, kind="percolation", form="PERC_LINEAR")
    with pytest.raises(ValueError, match="compte"):
        sp.SoilProfile(layers=2, processes=(sp.SoilProcess(layer=3, kind="lateral",
                                                           form="BASE_LINEAR"),))


def test_un_plafond_peut_venir_du_champ_spatial():
    """Une valeur symbolique est remplacée par le tenseur du champ, par tronçon.

    C'est le mécanisme qui rend le plafond de percolation spatial. La texture du sol le
    prédit à 27 % contre le témoin sur l'indice d'écoulement de base, là où l'exposant et
    les constantes de temps ne sont prédits par aucune covariable disponible.
    """
    par_troncon = torch.tensor([1e-6, 1e-4, 1e-3])
    profil = sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER",
         "tau_days": 2.0, "ceiling_mm_per_day": "k_sub"}]})
    proc = profil.of(3, "percolation")[0]
    assert proc.ceiling == "k_sub", "le nom reste symbolique tant qu'il n'est pas resolu"
    with pytest.raises(RuntimeError, match="non resolu"):
        proc.flux({"theta": torch.tensor([0.4]), "theta_fc": torch.tensor([0.3]),
                   "thickness": torch.tensor([2.7]), "porosity": torch.tensor([0.5]),
                   "conductivity": torch.tensor([1e-3]), "sin_slope": torch.tensor([0.04])})
    resolu = profil.resolved({"k_sub": par_troncon})
    ctx = {"theta": torch.full((3,), 0.49), "theta_fc": torch.full((3,), 0.30),
           "thickness": torch.full((3,), 2.70), "porosity": torch.full((3,), 0.50),
           "conductivity": torch.full((3,), 1e-3), "sin_slope": torch.full((3,), 0.04)}
    q = resolu.total(3, "percolation", ctx)
    assert torch.allclose(q, par_troncon), "chaque troncon doit etre borne par SON plafond"


def test_un_nom_inconnu_du_champ_est_refuse():
    profil = sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER",
         "tau_days": 2.0, "ceiling_mm_per_day": "parametre_inexistant"}]})
    with pytest.raises(KeyError, match="absent du champ"):
        profil.resolved({"k_sub": torch.tensor([1e-4])})
