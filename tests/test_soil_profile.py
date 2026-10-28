"""Le profil de sol déclaré doit reproduire le clone quand il décrit le clone.

C'est le garde-fou du chantier ouvert le 2026-09-20 : rendre les processus de sol déclaratifs
à la manière de Raven, sans perdre la fidélité au binaire C++ validée par UHRH à la décimale.
Deux exigences, et ce fichier les épingle. Sans déclaration, le chemin de code d'origine est
intact. Avec la déclaration qui décrit Hydrotel, la sortie est la MÊME au bit près.
"""
import pytest
import torch

from meandre.hydrotel.bv3c2 import BV3C2Clone, make_params
from meandre.vertical import soil_processes as sp


def _entrees(n=4, device="cpu"):
    g = torch.Generator(device=device).manual_seed(11)
    p = make_params(device=device)
    theta = [torch.full((n,), v) for v in (0.30, 0.32, 0.34)]
    apport = torch.rand(n, generator=g) * 12.0
    etp = torch.rand(n, generator=g) * 3.0
    return p, theta, apport, etp


def _run_pass(p, theta, apport, etp, n_substep=32):
    col = BV3C2Clone(n_substep=n_substep)
    zero = torch.zeros_like(apport)
    return col.forward(theta[0], theta[1], theta[2], apport, etp, zero, zero, p)


def test_hydrotel_profile_reproduces_clone_bit_for_bit():
    p, theta, apport, etp = _entrees()
    avant = _run_pass(dict(p), theta, apport, etp)
    p2 = dict(p)
    p2["soil_profile"] = sp.hydrotel_profile(krec=p["krec"])
    apres = _run_pass(p2, theta, apport, etp)
    for a, b, nom in zip(avant[:4], apres[:4], ("surface", "hypodermique", "base", "recharge")):
        assert torch.equal(a, b), f"{nom} : la declaration ne reproduit pas le clone"


def test_without_declaration_nothing_changes():
    """Le champ absent laisse le chemin d'origine, ce qui protege les harnais de validation."""
    p, theta, apport, etp = _entrees()
    un = _run_pass(dict(p), theta, apport, etp)
    deux = _run_pass({**p, "soil_profile": None}, theta, apport, etp)
    assert all(torch.equal(a, b) for a, b in zip(un[:4], deux[:4]))


def test_layer_without_declared_process_produces_no_flux():
    """Semantique de Raven : la declaration est complete, elle ne complete pas l'existant."""
    p, theta, apport, etp = _entrees()
    vide = sp.SoilProfile(layers=3, processes=())
    step_output = _run_pass({**p, "soil_profile": vide}, theta, apport, etp)
    assert torch.all(step_output[1] == 0.0), "aucun lateral declare : l'hypodermique doit etre nul"
    assert torch.all(step_output[2] == 0.0), "aucune percolation declaree : la base doit etre nulle"


def test_deep_lateral_output_is_a_catalogue_form():
    """La loi ecrite a la main le 2026-09-19 est BASE_THRESH_POWER d'exposant un."""
    ctx = {"theta": torch.tensor([0.40]), "theta_fc": torch.tensor([0.30]),
           "thickness": torch.tensor([2.70]), "porosity": torch.tensor([0.50]),
           "conductivity": torch.tensor([1e-3]), "sin_slope": torch.tensor([0.04])}
    a_la_main = torch.clamp(ctx["theta"] - ctx["theta_fc"], min=0.0) * ctx["thickness"] / 48.0
    du_catalogue = sp.base_thresh_power(ctx, {"tau": 48.0})
    assert torch.allclose(a_la_main, du_catalogue)


def test_ceiling_bounds_the_flux():
    ctx = {"theta": torch.tensor([0.49]), "theta_fc": torch.tensor([0.30]),
           "thickness": torch.tensor([2.70]), "porosity": torch.tensor([0.50]),
           "conductivity": torch.tensor([1e-3]), "sin_slope": torch.tensor([0.04])}
    libre = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                           params={"tau": 48.0})
    borne = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                           params={"tau": 48.0}, ceiling=1e-6)
    assert float(borne.flux(ctx)) == pytest.approx(1e-6)
    assert float(libre.flux(ctx)) > 1e-6


def test_toml_reading_converts_units():
    profil = sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER",
         "tau_days": 2.0, "ceiling_mm_per_day": 24.0}]})
    proc = profil.of(3, "percolation")[0]
    assert proc.params["tau"] == pytest.approx(48.0), "deux jours font quarante-huit heures"
    assert proc.ceiling == pytest.approx(1e-3), "24 mm/j font 1e-3 m/h"


def test_unknown_or_misplaced_form_refused():
    with pytest.raises(ValueError, match="catalogue"):
        sp.SoilProcess(layer=1, kind="lateral", form="BASE_INVENTEE")
    with pytest.raises(ValueError, match="ne convient pas"):
        sp.SoilProcess(layer=3, kind="lateral", form="PERC_LINEAR", params={"krec": 1.0})
    with pytest.raises(ValueError, match="manquant"):
        sp.SoilProcess(layer=3, kind="percolation", form="PERC_LINEAR")
    with pytest.raises(ValueError, match="compte"):
        sp.SoilProfile(layers=2, processes=(sp.SoilProcess(layer=3, kind="lateral",
                                                           form="BASE_LINEAR"),))


def test_ceiling_can_come_from_spatial_field():
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


def test_unknown_field_name_refused():
    profil = sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER",
         "tau_days": 2.0, "ceiling_mm_per_day": "parametre_inexistant"}]})
    with pytest.raises(KeyError, match="absent du champ"):
        profil.resolved({"k_sub": torch.tensor([1e-4])})


def test_soil_section_without_process_declares_no_profile():
    """Piège évité de justesse : `[soil]` existe déjà pour `hydrotel_calib_dir`.

    Rendre un profil dès que la section est présente aurait donné un profil VIDE, c'est-à-dire
    un sol sans aucun flux, sur toutes les configurations régionales existantes, et sans un
    mot dans le journal.
    """
    assert sp.from_toml({"hydrotel_calib_dir": "/quelque/part"}) is None
    assert sp.from_toml({}) is None
    assert sp.from_toml(None) is None
    assert sp.from_toml({"layers": 3}) is None
    avec = sp.from_toml({"hydrotel_calib_dir": "/quelque/part", "layers": 3,
                         "process": [{"layer": 2, "kind": "lateral", "form": "BASE_LINEAR"}]})
    assert avec is not None and len(avec.processes) == 1


def test_description_supports_both_ceiling_types():
    """Le formatage vit dans le module et non dans le pilote, qu'aucun test ne couvre.

    Une première version imprimait le plafond avec un format numérique, ce qui plante dès
    qu'il nomme une sortie du champ spatial. Deux passes perdues le 2026-09-21.
    """
    nombre = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                            params={"tau": 48.0}, ceiling=1.25e-4)
    nomme = sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                           params={"tau": 48.0}, ceiling="k_sub")
    without = sp.SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR")
    assert "1.250e-04" in nombre.describe()
    assert "k_sub" in nomme.describe() and "champ spatial" in nomme.describe()
    assert "plafond" not in without.describe()
    for proc in (nombre, nomme, without):
        assert isinstance(proc.describe(), str) and proc.describe()


def test_gravity_percolation_equals_conductivity_and_collapses_when_dry():
    """PERC_GRAVITY : flux = K(theta), plafonné par le substratum (2026-10-27)."""
    k = torch.tensor([1e-3, 1e-6])
    ctx = {"theta": torch.tensor([0.4, 0.1]), "conductivity": k, "thickness": torch.tensor([1.0, 1.0])}
    proc = sp.SoilProcess(layer=3, kind="percolation", form="PERC_GRAVITY")
    assert torch.allclose(proc.flux(ctx), k)
    plafonne = sp.SoilProcess(layer=3, kind="percolation", form="PERC_GRAVITY", ceiling=1e-4)
    assert torch.allclose(plafonne.flux(ctx), torch.tensor([1e-4, 1e-6]))


def test_declared_conductivity_rescales_k_theta_curve():
    """`conductivity` declaree : K(theta) est rapportee a la saturation puis multipliee par la valeur
    declaree ; la forme en theta reste celle du sol (2026-10-27)."""
    ctx = {"theta": torch.tensor([0.3, 0.4]), "conductivity": torch.tensor([1e-4, 1e-2]), "k_sat": torch.tensor([1e-2, 1e-2]), "sin_slope": torch.tensor([0.1, 0.1]), "thickness": torch.tensor([1.0, 1.0])}
    sub = torch.tensor([2e-5, 4e-5])
    perc = sp.SoilProcess(layer=3, kind="percolation", form="PERC_GRAVITY", params={"conductivity": sub})
    assert torch.allclose(perc.flux(ctx), torch.tensor([2e-7, 4e-5]))
    lat = sp.SoilProcess(layer=3, kind="lateral", form="BASE_LINEAR", params={"conductivity": sub})
    assert torch.allclose(lat.flux(ctx), torch.tensor([2e-8, 4e-6]))
    without = sp.SoilProcess(layer=3, kind="lateral", form="BASE_LINEAR")
    assert torch.allclose(without.flux(ctx), torch.tensor([1e-5, 1e-3]))
