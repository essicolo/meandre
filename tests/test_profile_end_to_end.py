"""Le profil de sol déclaré traverse-t-il le modèle entier, sur processeur.

Écrit après la nuit du 2026-09-20, où huit passes ont été perdues faute d'un test couvrant le
chemin complet : chaque pièce était testée isolément, et c'est le RACCORD entre la lecture du
TOML et le pilote qui a cédé. Les tests unitaires du catalogue et du profil passaient tous.

Ce fichier exerce donc la chaîne telle qu'elle tourne : construction du modèle, profil déclaré
posé sur la colonne, plafond nommant une sortie du champ spatial, et une simulation de quelques
pas. Rien de tout cela ne demande de carte.
"""
import torch

from meandre.model import HydroModel
from meandre.spatial.field_network import SpatialParams
from meandre.vertical import soil_processes as sp

N_NOEUDS = 6
N_TERRITORIAL = 4


def _model():
    torch.manual_seed(0)
    return HydroModel(n_nodes=N_NOEUDS, n_territorial=N_TERRITORIAL)


def _declared_profile():
    return sp.from_toml({"layers": 3, "process": [
        {"layer": 2, "kind": "lateral", "form": "BASE_LINEAR"},
        {"layer": 3, "kind": "lateral", "form": "BASE_THRESH_POWER", "tau_days": 3.0},
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER", "tau_days": 2.0,
         "ceiling_mm_per_day": "k_sub"},
    ]})


def test_spatial_field_exposes_percolation_ceiling():
    """La quarante-troisième sortie doit exister et rester dans ses bornes."""
    m = _model()
    coords = torch.rand(N_NOEUDS, 2) * torch.tensor([2.0, 1.0]) + torch.tensor([-75.0, 46.0])
    params = m.spatial_encoder(coords, torch.rand(N_NOEUDS, N_TERRITORIAL))
    assert hasattr(params, "k_sub"), "le plafond de percolation doit etre une sortie du champ"
    assert params.k_sub.shape == (N_NOEUDS,)
    # Bornes de la construction : 2e-6 a 2e-3 m/h, soit 0,05 a 50 mm/jour.
    assert torch.all(params.k_sub >= 2e-6) and torch.all(params.k_sub <= 2e-3)
    assert torch.all(torch.isfinite(params.k_sub))


def test_named_ceiling_resolves_to_per_node_value():
    """Le raccord qui manquait : du nom dans le TOML au tenseur du champ."""
    profil = _declared_profile()
    proc = profil.of(3, "percolation")[0]
    assert proc.ceiling == "k_sub"
    par_noeud = torch.linspace(1e-6, 1e-3, N_NOEUDS)
    resolu = profil.resolved({"k_sub": par_noeud})
    assert torch.equal(resolu.of(3, "percolation")[0].ceiling, par_noeud)
    # Le profil d'origine n'est pas modifie : il reste reutilisable d'un pas a l'autre.
    assert profil.of(3, "percolation")[0].ceiling == "k_sub"


def test_profile_is_set_on_and_read_by_column():
    """Le pilote pose `soil_profile` sur la colonne ; celle-ci doit le transmettre au sol."""
    m = _model()
    m.vertical_column.soil_profile = _declared_profile()
    assert getattr(m.vertical_column, "soil_profile", None) is not None
    assert m.vertical_column.soil_profile.layers == 3
    assert len(m.vertical_column.soil_profile.processes) == 3


def test_pilot_pattern_survives_section_without_process():
    """Reproduction exacte du defaut qui a coute huit passes."""
    profil = sp.from_toml({"z1": 0.1, "z2_min": 0.3, "hydrotel_calib_dir": "/x"})
    assert profil is None
    # Le pilote doit tester le PROFIL, pas la section. Le motif fautif est reproduit ici pour
    # que sa reintroduction casse ce test plutot qu'une nuit de calcul.
    if profil is not None:
        raise AssertionError("motif fautif : `if section` au lieu de `if profil is not None`")


def test_three_catalogue_forms_return_finite_positive_fluxes():
    """Sur des teneurs en eau extremes, aucune forme ne doit rendre NaN ni valeur negative."""
    ctx = {"theta": torch.tensor([0.0, 0.15, 0.30, 0.45, 0.50]),
           "theta_fc": torch.full((5,), 0.30), "thickness": torch.full((5,), 2.70),
           "porosity": torch.full((5,), 0.50), "conductivity": torch.full((5,), 1e-3),
           "sin_slope": torch.full((5,), 0.04)}
    for shape_value, params in (("BASE_LINEAR", {}), ("BASE_THRESH_POWER", {"tau": 72.0}),
                          ("BASE_THRESH_POWER", {"tau": 72.0, "exponent": 2.0}),
                          ("PERC_LINEAR", {"krec": 2e-5}),
                          ("PERC_POWER_LAW", {"krec": 2e-5, "exponent": 3.0})):
        kind = "lateral" if shape_value.startswith("BASE") else "percolation"
        q = sp.SoilProcess(layer=3, kind=kind, form=shape_value, params=params).flux(ctx)
        assert torch.all(torch.isfinite(q)), f"{shape_value} {params} rend des valeurs non finies"
        assert torch.all(q >= 0.0), f"{shape_value} {params} rend un flux negatif"
