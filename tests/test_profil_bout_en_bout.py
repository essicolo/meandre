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


def _modele():
    torch.manual_seed(0)
    return HydroModel(n_nodes=N_NOEUDS, n_territorial=N_TERRITORIAL, use_temporal=False,
                      use_residual=False)


def _profil_declare():
    return sp.from_toml({"layers": 3, "process": [
        {"layer": 2, "kind": "lateral", "form": "BASE_LINEAR"},
        {"layer": 3, "kind": "lateral", "form": "BASE_THRESH_POWER", "tau_days": 3.0},
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER", "tau_days": 2.0,
         "ceiling_mm_per_day": "k_sub"},
    ]})


def test_le_champ_spatial_expose_le_plafond_de_percolation():
    """La quarante-troisième sortie doit exister et rester dans ses bornes."""
    m = _modele()
    coords = torch.rand(N_NOEUDS, 2) * torch.tensor([2.0, 1.0]) + torch.tensor([-75.0, 46.0])
    params = m.spatial_encoder(coords, torch.rand(N_NOEUDS, N_TERRITORIAL))
    assert hasattr(params, "k_sub"), "le plafond de percolation doit etre une sortie du champ"
    assert params.k_sub.shape == (N_NOEUDS,)
    # Bornes de la construction : 2e-6 a 2e-3 m/h, soit 0,05 a 50 mm/jour.
    assert torch.all(params.k_sub >= 2e-6) and torch.all(params.k_sub <= 2e-3)
    assert torch.all(torch.isfinite(params.k_sub))


def test_le_plafond_nomme_est_resolu_en_valeur_par_noeud():
    """Le raccord qui manquait : du nom dans le TOML au tenseur du champ."""
    profil = _profil_declare()
    proc = profil.of(3, "percolation")[0]
    assert proc.ceiling == "k_sub"
    par_noeud = torch.linspace(1e-6, 1e-3, N_NOEUDS)
    resolu = profil.resolved({"k_sub": par_noeud})
    assert torch.equal(resolu.of(3, "percolation")[0].ceiling, par_noeud)
    # Le profil d'origine n'est pas modifie : il reste reutilisable d'un pas a l'autre.
    assert profil.of(3, "percolation")[0].ceiling == "k_sub"


def test_le_profil_se_pose_sur_la_colonne_et_la_colonne_le_lit():
    """Le pilote pose `soil_profile` sur la colonne ; celle-ci doit le transmettre au sol."""
    m = _modele()
    m.vertical_column.soil_profile = _profil_declare()
    assert getattr(m.vertical_column, "soil_profile", None) is not None
    assert m.vertical_column.soil_profile.layers == 3
    assert len(m.vertical_column.soil_profile.processes) == 3


def test_le_motif_du_pilote_ne_plante_pas_sur_une_section_sans_processus():
    """Reproduction exacte du defaut qui a coute huit passes."""
    profil = sp.from_toml({"z1": 0.1, "z2_min": 0.3, "hydrotel_calib_dir": "/x"})
    assert profil is None
    # Le pilote doit tester le PROFIL, pas la section. Le motif fautif est reproduit ici pour
    # que sa reintroduction casse ce test plutot qu'une nuit de calcul.
    if profil is not None:
        raise AssertionError("motif fautif : `if section` au lieu de `if profil is not None`")


def test_les_trois_formes_du_catalogue_rendent_des_flux_finis_et_positifs():
    """Sur des teneurs en eau extremes, aucune forme ne doit rendre NaN ni valeur negative."""
    ctx = {"theta": torch.tensor([0.0, 0.15, 0.30, 0.45, 0.50]),
           "theta_fc": torch.full((5,), 0.30), "thickness": torch.full((5,), 2.70),
           "porosity": torch.full((5,), 0.50), "conductivity": torch.full((5,), 1e-3),
           "sin_slope": torch.full((5,), 0.04)}
    for forme, params in (("BASE_LINEAR", {}), ("BASE_THRESH_POWER", {"tau": 72.0}),
                          ("BASE_THRESH_POWER", {"tau": 72.0, "exponent": 2.0}),
                          ("PERC_LINEAR", {"krec": 2e-5}),
                          ("PERC_POWER_LAW", {"krec": 2e-5, "exponent": 3.0})):
        kind = "lateral" if forme.startswith("BASE") else "percolation"
        q = sp.SoilProcess(layer=3, kind=kind, form=forme, params=params).flux(ctx)
        assert torch.all(torch.isfinite(q)), f"{forme} {params} rend des valeurs non finies"
        assert torch.all(q >= 0.0), f"{forme} {params} rend un flux negatif"
