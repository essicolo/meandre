"""Parametres appris du profil de sol declare (2026-10-01).

La declaration `learn = ["tau_days", "ceiling_mm_per_day"]` doit creer un scalaire par
parametre, parti de la valeur declaree, et recevoir un gradient a travers le flux.
"""
import math

import torch

from meandre.vertical import soil_processes as sp
from meandre.vertical.hydrotel_column import HydrotelColumn


def _profile():
    return sp.from_toml({"layers": 3, "process": [
        {"layer": 3, "kind": "lateral", "form": "BASE_THRESH_POWER", "tau_days": 3.0, "learn": ["tau_days"]},
        {"layer": 3, "kind": "percolation", "form": "PERC_THRESH_POWER", "tau_days": 2.0, "ceiling_mm_per_day": 2.0, "learn": ["ceiling_mm_per_day"]},
    ]})


def test_read_toml():
    prof = _profile()
    assert prof.processes[0].learn == ("tau",)
    assert prof.processes[1].learn == ("ceiling",)


def test_scalars_created_with_starting_values():
    col = HydrotelColumn.__new__(HydrotelColumn)
    torch.nn.Module.__init__(col)
    noms = col.declare_soil_profile(_profile())
    assert noms == ["soil_learn_0_tau", "soil_learn_1_ceiling"]
    vals = col.soil_learned_values()
    assert math.isclose(vals["soil_learn_0_tau"][0], 3.0, rel_tol=1e-6)
    assert math.isclose(vals["soil_learn_1_ceiling"][0], 2.0, rel_tol=1e-6)


def test_gradient_crosses_the_flux():
    col = HydrotelColumn.__new__(HydrotelColumn)
    torch.nn.Module.__init__(col)
    col.declare_soil_profile(_profile())
    champ = {n: torch.exp(getattr(col, n)) for n in col._soil_learn_names}
    prof = col.soil_profile.resolved(champ)
    ctx = {"theta": torch.tensor([0.35]), "theta_fc": torch.tensor([0.20]), "thickness": torch.tensor([2.65])}
    q = prof.total(3, "lateral", ctx) + prof.total(3, "percolation", ctx)
    q.sum().backward()
    assert col.soil_learn_0_tau.grad is not None and float(col.soil_learn_0_tau.grad) != 0.0
    assert col.soil_learn_1_ceiling.grad is not None
