"""Drainage agricole declare (DRAIN_HOOGHOUDT) : nul sans part agricole, croissant avec la charge."""
import torch

from meandre.vertical.soil_processes import SoilProcess, from_toml


def _ctx(theta, agri):
    n = len(theta)
    return {"theta": torch.tensor(theta), "porosity": torch.full((n,), 0.45), "thickness": torch.full((n,), 0.6),
            "z_top": torch.full((n,), 0.15), "conductivity": torch.full((n,), 5e-3), "agri_frac": torch.tensor(agri)}


def test_drain_nul_sans_agriculture_et_croissant_avec_la_charge():
    p = SoilProcess(layer=2, kind="lateral", form="DRAIN_HOOGHOUDT", params={"spacing_m": 15.0, "depth_m": 1.0, "share": 0.6})
    q = p.flux(_ctx([0.30, 0.40, 0.40], [0.5, 0.5, 0.0]))
    assert float(q[2]) == 0.0
    assert float(q[1]) > float(q[0]) > 0.0


def test_drain_lu_depuis_le_toml_et_espacement_apprenable():
    prof = from_toml({"layers": 3, "process": [{"layer": 2, "kind": "lateral", "form": "DRAIN_HOOGHOUDT",
                                                          "spacing_m": 15.0, "depth_m": 1.0, "share": 0.6, "learn": ["spacing_m"]}]})
    proc = prof.processes[0]
    assert proc.learn == ("spacing_m",)
    assert proc.params["spacing_m"] == 15.0
