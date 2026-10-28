"""Observables déclarés (étape 4, 2026-10-26) : traduction vers la perte, refus de ce que personne
ne lirait, et équivalence avec la configuration du modèle gelé.
"""
import tomllib
from pathlib import Path

import pytest

from meandre import observations as obs

CONFIG = Path(__file__).resolve().parents[1] / "experiments/archive/config/modele-provincial-2026-10.toml"


def test_translation_to_the_loss():
    loss, training, others, msg = obs.translate({"et": {"weight": 0.4, "mode": "anomaly", "snow_scale_mm": 10.0}, "tws": {"weight": 0.0, "mode": "shape"}}, {"w_kge": 1.0}, {})
    assert loss["w_et"] == 0.4 and loss["et_mode"] == "anomaly" and loss["w_tws"] == 0.0 and loss["w_kge"] == 1.0
    assert training["tws_shape_only"] is True
    assert others == {"evaporation": {"mod16_snow_scale_mm": 10.0}}
    assert len(msg) == 2


def test_unread_keys_are_refused():
    with pytest.raises(ValueError, match="inconnu"):
        obs.translate({"pluie": {"weight": 1.0}}, {}, {})
    with pytest.raises(ValueError, match="clé"):
        obs.translate({"et": {"weight": 0.4, "poids": 1.0}}, {}, {})
    with pytest.raises(ValueError, match="mode"):
        obs.translate({"tws": {"weight": 0.1, "mode": "level_ish"}}, {}, {})
    with pytest.raises(ValueError, match="un seul endroit"):
        obs.translate({"et": {"weight": 0.4}}, {"w_et": 0.2}, {})


def test_observation_missing_from_domain():
    assert obs.missing({"swe_mass": {"weight": 0.3}, "et": {"weight": 0.4}}, {"et": object()}) == ["swe_mass : poids 0.3 mais le domaine ne porte pas d'observation « swe_mass »"]


def test_declaration_reproduces_the_frozen_model():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    declared = {"et": {"weight": cfg["loss"]["w_et"], "mode": cfg["loss"]["et_mode"], "snow_scale_mm": cfg["evaporation"]["mod16_snow_scale_mm"]}, "tws": {"weight": cfg["loss"]["w_tws"], "mode": "shape" if cfg["training"]["tws_shape_only"] else "level"}, "swe_mass": {"weight": cfg["loss"]["w_swe_mass"]}, "snow": {"weight": cfg["loss"]["w_snow"]}}
    without = {k: v for k, v in cfg["loss"].items() if k not in ("w_et", "et_mode", "w_tws", "w_swe_mass", "w_snow")}
    tr = {k: v for k, v in cfg["training"].items() if k != "tws_shape_only"}
    loss, training, others, _ = obs.translate(declared, without, tr)
    assert loss == cfg["loss"]
    assert training == cfg["training"]
    assert others["evaporation"]["mod16_snow_scale_mm"] == cfg["evaporation"]["mod16_snow_scale_mm"]


def test_area_ratio_is_reach_over_station_and_one_without_area():
    import numpy as np
    from meandre.observations import area_ratio
    r = area_ratio([26.8, 100.0, np.nan, 0.0], [56.6, 100.0, 50.0, 50.0])
    assert abs(r[0] - 56.6 / 26.8) < 1e-9 and r[1] == 1.0 and r[2] == 1.0 and r[3] == 1.0


def test_flagged_discharges_become_missing():
    import datetime as dt
    import numpy as np
    import torch
    from meandre.observations import mask_flagged
    times = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[D]")
    v = torch.tensor([[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]])
    m, n = mask_flagged(v, times, ["a", "b"], {"a": {dt.date(2001, 1, 2)}, "b": {dt.date(2001, 1, 2), dt.date(2001, 1, 3)}})
    assert n == 2, "seules les valeurs finies comptent"
    assert torch.isnan(m[1, 0]) and torch.isnan(m[2, 1]) and m[0, 0] == 1.0 and m[2, 0] == 5.0
