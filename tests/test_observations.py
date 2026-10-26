"""Observables déclarés (étape 4, 2026-10-26) : traduction vers la perte, refus de ce que personne
ne lirait, et équivalence avec la configuration du modèle gelé.
"""
import tomllib
from pathlib import Path

import pytest

from meandre import observations as obs

CONFIG = Path(__file__).resolve().parents[1] / ".runs/quebec/config/modele-provincial-2026-10.toml"


def test_traduction_vers_la_perte():
    loss, training, others, msg = obs.translate({"et": {"weight": 0.4, "mode": "anomaly", "snow_scale_mm": 10.0}, "tws": {"weight": 0.0, "mode": "shape"}}, {"w_kge": 1.0}, {})
    assert loss["w_et"] == 0.4 and loss["et_mode"] == "anomaly" and loss["w_tws"] == 0.0 and loss["w_kge"] == 1.0
    assert training["tws_shape_only"] is True
    assert others == {"evaporation": {"mod16_snow_scale_mm": 10.0}}
    assert len(msg) == 2


def test_ce_que_personne_ne_lirait_est_refuse():
    with pytest.raises(ValueError, match="inconnu"):
        obs.translate({"pluie": {"weight": 1.0}}, {}, {})
    with pytest.raises(ValueError, match="clé"):
        obs.translate({"et": {"weight": 0.4, "poids": 1.0}}, {}, {})
    with pytest.raises(ValueError, match="mode"):
        obs.translate({"tws": {"weight": 0.1, "mode": "level_ish"}}, {}, {})
    with pytest.raises(ValueError, match="un seul endroit"):
        obs.translate({"et": {"weight": 0.4}}, {"w_et": 0.2}, {})


def test_observation_absente_du_domaine():
    assert obs.missing({"swe_mass": {"weight": 0.3}, "et": {"weight": 0.4}}, {"et": object()}) == ["swe_mass : poids 0.3 mais le domaine ne porte pas d'observation « swe_mass »"]


def test_la_declaration_reproduit_le_modele_gele():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    declared = {"et": {"weight": cfg["loss"]["w_et"], "mode": cfg["loss"]["et_mode"], "snow_scale_mm": cfg["evaporation"]["mod16_snow_scale_mm"]}, "tws": {"weight": cfg["loss"]["w_tws"], "mode": "shape" if cfg["training"]["tws_shape_only"] else "level"}, "swe_mass": {"weight": cfg["loss"]["w_swe_mass"]}, "snow": {"weight": cfg["loss"]["w_snow"]}}
    sans = {k: v for k, v in cfg["loss"].items() if k not in ("w_et", "et_mode", "w_tws", "w_swe_mass", "w_snow")}
    tr = {k: v for k, v in cfg["training"].items() if k != "tws_shape_only"}
    loss, training, others, _ = obs.translate(declared, sans, tr)
    assert loss == cfg["loss"]
    assert training == cfg["training"]
    assert others["evaporation"]["mod16_snow_scale_mm"] == cfg["evaporation"]["mod16_snow_scale_mm"]
