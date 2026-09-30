"""Les trois pièces du banc entrent dans le pilote par la configuration, pas par l'environnement.

Le modulateur phénologique prend son mode et ses paramètres en arguments, la colonne lit les
classes phénologiques en attribut, et le fichier de configuration candidate déclare Penman, la
phénologie et la porte de gel continue (2026-09-30).
"""
import math
import os
import tomllib

import torch

from meandre.temporal.phenology_modulator import PhenologyModulator


def test_le_modulateur_prend_ses_parametres_en_arguments():
    os.environ.pop("MEANDRE_PHENOLOGIE_MODE", None)
    os.environ.pop("MEANDRE_PHENOLOGIE_PARAMS", None)
    m = PhenologyModulator(mode="photo", params=[56.81, 11.82, 21.1, 0.9])
    assert m.mode == "photo"
    assert abs(float(m.gdd_emerg) - 56.81) < 1e-3
    seuil = 12.0 + 3.0 * math.tanh(float(m.photo_crit))
    assert abs(seuil - 11.82) < 1e-4
    assert not m.gdd_emerg.requires_grad and not m.photo_crit.requires_grad
    assert (m._s1_cal, m._s2_cal) == (21.1, 0.9)


def test_sans_argument_l_environnement_garde_son_role():
    os.environ["MEANDRE_PHENOLOGIE_MODE"] = "photo"
    os.environ["MEANDRE_PHENOLOGIE_PARAMS"] = "207.94,11.91,88.4,0.99"
    try:
        m = PhenologyModulator()
        assert m.mode == "photo" and abs(float(m.gdd_emerg) - 207.94) < 1e-3
    finally:
        os.environ.pop("MEANDRE_PHENOLOGIE_MODE", None)
        os.environ.pop("MEANDRE_PHENOLOGIE_PARAMS", None)


def test_la_configuration_candidate_declare_les_trois_pieces():
    with open(".runs/quebec/config/configuration-nuit-2026-09-30.toml", "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["et"]["formula"] == "penman"
    assert cfg["phenology"]["enabled"] and cfg["phenology"]["mode"] == "photo"
    assert cfg["phenology"]["classes"] == ["feuillus", "agri"]
    assert cfg["soil"]["frozen_gate_continuous"] is True
    assert cfg["loss"]["et_mode"] == "anomaly"
    from meandre.vertical import soil_processes as sp
    prof = sp.from_toml(cfg["soil"])
    assert prof is not None and len(prof.processes) == 3
