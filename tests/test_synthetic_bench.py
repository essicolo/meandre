"""Banc fictif : la réponse d'orage se lit contre le témoin sans orages."""
import numpy as np
import pandas as pd
import torch

from meandre.probes.synthetic_bench import _control_forcing, metrics


def _case():
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    storms = pd.DataFrame({"date": ["2001-06-10", "2001-07-10", "2001-08-10"], "pluie_mm": [10.0, 20.0, 60.0], "apres_pluie": [False, False, False]})
    storms = pd.concat([storms, pd.DataFrame({"date": ["2001-09-10"], "pluie_mm": [20.0], "apres_pluie": [True]})], ignore_index=True)
    return dates, storms


def test_coefficient_is_storm_response_above_control():
    dates, storms = _case()
    n = 3
    base = np.full((len(dates), n), 1.0)
    lateral = base.copy()
    for _, s in storms.iterrows():
        t = int(np.flatnonzero(dates == pd.Timestamp(s.date))[0])
        lateral[t:t + 3] += 0.1 * s.pluie_mm / 3.0
    etr = np.ones_like(base)
    m = metrics(dates, storms, (lateral, etr), (base, etr))
    assert np.allclose(m["coef_sec_10"], 0.1)
    assert np.allclose(m["coef_sec_60"], 0.1)
    assert np.allclose(m["pente_pluie"], 0.0, atol=1e-9)
    assert np.allclose(m["rapport_humide_sec_20"], 1.0)


def test_control_forcing_removes_only_storm_rain():
    dates, storms = _case()
    f = torch.ones(len(dates), 2, 6)
    c = _control_forcing(f, dates, storms)
    days = np.isin(dates.strftime("%Y-%m-%d"), storms.date.to_numpy())
    assert float(c[days, :, 0].abs().sum()) == 0.0
    assert float(c[~days, :, 0].sum()) == float(f[~days, :, 0].sum())
    assert torch.equal(c[:, :, 1:], f[:, :, 1:])


def test_lake_k_factor_is_set_then_restored():
    from types import SimpleNamespace

    from meandre.probes.synthetic_bench import apply_variant
    model = SimpleNamespace(spatial_encoder=SimpleNamespace(), vertical_column=SimpleNamespace())
    undo = apply_variant(model, {"name": "lacs", "lake_k_factor": 3})
    assert model.spatial_encoder.lake_k_factor == 3.0
    undo()
    assert model.spatial_encoder.lake_k_factor is None
