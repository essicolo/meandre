"""Forme saisonniere propre aux grandes cultures (phenology.crop_params)."""
import pytest


def test_crop_shape_rises_later_than_main_shape():
    """La forme des grandes cultures, levee a 500 degres-jours, reste a la dormance quand les
    feuillus (levee a 57) sont deja au plateau (2026-11-10)."""
    import torch

    from meandre.temporal.phenology_modulator import PhenologyModulator
    m = PhenologyModulator(params=[56.81, 11.82, 21.1, 0.9])
    gdd = torch.tensor([200.0])
    kc = torch.tensor([1.0])
    main = m(gdd, kc, doy=160, lat_deg=torch.tensor([45.0]))
    crop = m.crop_kc(gdd, kc, [500.0, 11.82, 100.0, 0.9], doy=160, lat_deg=torch.tensor([45.0]))
    assert float(crop) < 0.5 * float(main)
    late = m.crop_kc(torch.tensor([1200.0]), kc, [500.0, 11.82, 100.0, 0.9], doy=160, lat_deg=torch.tensor([45.0]))
    assert float(late) == pytest.approx(float(m(torch.tensor([1200.0]), kc, doy=160, lat_deg=torch.tensor([45.0]))), rel=1e-3)
