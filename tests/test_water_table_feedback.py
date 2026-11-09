"""Retroaction de transmissivite et echange souterrain de la nappe libre : eteints par defaut, et dans le bon sens."""
import torch

from meandre.utils import settings
from meandre.vertical.water_table import WaterTable


def _step(z0):
    wt = WaterTable(n_substep=4, exponent=2.0)
    z = torch.tensor([z0])
    one = torch.ones(1)
    return wt(z, recharge=torch.zeros(1), sy=0.05 * one, k_b=1e-3 * one, z_riv=5.0 * one, h_ref=2.0 * one)


def test_off_by_default_and_feedback_grows_near_surface():
    base_shallow, base_deep = _step(0.2)[1], _step(4.0)[1]
    settings.override("groundwater.transmissivity_feedback_mm_per_day", 5.0, force=True)
    try:
        fb_shallow, fb_deep = _step(0.2)[1], _step(4.0)[1]
    finally:
        settings.override("groundwater.transmissivity_feedback_mm_per_day", 0.0, force=True)
    assert float(fb_shallow - base_shallow) > 100 * float(fb_deep - base_deep) > 0


def test_exchange_loss_lowers_the_water_table():
    z_base = _step(1.0)[0]
    settings.override("groundwater.exchange_mm_per_day", 0.5, force=True)
    try:
        z_loss = _step(1.0)[0]
    finally:
        settings.override("groundwater.exchange_mm_per_day", 0.0, force=True)
    assert float(z_loss) > float(z_base)
