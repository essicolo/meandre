"""La phénologie sur le coefficient de culture ne s'applique qu'aux classes déclarées.

Un couvert moitié feuillus, moitié conifères, un jour d'été sans stress hydrique. La forme
saisonnière vaut 0,5. Sans restriction, toute la demande est divisée par deux ; restreinte aux
feuillus, seule leur moitié l'est, et l'évapotranspiration totale vaut trois quarts de la
demande. Les conifères gardent leur demande.
"""
import os

import torch

from hydrotel_clone.frost import n_intervalles
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params

N = 2
OCC = dict(feuillus=0.5, conifers=0.5, ouverts=0.0, humides=0.0, urbain=0.0, routes=0.0, eau=0.0, mixtes=0.0)


def _etr(forme, classes):
    torch.set_default_dtype(torch.float64)
    psnow, psoil, petr = build_static_params(N, lat=45.3, slope=0.02, orientation=7, texture="loam", z=(0.2, 0.4, 1.0), occupation=OCC)
    col = HydrotelColumn(et_mode="mcguinness", use_frost=False)
    col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))
    st = col.init_state(N, theta_init=(0.40, 0.40, 0.40))
    if classes is not None:
        os.environ["MEANDRE_PHENOLOGIE_CLASSES"] = classes
    else:
        os.environ.pop("MEANDRE_PHENOLOGIE_CLASSES", None)
    col._kc_dynamique = torch.full((N,), forme) if forme is not None else None
    T = lambda x: torch.full((N,), float(x))
    _, _, diag = col(T(0.0), T(15.0), T(28.0), T(20.0), T(2.0), T(1.5), 200.0, st)
    os.environ.pop("MEANDRE_PHENOLOGIE_CLASSES", None)
    return float(diag["etr"].mean())


def test_les_coniferes_gardent_leur_demande():
    plein = _etr(None, None)
    moitie = _etr(0.5, None)
    feuillus = _etr(0.5, "feuillus,agri")
    assert abs(moitie / plein - 0.5) < 0.05
    assert abs(feuillus / plein - 0.75) < 0.05


def test_sans_restriction_rien_ne_change():
    assert abs(_etr(0.5, None) - _etr(0.5, "feuillus,conifers")) < 1e-9
