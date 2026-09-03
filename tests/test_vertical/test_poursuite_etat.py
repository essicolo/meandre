"""L'état interne de la colonne survit-il à une frontière de bloc d'entraînement ?

Le manteau neigeux par classe, le profil de gel multicouche et le volume de milieu
humide ne tiennent pas dans `HydroState` : ils vivent dans `HydrotelColumn._aux`. Comme
l'entraînement appelle `simulate` une fois par bloc de 45 jours, ces états repartaient de
zéro six fois par an, et aucun hiver continu n'a jamais été simulé pendant
l'entraînement (mesure du 2026-09-03 : 538,86 mm d'équivalent en eau effacés).

Ces tests fixent les deux comportements : sans poursuite, l'état est refabriqué, ce qui
reste le défaut historique et le bon comportement pour une simulation qui DÉMARRE ; avec
poursuite, il est conservé.
"""
import pytest
import torch

from hydrotel_clone.frost import n_intervalles
from hydrotel_clone.snow import DegreJourModifie
from meandre.spatial.field_network import SpatialFieldNetwork
from meandre.spatial.territorial import TerritorialFeatures
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params

N = 4


@pytest.fixture(autouse=True)
def _double_precision():
    """Le clone travaille en double precision ; on la pose SANS fuite vers les autres
    modules de test (un set_default_dtype global en avait casse seize)."""
    ancien = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(ancien)


def _colonne():
    occ = dict(feuillus=102 / 1754, ouverts=118 / 1754, humides=24 / 1754,
               urbain=1111 / 1754, routes=280 / 1754, eau=119 / 1754)
    psnow, psoil, petr = build_static_params(
        N, lat=45.3, slope=0.026, orientation=7, texture="sandy_loam",
        z=(0.1, 0.4, 1.0), occupation=occ)
    col = HydrotelColumn(et_mode="mcguinness", use_frost=True)
    col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))
    return col


def _swe_total(snow):
    """Équivalent en eau total (mm) : les stocks des trois classes sont en mètres."""
    return sum(float(snow[c][0].mean()) * 1000.0 for c in DegreJourModifie.CLASSES)


def _hiver(col, st, jours=45):
    """Accumule un manteau : neige tous les jours par froid franc."""
    t = lambda x: torch.full((N,), float(x))
    for i in range(jours):
        _, st, _ = col(t(4.0), t(-12.0), t(-4.0), t(8.0), t(1.5), t(0.8),
                       float((i % 365) + 1), st)
    return st


def _champ(col):
    reseau = SpatialFieldNetwork(n_territorial=17)
    coords = torch.tensor([[-74.0, 45.3 + 0.01 * i] for i in range(N)],
                          dtype=torch.float64)
    terr = TerritorialFeatures.zeros(n_nodes=N, n_features=17)
    terr.physical["area_km2_physical"] = torch.ones(N) * 10.0
    terr.physical["slope_fraction"] = torch.ones(N) * 0.026
    return reseau(coords, terr.to_tensor()), terr, coords


def test_sans_poursuite_l_etat_est_refabrique():
    """Comportement historique, conservé : une simulation qui démarre repart à neuf."""
    col = _colonne()
    st = _hiver(col, col.init_state(N, theta_init=(0.36, 0.36, 0.36)))
    assert _swe_total(st.snow) > 50.0, "45 jours de neige froide doivent bâtir un manteau"
    col._aux = st
    sp, terr, coords = _champ(col)

    col.setup_simulate(sp, terr, coords, st)

    assert _swe_total(col._aux.snow) == 0.0


def test_avec_poursuite_le_manteau_et_le_gel_survivent():
    """Le bloc suivant d'un entraînement découpé doit hériter de l'hiver précédent."""
    col = _colonne()
    st = _hiver(col, col.init_state(N, theta_init=(0.36, 0.36, 0.36)))
    swe_avant = _swe_total(st.snow)
    gel_avant = float(st.frost_profile.abs().mean())
    col._aux = st
    sp, terr, coords = _champ(col)

    col.setup_simulate(sp, terr, coords, st, poursuivre=True)

    assert _swe_total(col._aux.snow) == swe_avant
    assert float(col._aux.frost_profile.abs().mean()) == gel_avant
    # L'état conservé est DÉTACHÉ : la troncature du gradient entre blocs doit tenir,
    # sans quoi le graphe d'autograd croîtrait sur toute la séquence.
    assert not col._aux.frost_profile.requires_grad
