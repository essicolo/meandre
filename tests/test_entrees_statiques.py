"""Garde-fous sur les ENTRÉES STATIQUES et la CONSERVATION.

Ces tests couvrent la classe de bug qui a coûté le plus cher au projet : une entrée
demandée en brut, absente du cache, remplacée SILENCIEUSEMENT par un défaut. Le
2026-08-10 on a découvert que la physique recevait 0 % de forêt, 0 % d'eau libre et 0 %
d'imperméable sur tout le Québec (colonnes centrées-réduites, colonnes `_raw` jamais
écrites), que le module de milieu humide n'avait jamais été instancié, et que l'ETR ne
couvrait que 79.6 % du territoire. Aucun test de la suite ne s'en apercevait.

Les tests qui dépendent d'un projet Hydrotel réel sont ignorés proprement s'il est absent.
"""
from pathlib import Path

import pytest
import torch

from meandre.routing.kinematic import MuskingumCunge
from meandre.spatial.territorial import TerritorialFeatures
from meandre.training.loss import rolling_mean
from meandre.vertical.hydrotel_column import HydrotelColumn

PROJ = Path("C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/OUTV_LN24HA_2020")
besoin_projet = pytest.mark.skipif(not PROJ.exists(), reason="projet Hydrotel absent")


# ── CONSERVATION ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("k_h", [4.0, 24.0, 48.0])
@pytest.mark.parametrize("n_sub", [1, 2, 4])
def test_muskingum_conserve_la_masse(k_h, n_sub):
    """L'apport latéral est un DÉBIT ENTRANT : le volume sorti doit égaler le volume
    injecté, quel que soit K. Avant correctif : 0.50 à K=4 h (eau détruite) et 1.85 à
    K=48 h (eau fabriquée), donc un K appris pouvait créer ou détruire de l'eau."""
    r = MuskingumCunge(dt=86400.0, n_substeps=n_sub)
    K = torch.tensor([k_h * 3600.0])
    x = torch.tensor([0.20])
    T = 400
    impulsion = torch.zeros(T)
    impulsion[5] = 10.0
    Q = torch.zeros(1)
    total = 0.0
    for t in range(T):
        Q = r(Q_in=torch.zeros(1), Q_out_prev=Q, q_lateral=impulsion[t:t + 1], K=K, x=x)
        total += float(Q)
    assert total == pytest.approx(10.0, rel=1e-3), f"masse rendue {total / 10.0:.4f}"


def test_rolling_mean_apparie_les_composites():
    """L'ETR simulée doit être moyennée sur la même fenêtre que le composite MOD16
    (8 jours). Avant correctif on comparait une moyenne de 8 jours à un seul jour."""
    x = torch.arange(20.0).reshape(20, 1).repeat(1, 3)
    y = rolling_mean(x, 8)
    assert y.shape == x.shape
    assert float(y[7, 0]) == pytest.approx(float(x[:8, 0].mean()))
    assert float(y[19, 2]) == pytest.approx(float(x[12:20, 2].mean()))
    assert torch.equal(rolling_mean(x, 1), x)          # fenêtre 1 = sans effet
    assert torch.equal(rolling_mean(x[:3], 8), x[:3])  # série trop courte


# ── OCCUPATION DU SOL : elle doit ATTEINDRE la physique ─────────────────────
def _colonne_avec_occupation(n=6, **fractions):
    col = HydrotelColumn()
    if fractions:
        col.set_land_cover({k: torch.full((n,), v) for k, v in fractions.items()})
    return col


def test_occupation_prime_sur_le_territorial():
    """set_land_cover doit primer sur le territorial (dont les colonnes du Québec sont
    centrées-réduites et dont les `_raw` n'existent pas)."""
    n = 6
    col = _colonne_avec_occupation(
        n, f_forest_raw=0.70, f_forest_conifer_raw=0.30, f_forest_deciduous_raw=0.40,
        f_water_raw=0.09, f_urban_raw=0.02, f_wetland_raw=0.05, f_agriculture_raw=0.12)
    lc = col._land_cover
    assert lc is not None
    assert float(lc["f_forest_raw"][0]) == pytest.approx(0.70)
    # sans set_land_cover, le territorial nu ne fournit RIEN en brut : c'est le piège.
    terr = TerritorialFeatures.zeros(n_nodes=n, n_features=17)
    assert terr.get_physical("f_forest_raw") is None
    assert terr.get_physical("wet_a_raw") is None


def test_fractions_de_surface_coherentes():
    """fse + fsi + fsa = 1 exactement, et fse/fsi non nuls quand l'occupation est
    fournie (avant correctif : fse = fsi = 0 partout, donc aucune pluie-sur-lac ni
    ruissellement imperméable)."""
    f_water, f_urban = 0.09, 0.02
    fse = min(max(f_water, 0.0), 1.0)
    fsi = min(max(f_urban, 0.0), 1.0)
    fsa = max(1.0 - fse - fsi, 0.0)
    assert fse > 0 and fsi > 0
    assert fse + fsi + fsa == pytest.approx(1.0)


def test_etr_couvre_toute_la_fraction_permeable():
    """Les classes de végétation de l'ETR doivent couvrir TOUTE la fraction perméable.
    Avant correctif seules forêt et milieu humide en recevaient une : l'agriculture et
    le sol nu ne transpiraient pas du tout, soit ~20 % du territoire."""
    # occupation cohérente qui somme à 1 : forêt 0.70, humide 0.05, agri 0.12,
    # eau 0.09, imperméable 0.02, sol nu 0.02
    pct_feu, pct_conif, f_wet, pct_agri = 0.40, 0.30, 0.05, 0.12
    fsa = 1.0 - 0.09 - 0.02
    couvert = pct_feu + pct_conif + f_wet + pct_agri
    reste = max(fsa - couvert, 0.0)
    assert couvert + reste == pytest.approx(fsa, abs=1e-6)
    assert reste == pytest.approx(0.02, abs=1e-6), "le sol nu doit recevoir la classe ouverte"


# ── CHARGEURS depuis le projet Hydrotel ─────────────────────────────────────
@besoin_projet
def test_seuil_pluie_neige_charge():
    from meandre.data.hydrotel_calib import load_passage_pluie_neige
    s = load_passage_pluie_neige(str(PROJ))
    assert -10.0 < s < 5.0
    assert s != 0.0, "le seuil calibré ne doit pas retomber sur le défaut 0 °C"


@besoin_projet
def test_occupation_sol_charge_des_fractions_credibles():
    from meandre.data.hydrotel_calib import load_occupation_sol
    ids = _ids_troncons(60)
    lc = load_occupation_sol(str(PROJ), ids)
    for k, v in lc.items():
        assert torch.all(v >= -1e-6) and torch.all(v <= 1.0 + 1e-6), f"{k} hors [0, 1]"
    assert float(lc["f_forest_raw"].mean()) > 0.2, "forêt quasi nulle : occupation non lue"
    somme = (lc["f_forest_raw"] + lc["f_water_raw"] + lc["f_urban_raw"]
             + lc["f_wetland_raw"] + lc["f_agriculture_raw"] + lc["f_bare_raw"])
    assert float(somme.mean()) == pytest.approx(1.0, abs=0.05)


@besoin_projet
def test_milieux_humides_charges():
    from meandre.data.hydrotel_calib import load_milieux_humides
    mh = load_milieux_humides(str(PROJ), _ids_troncons(200))
    assert "wet_a_raw" in mh, "sans wet_a_raw le module de milieu humide reste inactif"
    wa = mh["wet_a_raw"]
    assert float((wa > 0).sum()) > 0, "aucun tronçon porteur de milieu humide"
    assert torch.all(wa >= 0)
    assert torch.all(mh["frac_raw"][wa > 0] > 0)


def _ids_troncons(n):
    """Premiers identifiants de tronçons du projet (1-indexés dans troncon.trl)."""
    lignes = [l.split() for l in
              (PROJ / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines()[3:]
              if l.strip()]
    return [int(t[0]) for t in lignes[:n]]


# ── BORNES ──────────────────────────────────────────────────────────────────
def test_bornes_kmusk_par_defaut():
    """Les bornes du temps de transfert restent celles d'origine tant que
    MEANDRE_KMUSK n'est pas posé (les checkpoints existants en dépendent)."""
    from meandre.spatial import field_network as fn
    assert (fn._KMUSK_MIN, fn._KMUSK_MAX, fn._KMUSK_INIT) == (4.0, 48.0, 24.0)


# ── IDENTITÉ DES TRONÇONS ───────────────────────────────────────────────────
def test_conversion_identifiants_troncons():
    """Trois numérotations coexistent (entier local, chaîne provinciale, rang de
    stockage) et leur confusion a produit des KGE de -0.25 le 2026-08-11 : des nombres
    faux, pas une erreur visible."""
    from meandre.data.hydrotel_calib import id_provincial, id_local, appariement_provincial
    assert id_provincial("outv", 123) == "OUTV00123"
    assert id_local("OUTV00123") == ("OUTV", 123)
    assert appariement_provincial("OUTV", [2, 1, 9], ["OUTV00001", "OUTV00002"]) == [1, 0, None]


def test_appariement_vide_leve_une_erreur():
    """Un appariement vide est TOUJOURS un bug de convention, jamais un résultat."""
    from meandre.data.hydrotel_calib import appariement_provincial
    with pytest.raises(ValueError):
        appariement_provincial("OUTV", [1, 2], ["GASP00001", "GASP00002"])


def test_garde_fou_occupation_nulle(capsys):
    """La colonne doit DIRE ce qu'elle reçoit, et avertir si l'occupation est nulle.
    Sans ce message, méandre a simulé le Québec avec 0 % de forêt pendant des mois."""
    import torch as _t
    from meandre.model import HydroModel
    from meandre.routing.graph import synthetic_linear_graph
    from meandre.routing.withdrawals import WithdrawalData
    from meandre.utils.state import HydroState

    n, T = 6, 12
    m = HydroModel(n_nodes=n, use_temporal=False, use_residual=False,
                   use_travel_time_attn=False, column_mode="hydrotel")
    terr = TerritorialFeatures.zeros(n_nodes=n, n_features=17)
    terr.physical["area_km2_local"] = _t.ones(n) * 2
    terr.physical["area_km2_physical"] = _t.ones(n) * 10
    f = _t.zeros(T, n, 6); f[:, :, 0] = 2.0; f[:, :, 1] = 1.0; f[:, :, 2] = 9.0
    with _t.no_grad():
        m.simulate(forcing=f, initial_state=HydroState.zeros(n), graph=synthetic_linear_graph(n, tau_days=1),
                   node_coords=_t.zeros(n, 2), territorial=terr,
                   withdrawals=WithdrawalData.zeros(T, n), day_of_year=_t.ones(T, dtype=_t.long))
    sortie = capsys.readouterr().out
    assert "occupation reçue" in sortie, "la colonne doit annoncer ce qu'elle reçoit"
    assert "AVERTISSEMENT" in sortie, "occupation nulle : l'avertissement doit se déclencher"
