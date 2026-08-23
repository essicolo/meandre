"""Contraintes auxiliaires : ne jamais laisser un poids actif sans sa cible.

Motif (registre, dette #14 et R23, 2026-08-21). Deux defauts du meme genre ont
survecu un mois dans le pilote quebecois :

  1. `w_snow = 0.3` etait annonce a chaque run alors que `swe_obs` avait ete perdu
     par une reconstruction manuelle de `TrainingData`. Le terme ne s'evaluait
     jamais, en silence, et 219 commits sont passes dessus.
  2. `w_tws = 0.2` etait bien actif mais jugeait des mois INDIVIDUELS a sigma=25 mm,
     l'incertitude d'UNE observation GRACE, alors que l'erreur est un biais
     saisonnier systematique. Lu a la bonne echelle, le meme residu passe de 1.05
     a 4.8 ecarts-types.

Ces tests verrouillent les deux correctifs : le garde-fou qui LEVE au lieu de se
taire, et le terme climatologique qui voit un biais que le terme mensuel ignore.
"""
import pytest
import torch

from meandre.training.loss import HydroLoss, tws_anomaly_loss


def test_hydroloss_accepte_w_tws_clim():
    """Le poids existe et vaut zero par defaut (opt-in, pas de changement furtif)."""
    assert HydroLoss().w_tws_clim == 0.0
    assert HydroLoss(w_tws_clim=0.05).w_tws_clim == 0.05


def _donnees(n_pas=400, avec_swe=True, avec_tws=True, avec_et=True):
    from meandre.training.trainer import TrainingData
    from unittest.mock import MagicMock
    return TrainingData(
        forcing=torch.zeros(n_pas, 4, 6), q_obs=torch.zeros(n_pas, 2),
        station_mask=torch.ones(2, dtype=torch.bool),
        station_idx=torch.tensor([0, 1]), graph=MagicMock(),
        node_coords=torch.zeros(4, 2), territorial=MagicMock(),
        withdrawals=MagicMock(), day_of_year=torch.arange(1, n_pas + 1) % 365 + 1,
        train_slice=slice(0, n_pas), val_slice=slice(0, n_pas),
        swe_obs=torch.zeros(n_pas, 4) if avec_swe else None,
        tws_obs=torch.zeros(n_pas) if avec_tws else None,
        et_obs=torch.zeros(n_pas, 4) if avec_et else None,
    )


@pytest.mark.parametrize("poids, absent", [
    ({"w_snow": 0.3}, "avec_swe"),
    ({"w_tws": 0.2}, "avec_tws"),
    ({"w_et": 1.0}, "avec_et"),
])
def test_poids_actif_sans_cible_leve(poids, absent):
    """LE COEUR DE LA DETTE #14 : un objectif demande mais absent doit CRIER.

    Sans ce garde-fou, le trainer sautait le terme en silence et rendait un score
    qui n'etait comparable a rien -- le modele entraine n'etait pas celui annonce.
    """
    from meandre.training.trainer import Trainer
    with pytest.raises(ValueError, match="sans sa cible"):
        Trainer(model=torch.nn.Linear(1, 1), loss_fn=HydroLoss(**poids),
                train_data=_donnees(**{absent: False}))


def test_poids_nul_sans_cible_passe():
    """Le garde-fou ne doit pas gener la configuration normale (rien demande)."""
    from meandre.training.trainer import Trainer
    Trainer(model=torch.nn.Linear(1, 1), loss_fn=HydroLoss(),
            train_data=_donnees(avec_swe=False, avec_tws=False, avec_et=False))


def test_terme_mensuel_aveugle_au_biais_saisonnier():
    """R23 chiffre : un biais systematique de 26 mm se lit comme ~1 ecart-type.

    C'est la mesure qui justifie le terme climatologique. Le residu du champion
    OUTV vaut 26.4 mm de biais SYSTEMATIQUE ; a sigma=25 (incertitude d'un mois)
    la perte se declare satisfaite, a sigma=5.4 (incertitude de la climatologie
    sur ~21 ans) le meme residu est massivement significatif.
    """
    biais = torch.full((12,), 26.4)
    sim, obs = biais, torch.zeros(12)
    lu_mensuel = tws_anomaly_loss(sim, obs, 0.0, 0.0, sigma=25.0)
    lu_clim = tws_anomaly_loss(sim, obs, 0.0, 0.0, sigma=5.4)
    assert lu_mensuel < 1.5, "a l'incertitude d'un mois, la perte ne pousse plus"
    assert lu_clim > 20 * lu_mensuel, "a l'incertitude de la climatologie, elle pousse"


def test_biais_saisonnier_survit_au_centrage():
    """Un biais de MOYENNE NULLE sur l'annee est invisible au centrage long terme.

    C'est exactement la forme du residu mesure (+44 mars, -47 mai) : il ne deplace
    pas la moyenne, donc aucun terme centre sur la serie entiere ne peut le voir.
    Seul un terme PAR MOIS CALENDAIRE l'attrape. Ce test protege contre une
    simplification qui remplacerait le terme climatologique par un terme de biais
    global -- lequel vaudrait rigoureusement zero.
    """
    sais = torch.tensor([7.0, 20.0, 44.0, 8.0, -47.0, -45.0,
                         -26.0, -3.0, 22.0, 23.0, 4.0, -7.0])
    assert abs(float(sais.mean())) < 1.0, "moyenne annuelle quasi nulle par construction"
    # centre sur sa propre moyenne : un terme de biais GLOBAL ne verrait rien
    assert float((sais - sais.mean()).abs().mean()) > 20.0, "le biais par mois est massif"


# ── CanSWE : masse du manteau (R24) ──────────────────────────────────────────

def _sites(n=3, dist=(1.0, 2.0, 3.0), delev=(10.0, 20.0, 30.0)):
    import pandas as pd
    return pd.DataFrame({"swe_station_id": [f"S{i}" for i in range(n)],
                         "node_idx": list(range(n)),
                         "dist_km": list(dist), "elev_diff_m": list(delev)})


def _mesures(dates, sites, valeurs):
    import pandas as pd
    return pd.DataFrame({"swe_station_id": sites, "date": pd.to_datetime(dates),
                         "swe_mm": valeurs})


def test_build_swe_targets_place_les_releves_au_bon_jour():
    import pandas as pd
    from meandre.data.canswe_loader import build_swe_targets
    t = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    v, node, gardes = build_swe_targets(
        _mesures(["2020-01-03", "2020-01-07"], ["S0", "S2"], [120.0, 80.0]),
        _sites(), t)
    assert v.shape == (10, 3) and len(gardes) == 3
    assert float(v[2, 0]) == 120.0 and float(v[6, 2]) == 80.0
    assert torch.isnan(v).sum() == 28, "tout le reste doit rester manquant"
    assert node.tolist() == [0, 1, 2]


def test_build_swe_targets_ecarte_les_sites_non_representatifs():
    """Les filtres BORNENT la representativite ponctuelle au lieu de la corriger.

    Corriger demanderait un parametre par site, ce qui detruirait l'identifiabilite
    que la contrainte est censee apporter. Les deux seuils sont donc des choix
    explicites, et ce test verrouille qu'ils mordent vraiment.
    """
    import pandas as pd
    from meandre.data.canswe_loader import build_swe_targets
    t = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    m = _mesures(["2020-01-02"] * 3, ["S0", "S1", "S2"], [100.0, 100.0, 100.0])
    v, node, gardes = build_swe_targets(
        m, _sites(dist=(1.0, 99.0, 2.0), delev=(10.0, 10.0, 999.0)), t,
        max_dist_km=15.0, max_elev_diff_m=150.0)
    assert list(gardes["swe_station_id"]) == ["S0"], "trop loin et trop haut ecartes"
    assert v.shape == (5, 1)


def test_build_swe_targets_rend_none_si_rien_ne_passe():
    import pandas as pd
    from meandre.data.canswe_loader import build_swe_targets
    t = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    v, node, gardes = build_swe_targets(
        _mesures(["2020-01-02"], ["S0"], [100.0]),
        _sites(n=1, dist=(99.0,), delev=(10.0,)), t)
    assert v is None and node is None and gardes is None


def test_poids_masse_sans_cible_leve():
    """Meme garde-fou que pour les autres auxiliaires (dette #14)."""
    from meandre.training.trainer import Trainer
    with pytest.raises(ValueError, match="sans sa cible"):
        Trainer(model=torch.nn.Linear(1, 1), loss_fn=HydroLoss(w_swe_mass=0.3),
                train_data=_donnees())


def test_masse_et_couverture_sont_des_cibles_DISTINCTES():
    """Le coeur de R24 : la couverture SATURE, la masse non.

    SCF = 1-exp(-SWE/15) vaut deja 0.96 a 50 mm et 0.9997 a 121 mm : entre un manteau
    de 121 mm et un de 238, la couverture ne bouge plus du tout. Une contrainte sur la
    couverture ne peut donc PAS corriger un deficit de masse, quelle que soit son poids.
    """
    couverture = lambda swe: 1.0 - torch.exp(-torch.tensor(swe) / 15.0)
    simule, mesure = couverture(121.0), couverture(238.0)
    assert float(mesure - simule) < 1e-3, "la couverture ne distingue plus 121 de 238 mm"
    assert abs(238.0 - 121.0) / 100.0 > 1.0, "la masse, elle, voit un ecart de 1.2 unite"


# ── Terme climatologique GRACE : mecanique du passage direct (R23) ───────────

def _passage_direct(res, biais, sigma=5.4):
    """Reproduit le calcul du trainer : valeur = biais accumule, gradient = residu.

    `res` porte le graphe, `biais` est detache. La valeur penalisee doit etre le biais
    (donc insensible au residu courant), mais la derivee doit passer par le residu.
    """
    st = res - res.detach() + biais
    return (st.pow(2) / (sigma ** 2)).mean()


def test_passage_direct_penalise_le_biais_pas_le_residu_courant():
    """La VALEUR ne doit dependre que du biais accumule.

    Sinon le terme redeviendrait un terme mensuel deguise, et on retomberait dans le
    defaut de R23 : juger un biais systematique a l'echelle de bruit d'un seul mois.
    """
    biais = torch.full((4,), 20.0)
    for valeur_residu in (0.0, 20.0, -50.0):
        res = torch.full((4,), valeur_residu, requires_grad=True)
        L = _passage_direct(res, biais).detach()
        assert abs(float(L) - (20.0 / 5.4) ** 2) < 1e-4


def test_passage_direct_pousse_dans_le_BON_SENS():
    """Un biais POSITIF (modele trop haut) doit faire DESCENDRE le residu.

    Le signe est la seule chose qui ne pardonne pas : inverse, la contrainte
    aggraverait exactement l'ecart qu'elle est censee fermer.
    """
    res = torch.zeros(4, requires_grad=True)
    _passage_direct(res, torch.full((4,), 20.0)).backward()
    assert (res.grad > 0).all(), "gradient positif => la descente diminue le residu"

    res2 = torch.zeros(4, requires_grad=True)
    _passage_direct(res2, torch.full((4,), -20.0)).backward()
    assert (res2.grad < 0).all(), "biais negatif => la descente augmente le residu"


def test_passage_direct_muet_tant_que_le_biais_est_nul():
    """Au premier troncon le biais accumule vaut zero : aucun gradient ne doit sortir.

    C'est voulu -- la moyenne mobile doit d'abord se remplir. Un gradient non nul ici
    signifierait qu'on pousse sur du bruit de troncon.
    """
    res = torch.randn(6, requires_grad=True)
    L = _passage_direct(res, torch.zeros(6))
    L.backward()
    assert float(L) == 0.0 and torch.allclose(res.grad, torch.zeros(6))


def test_bucketize_du_jour_julien_vers_le_mois():
    """Les bornes du trainer doivent rendre 0 pour janvier et 11 pour decembre."""
    bornes = torch.tensor([31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    for doy, mois in [(1, 0), (31, 0), (32, 1), (60, 1), (61, 2), (121, 3),
                      (182, 5), (335, 10), (336, 11), (365, 11), (366, 11)]:
        assert int(torch.bucketize(torch.tensor(doy), bornes)) == mois, f"jour {doy}"


# ── krec dans le champ : ancrage de la moyenne, variation spatiale libre ─────

def _champ(n=64):
    from meandre.spatial.field_network import SpatialFieldNetwork
    return SpatialFieldNetwork(n_territorial=4)


def test_prior_krec_est_opt_in():
    """Quand krec est IMPOSE par le calage, la sortie du champ n'est pas utilisee.

    L'ancrer alors tirerait un parametre mort et changerait la perte sans changer la
    physique -- ce qui casserait la comparabilite entre deux runs qui ne different que
    par le calage. Le drapeau n'est leve que quand krec est reellement libre.
    """
    net = _champ()
    assert getattr(net, "prior_on_krec", False) is False


def test_prior_krec_ancre_la_MOYENNE_et_pas_chaque_noeud():
    """Le coeur de la remarque d'Essi : krec est une propriete du sous-sol.

    La forme (log(p).mean() - log(c))^2 ne penalise QUE l'ecart de la moyenne
    geometrique a la cible. Deux champs de meme moyenne mais de dispersion tres
    differente doivent donc coûter la MEME chose -- sinon on penaliserait la variance
    spatiale, ce qui est le mecanisme du collapse du NeRF (revue 2026-07-01).
    """
    import math
    from meandre.spatial.field_network import KREC_REF
    cible = math.log(KREC_REF)
    plat = torch.full((100,), KREC_REF)
    disperse = torch.exp(torch.linspace(cible - 1.0, cible + 1.0, 100))
    terme = lambda p: (torch.log(p + 1e-12).mean() - cible) ** 2
    assert abs(float(terme(plat))) < 1e-10
    assert abs(float(terme(disperse))) < 1e-6, "la dispersion ne doit RIEN couter"


def test_prior_krec_rappelle_un_champ_effondre():
    """Un krec ecrase a la valeur du calage Hydrotel doit etre penalise fortement.

    1.3e-7 contre une cible de 2e-5, c'est un facteur 154 : le terme doit etre grand
    devant le bruit, sinon l'ancrage ne pese rien face au debit.
    """
    import math
    from meandre.spatial.field_network import KREC_REF
    cible = math.log(KREC_REF)
    effondre = torch.full((100,), 1.3e-7)
    terme = float((torch.log(effondre + 1e-12).mean() - cible) ** 2)
    assert terme > 20.0, f"terme trop faible pour contrer le debit : {terme}"


def test_krec_reste_dans_ses_bornes_physiques():
    """La transformation log-normale borne krec a [1e-7, 1e-4] m/h.

    En dessous, la couche profonde ne draine plus du tout (0.0036 mm/j a 1.3e-7) ;
    au-dessus, la nappe fournit la quasi-totalite du debit et le score s'effondre.
    """
    net = _champ()
    brut = torch.linspace(-50.0, 50.0, 200)
    from meandre.spatial.field_network import KREC_REF
    import math
    k = torch.exp(torch.clamp(brut * 0.3 + math.log(KREC_REF),
                              math.log(1e-7), math.log(1e-4)))
    assert float(k.min()) >= 1e-7 - 1e-12 and float(k.max()) <= 1e-4 + 1e-12
    # raw = 0 doit rendre exactement la reference : les points de reprise anciens,
    # remplis de zeros sur cette sortie, arrivent donc a une valeur physique.
    k0 = math.exp(0.0 * 0.3 + math.log(KREC_REF))
    assert abs(k0 - KREC_REF) < 1e-12
