"""Reservoir souterrain non lineaire Q = q_ref * (S/S_REF)^b (R29, opt-in).

Ce que le lineaire ne sait pas faire et que la loi puissance doit faire : porter DEUX
constantes de temps avec un seul jeu de parametres. Les recessions mesurees d'OUTV
donnent 37 jours en mediane et 111 jours pour la composante lente ; un reservoir
lineaire n'a qu'un seul temps de residence, par construction.
"""
import torch
import pytest

from meandre.vertical.aquifer import AquiferModule, PowerLawAquifer


def _recession(module, S0=200.0, jours=120, **params):
    """Vidange libre depuis S0, sans recharge. Retourne la serie de Q (mm/j)."""
    S = torch.tensor([S0])
    zero = torch.zeros(1)
    qs = []
    for _ in range(jours):
        Q, S = module(zero, S, **params)
        qs.append(float(Q))
    return torch.tensor(qs)


def test_b_egal_1_reproduit_le_lineaire():
    """A b=1 la loi puissance EST un lineaire de k = q_ref/S_REF : la nouveaute doit
    contenir l'ancien comme cas particulier, sinon les deux ne sont pas comparables."""
    k = 0.03
    q_lin = _recession(AquiferModule(), k_gw=torch.tensor([k]))
    q_pow = _recession(PowerLawAquifer(n_substeps=8),
                       q_ref=torch.tensor([k * PowerLawAquifer.S_REF]),
                       b=torch.tensor([1.0]))
    # discretisations differentes (analytique contre sous-pas) : 2 % de tolerance
    assert torch.allclose(q_lin[:60], q_pow[:60], rtol=0.02)


def test_deux_constantes_de_temps():
    """LE motif d'existence : la recession se raidit puis s'etire.

    On mesure k_apparent = -dln(Q)/dt au debut (stock plein) et a la fin (stock bas).
    Pour b > 1 le rapport debut/fin doit etre franc (les recessions d'OUTV donnent
    0.0273 contre 0.0090, rapport 3) ; pour b = 1 il vaut exactement 1.
    """
    q = _recession(PowerLawAquifer(), S0=300.0, jours=150,
                   q_ref=torch.tensor([3.0]), b=torch.tensor([2.0]))
    lnq = torch.log(q.clamp(min=1e-12))
    k_debut = float(lnq[2] - lnq[7]) / 5.0
    k_fin = float(lnq[100] - lnq[140]) / 40.0
    assert k_debut > 2.5 * k_fin, f"raide {k_debut:.4f} vs etire {k_fin:.4f}"
    q1 = _recession(PowerLawAquifer(), S0=300.0, jours=150,
                    q_ref=torch.tensor([3.0]), b=torch.tensor([1.0]))
    lnq1 = torch.log(q1.clamp(min=1e-12))
    k1_debut = float(lnq1[2] - lnq1[7]) / 5.0
    k1_fin = float(lnq1[100] - lnq1[140]) / 40.0
    assert abs(k1_debut - k1_fin) / k1_debut < 0.05, "lineaire : UNE constante"


def test_conservation_stricte():
    """Sur chaque pas : S_new = S + recharge - Q. C'est la fermeture du bilan ; si elle
    casse, ETL_BILAN lira une fuite dans la nappe."""
    m = PowerLawAquifer()
    S = torch.tensor([150.0])
    for r in (5.0, 0.0, 12.0, 3.0):
        rech = torch.tensor([r])
        Q, S2 = m(rech, S, q_ref=torch.tensor([2.0]), b=torch.tensor([1.7]))
        assert abs(float(S + rech - Q - S2)) < 1e-4
        S = S2


def test_stock_nul_ne_produit_rien_et_ne_devient_pas_negatif():
    m = PowerLawAquifer()
    Q, S = m(torch.zeros(1), torch.zeros(1),
             q_ref=torch.tensor([4.0]), b=torch.tensor([2.0]))
    assert float(Q) == 0.0 and float(S) == 0.0


def test_pompage_assecha_sans_negatif():
    """Prelevement souterrain superieur au stock : la nappe se vide, point."""
    m = PowerLawAquifer()
    Q, S = m(torch.zeros(1), torch.tensor([2.0]),
             q_ref=torch.tensor([1.0]), b=torch.tensor([1.5]),
             gw_withdrawal=torch.tensor([-50.0]))
    assert float(S) >= 0.0 and float(Q) >= 0.0


def test_differentiable_vers_les_deux_parametres():
    """b et q_ref doivent recevoir du gradient : c'est le NeRF qui les apprendra."""
    q_ref = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([1.8], requires_grad=True)
    m = PowerLawAquifer()
    S = torch.tensor([120.0])
    tot = torch.zeros(1)
    for _ in range(10):
        Q, S = m(torch.tensor([4.0]), S, q_ref=q_ref, b=b)
        tot = tot + Q
    tot.sum().backward()
    assert q_ref.grad is not None and float(q_ref.grad.abs()) > 0
    assert b.grad is not None and float(b.grad.abs()) > 0


def test_residences_mesurees_atteignables():
    """Les deux cibles d'OUTV (37 j a stock plein, 111 j a stock bas) doivent etre
    DANS l'enveloppe d'un meme jeu (q_ref, b) plausible -- sinon le module ne peut pas
    representer le bassin qu'on lui demande de representer."""
    q = _recession(PowerLawAquifer(), S0=250.0, jours=200,
                   q_ref=torch.tensor([2.6]), b=torch.tensor([2.2]))
    lnq = torch.log(q.clamp(min=1e-12))
    k_plein = float(lnq[1] - lnq[6]) / 5.0
    k_bas = float(lnq[150] - lnq[190]) / 40.0
    assert 1.0 / k_plein < 60.0, f"residence a stock plein {1/k_plein:.0f} j, cible ~37"
    assert 1.0 / k_bas > 80.0, f"residence a stock bas {1/k_bas:.0f} j, cible ~111"


# ── Drainage non lineaire de L3 dans bv3c2 (R37) : la vanne, pas la nappe ────

def _sol_un_pas(l3_exp=None, t3=0.40, ths3=0.42, krec=2e-5):
    """Un pas du clone sol, isole sur q3 : verifie la formule sans tout le sous-pas."""
    z3 = torch.tensor([2.65])
    t3_t, ths3_t, krec_t = torch.tensor([t3]), torch.tensor([ths3]), torch.tensor([krec])
    if l3_exp is None:
        return krec_t * z3 * t3_t
    return krec_t * z3 * ths3_t * torch.clamp(t3_t / ths3_t, min=0.0) ** l3_exp


def test_exposant_1_egale_le_lineaire_fidele():
    assert torch.allclose(_sol_un_pas(l3_exp=1.0), _sol_un_pas(l3_exp=None))


def test_meme_plafond_a_saturation():
    """A theta3 = thetas3, la variante draine EXACTEMENT comme le lineaire : la
    non-linearite ne change pas la capacite maximale, elle change QUAND on l'a."""
    for n in (2.0, 8.0, 16.0):
        assert torch.allclose(_sol_un_pas(l3_exp=n, t3=0.42), _sol_un_pas(t3=0.42))


def test_coupure_sous_saturation():
    """A 90 % de saturation, n=8 doit couper le drainage a moins de la moitie du
    lineaire : c'est la respiration -- vider vite quand c'est plein, retenir sinon."""
    lin = float(_sol_un_pas(t3=0.378))
    n8 = float(_sol_un_pas(l3_exp=8.0, t3=0.378))
    assert n8 < 0.5 * lin
    assert n8 > 0.0
