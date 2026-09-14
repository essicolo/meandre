"""L'exces d'infiltration sous-journalier : sans lui, aucune pointe d'ete n'est possible.

Mesure du 2026-09-14 : sous une averse de 40 mm sur un sol a 70 % d'humidite relative, la
colonne au pas journalier rend 0,4 % d'ecoulement rapide le jour meme, contre 5 a 30 %
attendus en foret boreale. Le pic simule vaut en consequence 0,48 a 0,55 de l'observe hors
crue printaniere, alors que la pluie recue n'a pas de biais mesurable.

Ces tests fixent le comportement du mecanisme correctif, qui doit rester OPT-IN.
"""
import pytest
import torch

from hydrotel_clone.bv3c2 import BV3C2Clone, SOIL_TEXTURES, EPAISSEUR, KREC_DEFAULT, CIN_DEFAULT, make_params


@pytest.fixture(autouse=True)
def _double():
    ancien = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(ancien)


def _colonne():
    p = make_params("silt_loam", "silt_loam", "silt_loam", slope=0.05, fsa=1.0, fse=0.0,
                    fsi=0.0, krec=KREC_DEFAULT, cin=CIN_DEFAULT, coef_recharge=0.0)
    z1, z2, z3 = EPAISSEUR
    for k, v in (("z1", z1), ("z2", z2), ("z3", z3)):
        p[k] = torch.full((1,), float(v))
    p = {k: (v if torch.is_tensor(v) and v.numel() == 1
             else (torch.full((1,), float(v)) if not torch.is_tensor(v) else v[:1].clone()))
         for k, v in p.items()}
    return BV3C2Clone(), p


def _rapide(pluie_mm, orage, theta_rel=0.7):
    cl, p = _colonne()
    tex = SOIL_TEXTURES["silt_loam"]
    t = tex["thetapf"] + theta_rel * (tex["thetas"] - tex["thetapf"])
    th = [torch.full((1,), float(t)) for _ in range(3)]
    sh = None if orage is None else torch.full((1,), float(orage))
    out = cl.forward(th[0], th[1], th[2], torch.full((1,), float(pluie_mm)),
                     torch.zeros(1), torch.zeros(1), torch.zeros(1), p, storm_hours=sh)
    return 100.0 * (float(out[0].item()) + float(out[1].item())) / pluie_mm


def test_sans_duree_d_orage_la_colonne_ne_produit_presque_rien():
    """Le clone fidele, au pas journalier, est sous un pour cent : c'est le defaut."""
    for pluie in (20.0, 40.0, 80.0):
        assert _rapide(pluie, None) < 1.0, pluie


def test_une_averse_courte_produit_un_ecoulement_rapide():
    """Deux heures d'orage font passer la colonne dans la plage physique et au-dela."""
    assert _rapide(20.0, 2.0) > 10.0
    assert _rapide(40.0, 2.0) > 10.0


def test_l_effet_croit_quand_l_averse_se_concentre():
    """Plus l'averse est courte, plus elle ruisselle : la monotonie est la propriete qui
    justifie de tirer la duree de la donnee horaire plutot que d'une constante."""
    vals = [_rapide(40.0, h) for h in (24.0, 12.0, 6.0, 3.0, 2.0)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] > vals[0] + 10.0


def test_une_pluie_etalee_sur_la_journee_restitue_le_clone():
    """Vingt-quatre heures d'orage doivent rendre le comportement du pas journalier, sans
    quoi le mecanisme modifierait les pluies frontales qu'il ne vise pas."""
    for pluie in (20.0, 40.0):
        assert abs(_rapide(pluie, 24.0) - _rapide(pluie, None)) < 1.0, pluie


def test_le_constructeur_de_forcage_expose_le_canal():
    """Le mecanisme reste inerte sans canal de duree d'orage dans le forcage : la ligne
    quebecoise n'en avait pas, et c'est pourquoi il n'avait jamais servi."""
    src = open(".runs/quebec/build_forcing_region.py", encoding="utf-8").read()
    assert "QC_INTENS" in src
    assert "def load_dt_eff" in src
    assert '"DT_eff"' in src
