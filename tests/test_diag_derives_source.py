"""Les diagnostics retenus se décident sur l'objet de perte, pas sur la configuration.

Certains poids sont posés sur l'objet de perte après sa construction, et `w_nappe` était dans
ce cas : lu dans la configuration il valait zéro, le diagnostic du souterrain n'était pas
retenu, et le terme des puits tournait sans gradient tout en s'affichant comme actif. La
garde du pilote d'entraînement l'a signalé ; la source de vérité doit être la même des deux
côtés.
"""
import pytest


def besoins(loss_fn, lcfg):
    """Reproduit la règle du pilote."""

    def poids(nom):
        v = getattr(loss_fn, nom, None)
        return float(v if v is not None else lcfg.get(nom, 0.0) or 0.0)

    d = {"etr": poids("w_et") or poids("w_nll_et"),
         "swe": poids("w_snow") or poids("w_swe_mass"),
         "s_gw": poids("w_nappe") or poids("w_tws"),
         "profondeur_nappe_m": poids("w_nappe"),
         "theta1": poids("w_tws")}
    return sorted(k for k, v in d.items() if v > 0.0)


class _Perte:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_un_poids_pose_apres_coup_est_vu():
    """Le cas exact qui a echoue : w_nappe absent de la configuration."""
    assert "s_gw" in besoins(_Perte(w_nappe=1.0), {})
    assert "profondeur_nappe_m" in besoins(_Perte(w_nappe=1.0), {})


def test_un_poids_de_la_configuration_seule_est_vu():
    assert "etr" in besoins(_Perte(), {"w_et": 0.4})


def test_lobjet_de_perte_prime_sur_la_configuration():
    """Le pilote peut éteindre un terme que la configuration allumait."""
    assert besoins(_Perte(w_et=0.0), {"w_et": 0.4}) == [], (
        "un repli sur la configuration rallumerait le terme en douce")


def test_rien_ne_sort_quand_tout_est_eteint():
    assert besoins(_Perte(), {}) == []


def test_la_gravimetrie_entraine_les_teneurs_en_eau():
    g = besoins(_Perte(w_tws=0.2), {})
    assert "theta1" in g and "s_gw" in g
