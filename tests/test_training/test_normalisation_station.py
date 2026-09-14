"""La normalisation par station rend l'ecart quadratique sans dimension.

Sans elle, le terme se calcule en metres cubes par seconde au carre : sur un bassin de
quelques dizaines de metres cubes par seconde il vaut une centaine, quand tous les autres
termes de la perte valent moins de deux. Le poids affiche dans la configuration ne dit
alors plus rien du poids reel, et la station la plus grosse emporte la perte a elle seule.
Le pilote regional divise chaque station par la variance observee de ses debits ; ce test
verifie que la division a bien lieu et qu'elle egalise les stations.
"""
import torch

from meandre.training.loss import HydroLoss


def _perte(station_var, echelle):
    """Une station, dix ans de debits, l'erreur valant un dixieme du signal."""
    torch.manual_seed(0)
    n = 3650
    q_obs = (torch.rand(n, 1) * 2.0 + 1.0) * echelle
    q_sim = q_obs * 1.1
    kw = dict(w_kge=0.0, w_pbias=0.0, w_mse=1.0, w_nse=0.0, w_nrmse=0.0,
              w_log_nse=0.0, w_log_mse=0.0, per_station=True)
    if station_var is not None:
        kw["station_var"] = station_var
    f = HydroLoss(**kw)
    out = f(q_obs=q_obs, q_sim=q_sim, station_mask=torch.ones(q_obs.shape[1], dtype=torch.bool))
    return float(out[0] if isinstance(out, tuple) else out)


def test_sans_normalisation_le_terme_croit_avec_le_carre_du_debit():
    """Deux bassins identiques a l'echelle pres donnent des pertes dans un rapport cent,
    ce qui est la source du desequilibre."""
    petit = _perte(None, 1.0)
    grand = _perte(None, 10.0)
    assert grand / petit > 50


def test_avec_normalisation_l_echelle_disparait():
    """Normalisee par la variance observee, la meme erreur relative donne la meme perte,
    que le bassin fasse un metre cube par seconde ou cent."""
    torch.manual_seed(0)
    n = 3650
    for ech in (1.0, 10.0, 100.0):
        q_obs = (torch.rand(n, 1) * 2.0 + 1.0) * ech
        v = torch.tensor([float(q_obs.var())])
        f = HydroLoss(w_kge=0.0, w_pbias=0.0, w_mse=1.0, w_nse=0.0, w_nrmse=0.0,
                      w_log_nse=0.0, w_log_mse=0.0, per_station=True, station_var=v)
        out = f(q_obs=q_obs, q_sim=q_obs * 1.1,
                station_mask=torch.ones(q_obs.shape[1], dtype=torch.bool))
        val = float(out[0] if isinstance(out, tuple) else out)
        if ech == 1.0:
            ref = val
        assert abs(val - ref) / ref < 0.05, f"echelle {ech} : {val:.4f} contre {ref:.4f}"


def test_le_terme_normalise_est_d_ordre_un():
    """Une erreur de dix pour cent doit donner une perte lisible a cote des autres termes,
    pas une centaine."""
    torch.manual_seed(0)
    q_obs = (torch.rand(3650, 1) * 2.0 + 1.0) * 40.0
    v = torch.tensor([float(q_obs.var())])
    f = HydroLoss(w_kge=0.0, w_pbias=0.0, w_mse=1.0, w_nse=0.0, w_nrmse=0.0,
                  w_log_nse=0.0, w_log_mse=0.0, per_station=True, station_var=v)
    out = f(q_obs=q_obs, q_sim=q_obs * 1.1,
            station_mask=torch.ones(q_obs.shape[1], dtype=torch.bool))
    val = float(out[0] if isinstance(out, tuple) else out)
    assert 0.001 < val < 10.0, val


def test_le_banc_de_sousbassin_normalise():
    """Garde-fou : le banc doit passer station_var a sa perte. Il ne le faisait pas
    jusqu'au 2026-09-13 et minimisait donc autre chose que la region qu'il represente."""
    src = open(".runs/quebec/banc_sousbassin.py", encoding="utf-8").read()
    debut = src.index("loss_fn = HydroLoss(")
    fin = src.index("loss_fn = HydroLoss(", debut + 10)
    assert "station_var=" in src[debut:fin + 600], "le banc construit une perte sans station_var"
    assert src.count("station_var=_svar") == 2, "les deux branches doivent normaliser"


def test_le_banc_expose_les_deux_termes_de_la_recette():
    """Le banc doit pouvoir reproduire la perte du pilote regional, faute de quoi il ne
    peut pas reproduire ses defauts. Deux termes y manquaient jusqu'au 2026-09-13 :
    l'ecart quadratique sur le logarithme des debits, que la recette pose a 0,3 et qui
    vise les basses eaux, et le terme de pics, pose a 0,5."""
    src = open(".runs/quebec/banc_sousbassin.py", encoding="utf-8").read()
    assert '"--w-log-mse"' in src
    assert '"--w-peak"' in src
    assert "w_log_mse=float(w_log_mse)" in src
    assert "peak_threshold=_pthr" in src
    assert "torch.quantile(_qtr[_mk, _i], 0.75)" in src, "le seuil de pics doit etre le Q75"
