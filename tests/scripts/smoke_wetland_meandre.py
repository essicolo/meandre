"""Smoke : la HydrotelColumn de meandre active le milieu humide isolé par nœud
depuis le territorial, masque les nœuds sans MH (no-op exact), et reste
différentiable. Vérifie aussi que la géométrie reconstruite égale le standalone
validé (hydrotel_clone, UHRH376 DELISLE : wet_a=0.2808 km2, uhrh_a=0.4264).

  python tests/smoke_wetland_meandre.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hydrotel_clone.milieu_humide import init_wetland_geom
from meandre.vertical.hydrotel_column import HydrotelColumn

torch.set_default_dtype(torch.float64)


class FakeTerritorial:
    """Stub minimal : seul get_physical est utilisé par _wetland_from_territorial."""
    def __init__(self, d): self._d = d
    def get_physical(self, k): return self._d.get(k)


def test_geom_matches_standalone():
    # 2 nœuds : nœud0 = UHRH376 (MH 66%), nœud1 = sans MH
    phys = dict(
        wet_a_raw=torch.tensor([0.2808, 0.0]),
        area_km2_local=torch.tensor([0.4264, 1.2]),
        wet_dra_fr_raw=torch.tensor([1.0, 0.0]),
        frac_raw=torch.tensor([0.8, 0.8]),
        wetdnor_raw=torch.tensor([0.2, 0.2]),
        wetdmax_raw=torch.tensor([0.3, 0.3]),
        ksat_bs_raw=torch.tensor([0.5, 0.5]),
        c_ev_raw=torch.tensor([0.6, 0.6]),
        c_prod_raw=torch.tensor([10.0, 10.0]))
    col = HydrotelColumn(use_frost=False)
    like = torch.zeros(2)
    w = col._wetland_from_territorial(FakeTerritorial(phys), like)
    assert w is not None
    # standalone DELISLE UHRH376
    A0, B0, nvol0, mxvol0 = init_wetland_geom(0.2808, 0.3, 0.8, 0.2)
    assert abs(float(w["A"][0]) - A0) < 1e-9, (float(w["A"][0]), A0)
    assert abs(float(w["B"][0]) - B0) < 1e-3, (float(w["B"][0]), B0)
    assert abs(float(w["wetnvol"][0]) - nvol0) < 1e-3
    assert abs(float(w["wetmxvol"][0]) - mxvol0) < 1e-3
    assert abs(float(w["wet_fr_area"][0]) - 0.2808 / 0.4264) < 1e-9
    assert abs(float(w["hru_ha"][0]) - 42.64) < 1e-6
    assert bool(w["wmask"][0]) and not bool(w["wmask"][1])
    # nœud sans MH : wet_dra_fr forcé 0 (no-op), géométrie factice finie (pas de NaN)
    assert float(w["wet_dra_fr"][1]) == 0.0
    assert torch.isfinite(w["A"]).all() and torch.isfinite(w["B"]).all()
    print(f"geom OK : A={float(w['A'][0]):.5f} B={float(w['B'][0]):.2f} "
          f"wetnvol={float(w['wetnvol'][0]):.1f} wet_fr_area={float(w['wet_fr_area'][0]):.4f}")
    return col, w, phys


def test_forward_mask_and_grad(col, w):
    """forward : nœud sans MH = no-op exact vs colonne wetland=None ; gradient fini."""
    # params sol/snow/etr minimaux (2 nœuds), comme smoke_hydrotel_column
    from hydrotel_clone.bv3c2 import make_params
    from hydrotel_clone.snow import init_ce
    T = lambda x: torch.full((2,), float(x))
    pso = make_params("sandy_loam", "sandy_loam", "sandy_loam", slope=0.03,
                      fsa=0.9, fse=0.0, fsi=0.1, krec=1e-5, cin=0.3)
    pso = {k: (v * torch.ones(2) if v.dim() == 0 else v) for k, v in pso.items()}
    ce1, ce0 = init_ce(T(46.0), T(0.03), T(7))
    psnow = dict(lat=T(46.0), ce1=ce1, ce0=ce0, pct_conifers=T(0.0), pct_feuillus=T(0.3),
                 pct_autres=T(0.7), coeff_fonte_conifers=T(.012), coeff_fonte_feuillus=T(.014),
                 coeff_fonte_decouver=T(.016), seuil_fonte_conifers=T(0.0), seuil_fonte_feuillus=T(0.0),
                 seuil_fonte_decouver=T(0.0), taux_fonte_geo=T(0.5), densite_max=T(466.0),
                 constante_tassement=T(0.1))
    petr = dict(thetacc=T(0.207), thetapf=T(0.095), alpha=T(4.5), des=T(0.6), coef_assech=T(1.0),
                z11=T(0.15), z22=T(0.4), z33=T(1.0),
                classes=[(T(0.3), [1, 365], [5, 5], [1.5, 1.5])])

    def run(wet):
        c = HydrotelColumn(et_mode="mcguinness", use_frost=False, soil_n_substep=200)
        c.set_static(psnow, pso, petr, wetland=wet, n_depth=1)
        st = c.init_state(2, theta_init=(0.36, 0.36, 0.36))
        st.wet_vol = torch.tensor([500.0, 0.0])
        out = []
        for i in range(10):
            P = T(30.0 if i == 6 else 1.0)
            prod, st, _ = c(P, T(5.0), T(15.0), T(15.0), T(2.0), T(1.0), float(150 + i), st)
            out.append(prod)
        return torch.stack(out), c

    prod_w, _ = run(w)
    prod_none, _ = run(None)
    # nœud1 (sans MH, wmask False) : prod identique à wetland=None
    d1 = (prod_w[:, 1] - prod_none[:, 1]).abs().max()
    assert float(d1) < 1e-9, f"nœud sans MH devrait être no-op exact, écart={float(d1):.2e}"
    # nœud0 (MH 66%) : le milieu humide DOIT réduire la prod sur l'orage
    red = float(prod_none[6, 0] - prod_w[6, 0])
    assert red > 0, f"le MH devrait réduire la prod sur l'orage, delta={red:.3f}"
    print(f"mask OK : nœud sans MH no-op (écart {float(d1):.1e}) ; nœud MH réduit l'orage de {red:.2f}mm")

    # différentiabilité du CHEMIN milieu humide : ksat_bs (param MH) requiert grad
    ks_bs = torch.tensor([0.5, 0.5], requires_grad=True)
    col2 = HydrotelColumn(et_mode="mcguinness", use_frost=False, soil_n_substep=200)
    wg = dict(w); wg["wet_k"] = ks_bs   # injecte le tenseur grad dans le dict MH
    col2.set_static(psnow, pso, petr, wetland=wg, n_depth=1)
    st = col2.init_state(2, theta_init=(0.36, 0.36, 0.36)); st.wet_vol = torch.tensor([500.0, 0.0])
    for i in range(6):
        prod, st, _ = col2(T(30.0 if i == 3 else 1.0), T(5.0), T(15.0), T(15.0), T(2.0), T(1.0), float(150 + i), st)
    g = torch.autograd.grad(prod.sum(), ks_bs)[0]
    assert torch.isfinite(g).all(), f"gradient MH NaN: {g}"
    assert abs(float(g[0])) > 0, f"grad nul sur nœud MH: {g}"
    assert float(g[1]) == 0.0, f"nœud sans MH devrait avoir grad nul (masqué): {g}"
    print(f"grad OK chemin MH : d(prod)/d(ksat_bs)={[round(float(x),4) for x in g]} (nœud0 MH non nul, nœud1 masqué nul)")


if __name__ == "__main__":
    col, w, phys = test_geom_matches_standalone()
    test_forward_mask_and_grad(col, w)
    print("SMOKE WETLAND MEANDRE OK")
