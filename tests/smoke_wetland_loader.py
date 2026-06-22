"""Smoke : la plomberie loader agrège les milieux humides isolés UHRH→troncon
et émet les colonnes territoriales *_raw, qui activent le milieu humide dans
HydrotelColumn. Validé sur les VRAIS troncons SLSO (.runs/slso/physitel) + les
VRAIES données milieu humide de la plateforme SLSO (forcées actives ici : le run
Hydrotel SLSO les désactive, mais on teste l'agrégation sur données réelles).

Vérifie : conservation (somme troncon = somme UHRH pour wet_a et wet_vol),
pondération wet_dra_fr, et flux jusqu'à _wetland_from_territorial.

  python tests/smoke_wetland_loader.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from meandre.data.physitel_loader import (
    _parse_uhrh, _parse_occupation_sol_cla, _parse_type_sol_cla,
    _parse_troncon, _build_graph, _build_territorial)

PHYSI = Path(".runs/slso/physitel")
PLAT = Path("../plateformes-hydrotel/LN24HA/SLSO_LN24HA_2020")
MH_CSV = PLAT / "simulation/simulation/milieux_humides_isoles.csv"


def read_wetlands(path):
    """milieux_humides_isoles.csv → dict uhrh→params (flag ignoré : on force)."""
    out = {}
    for ln in path.read_text(encoding="latin-1").splitlines():
        c = ln.split(";")
        if len(c) >= 10 and c[0].strip().isdigit():
            uid = int(c[0]); v = [float(x) for x in c[1:10]]
            out[uid] = dict(uhrh_a=v[0], wet_a=v[1], wet_dra_fr=v[2], frac=v[3],
                            wetdnor=v[4], wetdmax=v[5], ksat_bs=v[6], c_ev=v[7],
                            c_prod=v[8], wet_vol_init=float(uid % 7))  # vol synthétique non nul
    return out


def main():
    assert PHYSI.exists(), f"physitel SLSO introuvable: {PHYSI}"
    assert MH_CSV.exists(), f"CSV milieu humide SLSO introuvable: {MH_CSV}"
    uhrh = _parse_uhrh(PHYSI / "uhrh.csv")
    lc = _parse_occupation_sol_cla(PHYSI / "occupation_sol.cla", uhrh)
    soil = _parse_type_sol_cla(PHYSI / "type_sol.cla", uhrh)
    troncons = _parse_troncon(PHYSI / "troncon.trl")
    graph, node_ids, tidx = _build_graph(troncons, velocity_m_s=1.0, device=None)
    wet = read_wetlands(MH_CSV)
    print(f"{len(node_ids)} troncons, {len(uhrh)} UHRH, {len(wet)} UHRH avec MH")

    terr, _ = _build_territorial(troncons, tidx, node_ids, uhrh, lc, soil, graph,
                                 normalise=True, device=None, wetlands=wet)
    gp = terr.get_physical
    for k in ("wet_a_raw", "wet_dra_fr_raw", "wet_vol_init_raw", "wetdmax_raw"):
        assert gp(k) is not None, f"colonne {k} absente"

    # quels UHRH sont réellement rattachés à un troncon du réseau ?
    attached = set()
    for t in troncons:
        if tidx.get(t["id"]) is not None:
            attached.update(t["uhrh_ids"])
    wa_uhrh = sum(d["wet_a"] for u, d in wet.items() if u in attached)
    wv_uhrh = sum(d["wet_vol_init"] for u, d in wet.items() if u in attached)
    wa_tron = float(gp("wet_a_raw").sum())
    wv_tron = float(gp("wet_vol_init_raw").sum())
    print(f"CONSERVATION wet_a  : troncon {wa_tron:.4f} vs UHRH {wa_uhrh:.4f} km2")
    print(f"CONSERVATION wetvol : troncon {wv_tron:.2f} vs UHRH {wv_uhrh:.2f} m3")
    assert abs(wa_tron - wa_uhrh) < 1e-4, (wa_tron, wa_uhrh)
    assert abs(wv_tron - wv_uhrh) < 1e-2, (wv_tron, wv_uhrh)

    # wet_dra_fr ∈ [0,1] et > 0 là où il y a du MH
    wd = gp("wet_dra_fr_raw")
    assert float(wd.min()) >= 0.0 and float(wd.max()) <= 1.0
    nnz = int((gp("wet_a_raw") > 0).sum())
    print(f"troncons avec MH: {nnz}/{len(node_ids)} ; wet_dra_fr in [{float(wd.min()):.3f}, {float(wd.max()):.3f}]")

    # flux jusqu'à HydrotelColumn (active le milieu humide)
    from meandre.vertical.hydrotel_column import HydrotelColumn
    col = HydrotelColumn(use_frost=False)
    w = col._wetland_from_territorial(terr, torch.zeros(len(node_ids)))
    assert w is not None and torch.isfinite(w["A"]).all() and torch.isfinite(w["B"]).all()
    assert int(w["wmask"].sum()) == nnz, (int(w["wmask"].sum()), nnz)
    print(f"_wetland_from_territorial OK : {int(w['wmask'].sum())} nœuds MH, "
          f"wet_fr_area max {float(w['wet_fr_area'].max()):.3f}, A finis")
    print("SMOKE WETLAND LOADER OK")


if __name__ == "__main__":
    main()
