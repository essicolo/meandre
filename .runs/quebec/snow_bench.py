"""Banc NEIGE SEULE aux noeuds des sites CanSWE : des dizaines de variantes en minutes.

Motif (Essi, 2026-08-24 : « on peut tester la modulation sans passer par des runs
complets non ? »). Oui : le juge nival ne regarde que les noeuds des sites CanSWE
(~50-110 par region), et la neige est un module autonome -- ni sol, ni routage, ni GPU.
Ce banc rejoue DegreJourModifie (ou l'ETI) sur les 25 ans de forcage a ces seuls
noeuds, avec les MEMES ancrages de plateforme et le MEME appariement site-jour que le
pilote. Une variante prend quelques secondes ; l'inference complete en prenait 6 a 9
minutes et l'entrainement 90.

Ce que le banc NE JUGE PAS : le debit (pas de sol ni de routage) et la retroaction de
l'apprentissage. Il sert a DEGROSSIR -- eliminer les variantes qui cassent le manteau
avant de payer un run complet pour les survivantes. Le controle nu doit reproduire les
ratios du pilote (dette #10 : un banc sans ligne de controle mesure du vide).

    .venv/Scripts/python.exe .runs/quebec/snow_bench.py outv
    .venv/Scripts/python.exe .runs/quebec/snow_bench.py gasp --eti
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import xarray as xr

from hydrotel_clone.snow import DegreJourModifie, init_state, init_ce
from meandre.data.basin_cache import BasinCache
from meandre.data.hydrotel_calib import load_melt_nodes, load_passage_pluie_neige
from meandre.utils import paths as _paths

torch.set_default_dtype(torch.float64)
REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PLAT = f"{_paths.PLATFORMS_ROOT}/LN24HA/{REG.upper()}_LN24HA_2020"


def charger():
    """Forcage + sites + ancrages, restreints aux noeuds des sites CanSWE."""
    bc = BasinCache(_paths.data_path("quebec", f"{REG}.duckdb"))
    h = bc.load(device="cpu")
    mes, sites = bc.load_canswe("2000-01-01", "2024-12-31")
    noeuds = sorted(set(int(n) for n in mes.node_idx.unique()))
    ix = {n: i for i, n in enumerate(noeuds)}

    f = _paths.data_path("quebec", f"forcing-{REG}-hyb.nc")
    if not os.path.exists(f):
        f = _paths.data_path("quebec", f"forcing-{REG}-budyko.nc")
    ds = xr.open_dataset(f)
    F = ds["forcing"].values[:, noeuds, :]
    times = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    sw = None
    fsw = _paths.data_path("quebec", f"forcing-{REG}-swin.nc")
    if os.path.exists(fsw):
        d2 = xr.open_dataset(fsw)
        sw = torch.tensor(d2["forcing"].values[:, noeuds, 0])
        d2.close()

    t = h["territorial"]
    def gp(k):
        v = t.get_physical(k)
        return v[noeuds] if v is not None else None
    # occupation par classes de fonte, comme la colonne (repli : tout en feuillus)
    conif = gp("f_forest_conifer_raw")
    mixte = gp("f_forest_mixed_raw")
    feuil = gp("f_forest_deciduous_raw")
    foret = gp("f_forest_raw")
    if conif is None:
        conif = torch.zeros(len(noeuds)); feuil = foret if foret is not None else torch.full((len(noeuds),), 0.5)
    else:
        feuil = (feuil if feuil is not None else 0) + (mixte if mixte is not None else 0)
    pct_c = torch.as_tensor(conif).double()
    pct_f = torch.as_tensor(feuil).double()
    pct_d = torch.clamp(1.0 - pct_c - pct_f, 0.0, 1.0)

    lat = torch.tensor(h["node_coords"].numpy()[noeuds, 1]).double()
    ce1, ce0 = init_ce(lat, torch.zeros_like(lat), torch.zeros_like(lat))  # terrain plat
    mp = load_melt_nodes(PLAT, [h["node_ids"][n] for n in noeuds])
    seuil_air = load_passage_pluie_neige(PLAT)
    return dict(F=torch.tensor(F), times=times, sw=sw, mes=mes, ix=ix,
                lat=lat, ce1=ce1, ce0=ce0, mp=mp, seuil_air=seuil_air,
                pct=(pct_c, pct_f, pct_d), n=len(noeuds))


def split(P, tmin, tmax, seuil, ea=None, twb_seuil=None):
    s = torch.full_like(P, float(seuil))
    if twb_seuil is not None:
        T = (tmin + tmax) / 2.0
        es = 0.6108 * torch.exp(17.27 * T / (T + 237.3))
        rh = torch.clamp(ea / es, 0.05, 1.0) * 100.0
        twb = (T * torch.atan(0.151977 * torch.sqrt(rh + 8.313659)) + torch.atan(T + rh)
               - torch.atan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * torch.atan(0.023101 * rh)
               - 4.686035)
        s = float(twb_seuil) + (T - twb)
    taux = torch.clamp((tmax - s) / (tmax - tmin + 1e-6), 0.0, 1.0)
    taux = torch.where(tmax < s, torch.zeros_like(taux), taux)
    taux = torch.where(tmin >= s, torch.ones_like(taux), taux)
    return taux * P, (1.0 - taux) * P


def simuler(d, seuil_mode="twb", twb=-0.8, amp=None, melt_mode="degree_day",
            tf=1.2e-3, srf=9.4e-6, trans=None, tf_wind=None):
    """Rejoue la neige seule. Retourne la serie SWE (T, n) en mm."""
    mp = d["mp"]
    p = dict(lat=d["lat"], ce1=d["ce1"], ce0=d["ce0"],
             pct_conifers=d["pct"][0], pct_feuillus=d["pct"][1], pct_autres=d["pct"][2],
             coeff_fonte_conifers=mp["taux_c"] / 1000.0,
             coeff_fonte_feuillus=mp["taux_f"] / 1000.0,
             coeff_fonte_decouver=mp["taux_d"] / 1000.0,
             seuil_fonte_conifers=mp["seuil_c"], seuil_fonte_feuillus=mp["seuil_f"],
             seuil_fonte_decouver=mp["seuil_d"],
             taux_fonte_geo=mp["taux_geo"], densite_max=mp["dens_max"],
             constante_tassement=mp["tasse"], melt_mode=melt_mode,
             tf=torch.full((d["n"],), float(tf)), srf=torch.full((d["n"],), float(srf)))
    if amp is not None:
        p["melt_seasonal_amp"] = float(amp)
    if trans is not None:            # transmissivite de canopee (conifers, feuillus, decouvert)
        p["eti_trans_conifers"], p["eti_trans_feuillus"], p["eti_trans_decouver"] =             (float(x) for x in trans)
    if tf_wind is not None:          # terme turbulent tf + tf_wind*u2
        p["eti_tf_wind"] = float(tf_wind)
    mod = DegreJourModifie(24)
    st = init_state(d["n"], dtype=torch.float64)
    F, times = d["F"], d["times"]
    doys = torch.tensor(times.dayofyear.values, dtype=torch.float64)
    out = torch.empty(len(times), d["n"])
    with torch.no_grad():
        for i in range(len(times)):
            tmin, tmax = F[i, :, 1], F[i, :, 2]
            pluie, neige = split(F[i, :, 0], tmin, tmax,
                                 d["seuil_air"] if seuil_mode == "air" else 0.0,
                                 ea=F[i, :, 5],
                                 twb_seuil=(twb if seuil_mode == "twb" else None))
            sw_i = d["sw"][i] if (melt_mode == "eti" and d["sw"] is not None) else None
            _, st = mod(tmin, tmax, pluie, neige, doys[i], st, p, sw_in=sw_i,
                        u2=F[i, :, 4])
            out[i] = st["couvert_nival_mm"]
    return out


def juger(d, swe):
    """Ratios simule/mesure par mois, apparies site-jour comme le pilote."""
    m = d["mes"].copy()
    pos = pd.Series(np.arange(len(d["times"])), index=d["times"].normalize())
    m["t"] = pos.reindex(pd.DatetimeIndex(m.date).normalize()).to_numpy()
    m = m[np.isfinite(m.t)]
    m["j"] = [d["ix"][int(n)] for n in m.node_idx]
    m["sim"] = swe.numpy()[m.t.astype(int), m.j]
    m = m[np.isfinite(m.swe_mm) & (m.swe_mm >= 0)]
    m["mois"] = pd.DatetimeIndex(m.date).month
    r = {}
    for mo, g in m.groupby("mois"):
        if len(g) >= 20 and g.swe_mm.sum() > 0:
            r[int(mo)] = float(g.sim.sum() / g.swe_mm.sum())
    return r


def calendrier(d, swe):
    """CLIMATOLOGIE MENSUELLE du manteau, simule et mesure, en millimetres.

    Motif (2026-08-27). Deux jours de recherche sur la RETENTION du stockage ont elimine
    quatre leviers de reservoir lent : nappe trois puis dix fois plus lente, couche 3 non
    lineaire, et leur combinaison (R48). Un echec aussi complet suggere qu'il n'y a rien
    a freiner, et que le probleme est en amont : si l'eau de fonte ARRIVE trop tot, le
    stockage monte trop tot et redescend trop tot, et aucun frein sur la vidange ne
    corrige une entree mal datee.

    Les ratios ne peuvent pas repondre a ca : un ratio de 0.5 en avril ne dit pas si le
    manteau est deux fois trop leger tout l'hiver ou s'il a fondu trois semaines trop
    tot. Il faut les DEUX courbes, mois par mois, avec leur mois de pic et leur mois de
    disparition.
    """
    m = d["mes"].copy()
    pos = pd.Series(np.arange(len(d["times"])), index=d["times"].normalize())
    m["t"] = pos.reindex(pd.DatetimeIndex(m.date).normalize()).to_numpy()
    m = m[np.isfinite(m.t)]
    m["j"] = [d["ix"][int(n)] for n in m.node_idx]
    m["sim"] = swe.numpy()[m.t.astype(int), m.j]
    m = m[np.isfinite(m.swe_mm) & (m.swe_mm >= 0)]
    m["mois"] = pd.DatetimeIndex(m.date).month
    g = m.groupby("mois")[["sim", "swe_mm"]].mean()
    mois = [10, 11, 12, 1, 2, 3, 4, 5, 6]
    sim = [float(g.sim.get(x, np.nan)) for x in mois]
    obs = [float(g.swe_mm.get(x, np.nan)) for x in mois]
    print("")
    print("  manteau nival, climatologie mensuelle appariee site-jour (mm)")
    print("           " + "".join(f"{x:7d}" for x in mois))
    print("  simule   " + "".join(f"{v:7.0f}" for v in sim))
    print("  CanSWE   " + "".join(f"{v:7.0f}" for v in obs))
    def pic(v):
        k = int(np.nanargmax(v)); return mois[k]
    def fin(v):
        # premier mois du printemps ou le manteau tombe sous 10 % de son pic
        s = max(v); 
        for i, x in enumerate(v):
            if mois[i] in (3, 4, 5, 6) and x < 0.1 * s:
                return mois[i]
        return None
    print(f"  pic : simule mois {pic(sim)}, mesure mois {pic(obs)} | "
          f"disparition : simule mois {fin(sim)}, mesure mois {fin(obs)}")


def ligne(nom, r):
    print(f"  {nom:<34s}" + "".join(f"{r.get(mo, float('nan')):6.2f}"
                                    for mo in (11, 12, 1, 2, 3, 4)))


if __name__ == "__main__":
    d = charger()
    print(f"[{REG}] {d['n']} noeuds de sites | seuil air du projet {d['seuil_air']:+.2f}\n")
    print("  " + " " * 34 + "".join(f"{m:>6d}" for m in (11, 12, 1, 2, 3, 4)))
    # CONTROLE : la recette du pilote (Twb -0.8, amp 0.5) doit retrouver ses ratios
    # MODULATION SAISONNIERE DE LA FONTE, mise en accusation le 2026-08-27. Elle vaut
    # 1 + amp*sin(2*pi*(j-81)/365), donc elle AMPLIFIE la fonte a partir de fin mars :
    # +31 % fin avril a amp=0.5. Or CanSWE montre que le manteau simule perd un quart de
    # sa masse en avril quand le manteau reel n'en perd rien (SAGU 185->141 contre
    # 241->238). Elle avait ete validee sur le DEBIT (R36), jamais sur la MASSE de neige.
    # On la balaie ici contre son juge propre.
    for _a in (0.5, 0.25, 0.0, -0.25):
        _s = simuler(d, amp=(None if _a == 0.0 else _a))
        ligne(f"amp fonte {_a:+.2f}", juger(d, _s))
        calendrier(d, _s)

    # SEUIL DU PARTAGE PLUIE-NEIGE. CaSR est innocente sur le TOTAL hivernal (rapport
    # median 0.948 contre ECCC, 2026-08-27) : le deficit de 22 % du manteau simule des
    # decembre a SAGU vient donc probablement du PARTAGE, pas de la donnee. Le bulbe
    # humide est cale a -0.8 (R43) sur l'ensemble du domaine ; on le balaie ici contre
    # le seul juge qui compte pour cette question, la MASSE mesuree par CanSWE, et non
    # le debit qui l'a valide a l'origine.
    for _twb in (-0.8, -1.4, -2.0, -2.6):
        _s = simuler(d, twb=_twb)
        ligne(f"bulbe humide {_twb:+.1f}", juger(d, _s))
        calendrier(d, _s)
    ligne("seuil air projet, sans amp", juger(d, simuler(d, seuil_mode="air", amp=None)))
    ligne("Twb-0.8 sans amp", juger(d, simuler(d, amp=None)))
    if d["sw"] is not None:
        for tf, srf in ((1.2e-3, 9.4e-6), (4e-3, 4.7e-5), (2.5e-3, 2.5e-5)):
            ligne(f"ETI tf={tf*1000:g} srf={srf*1000:g}",
                  juger(d, simuler(d, amp=None, melt_mode="eti", tf=tf, srf=srf)))
    else:
        print("  (pas de cache sw_in : variantes ETI sautees)")
