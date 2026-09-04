"""Banc SYNTHETIQUE : trancher un mecanisme en secondes, sans bassin ni forcage reel.

POURQUOI (demande d'Essi, 2026-09-03). Une question de MECANISME -- brider la fonte de
decembre retient-il l'eau dans le manteau, et quand ressort-elle ? -- ne demande ni
station, ni reseau, ni vingt-cinq ans de meteo. Elle demande une colonne, un hiver, et
deux reglages a comparer. Le faire sur un bassin reel coute des heures de GPU et melange
la reponse a des dizaines d'autres effets ; le faire ici coute quelques secondes et ne
laisse qu'une variable libre.

Ce banc ne remplace pas la mesure sur bassin : il la PREPARE. Il dit si un mecanisme
existe et dans quel sens il joue, donc quelles valeurs meritent une passe couteuse. Un
mecanisme absent ici ne merite aucun essai sur bassin ; un mecanisme present ici doit
encore etre confirme la-bas, ou il peut etre masque par le reste de la chaine.

L'annee fabriquee est un climat boreal quebecois lisible :
  janvier a mars   froid franc, neige reguliere, DEUX redoux (jours 20 et 55)
  avril a mai      rechauffement progressif, la crue de printemps
  juin a septembre chaud et sec, quelques orages
  octobre a decembre  refroidissement, retour de la neige, UN redoux (jour 340)

Les redoux sont le coeur du test : c'est l'evenement que le modele efface quand la fonte
hivernale est trop bridee, et c'est ce que les hydrogrammes du rapport montrent comme des
plateaux de plusieurs semaines.

  python .runs/quebec/banc_synthetique.py fonte          amplitude de fonte saisonniere
  python .runs/quebec/banc_synthetique.py fonte 0,0.25,0.5,0.75
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from hydrotel_clone.frost import n_intervalles
from hydrotel_clone.snow import DegreJourModifie
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params

N = 6
REDOUX = (20, 55, 340)


def forcage_reel(reg="gasp", n_noeuds=8, graine=0):
    """Vraie meteo, systeme fabrique : P, Tmin, Tmax de quelques noeuds d'une region.

    Un climat invente demande de justifier sa representativite, et la premiere version de
    ce banc l'a montre : un janvier a -12 degres sans redoux ne fond jamais, donc aucun
    reglage de fonte n'y change quoi que ce soit. Le forcage reel supprime la question. Ce
    qui reste FABRIQUE, et qui fait la vitesse, c'est le systeme : une colonne isolee,
    sans reseau, sans routage, sans stations, sans entrainement.
    """
    import xarray as xr
    from meandre.utils import paths as _p
    d = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-hyb.nc")
    rng = np.random.default_rng(graine)
    idx = np.sort(rng.choice(d.sizes["node"], size=min(n_noeuds, d.sizes["node"]),
                             replace=False))
    f = d["forcing"].isel(node=idx).values          # (T, n, 6)
    t = d["time"].values
    d.close()
    return f, t


def annee_synthetique(n_annees=2, graine=0):
    """(P, Tmin, Tmax) par jour, climat boreal fabrique, avec redoux marques."""
    rng = np.random.default_rng(graine)
    P, TN, TX = [], [], []
    for _ in range(n_annees):
        for j in range(1, 366):
            # temperature moyenne : minimum le 20 janvier, maximum le 22 juillet
            tmoy = 5.0 - 17.0 * np.cos(2 * np.pi * (j - 20) / 365.0)
            if j in REDOUX:
                tmoy += 12.0                      # redoux franc, au-dessus de zero
            amp = 7.0
            tn, tx = tmoy - amp / 2, tmoy + amp / 2
            # precipitation : ~2.7 mm/j en moyenne, plus frequente en ete
            p = float(rng.gamma(0.55, 5.0)) if rng.random() < 0.45 else 0.0
            if 150 < j < 260 and rng.random() < 0.05:
                p += float(rng.gamma(2.0, 12.0))  # orage d'ete
            P.append(p); TN.append(tn); TX.append(tx)
    return np.array(P), np.array(TN), np.array(TX)


def _colonne(melt_seasonal_amp=None, n=N, et_mode="mcguinness"):
    """PIEGE PAYE le 2026-09-03 : `melt_seasonal_amp` n'est injecte dans les parametres
    de neige que par `params_from_nerf`, le chemin qui construit tout depuis le champ
    spatial. Un banc qui passe par `build_static_params` et `set_static` le contourne, et
    l'amplitude n'a alors AUCUN effet -- trois valeurs donnaient un resultat identique au
    millimetre, ce qui a trahi l'erreur. On l'injecte donc explicitement ici."""
    occ = dict(feuillus=0.25, conifers=0.45, ouverts=0.20, humides=0.05, eau=0.05)
    psnow, psoil, petr = build_static_params(
        n, lat=48.5, slope=0.03, orientation=7, texture="sandy_loam",
        z=(0.15, 0.45, 1.4), occupation=occ)
    if melt_seasonal_amp is not None:
        psnow["melt_seasonal_amp"] = float(melt_seasonal_amp)
    col = HydrotelColumn(et_mode=et_mode, use_frost=True)
    col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(2.0, 0.05))
    return col


def _swe(snow):
    return sum(float(snow[c][0].mean()) * 1000.0 for c in DegreJourModifie.CLASSES)


def essai_fonte_reel(amp, f, temps):
    """Colonne isolee sur vrai forcage, une amplitude. Retourne manteau et production."""
    torch.set_default_dtype(torch.float64)
    n = f.shape[1]
    col = _colonne(melt_seasonal_amp=amp, n=n)
    st = col.init_state(n, theta_init=(0.30, 0.30, 0.30))
    doys = np.array([int(str(x)[5:7]) for x in temps]), None
    import pandas as pd
    doy = pd.DatetimeIndex(temps).dayofyear.to_numpy()
    swe, prod = [], []
    tt = lambda a: torch.tensor(a, dtype=torch.float64)
    with torch.no_grad():
        for i in range(f.shape[0]):
            p, st, _ = col(tt(f[i, :, 0]), tt(f[i, :, 1]), tt(f[i, :, 2]),
                           tt(f[i, :, 3]), tt(f[i, :, 4]), tt(f[i, :, 5]),
                           float(doy[i]), st)
            swe.append(_swe(st.snow))
            prod.append(float(p.mean()))
    return np.array(swe), np.array(prod)


def banc_fonte(amplitudes, reg="gasp", n_noeuds=8):
    """L'amplitude de fonte retient-elle l'eau en hiver, et quand ressort-elle ?

    Colonne isolee sur le forcage REEL de la region : pas de reseau, pas de routage, pas
    de stations, pas d'entrainement. Une seule variable libre.
    """
    import pandas as pd
    f, temps = forcage_reel(reg, n_noeuds)
    # QUATRE ANNEES suffisent : la premiere met le manteau et le sol en regime, les trois
    # autres portent le verdict. Vingt-cinq ans coutaient plus de dix minutes par
    # amplitude et n'ajoutaient rien a une question de mecanisme.
    idx = pd.DatetimeIndex(temps)
    _garde4 = idx.year < idx.year.min() + 4
    f, temps = f[_garde4], temps[_garde4]
    idx = pd.DatetimeIndex(temps)
    mois = idx.month.to_numpy()
    an = idx.year.to_numpy()
    # On jette la premiere annee : mise en regime du manteau et du sol.
    garde = an > an.min()
    hiv = garde & np.isin(mois, (12, 1, 2, 3))
    pri = garde & np.isin(mois, (4, 5))
    ete = garde & np.isin(mois, (6, 7, 8, 9))
    aut = garde & np.isin(mois, (10, 11))
    n_ans = len(np.unique(an[garde]))

    print(f"Colonne isolee sur le forcage reel de {reg.upper()}, {n_noeuds} noeuds, "
          f"{n_ans} annees jugees (la premiere sert de mise en regime).")
    print("Production = eau quittant la colonne vers le troncon, en mm/an.")
    print()
    print(f"{'amplitude':>10s} {'manteau max':>12s} {'manteau 1er avr':>16s} "
          f"{'hiver':>8s} {'printemps':>10s} {'ete':>7s} {'automne':>8s} {'total':>7s}")
    lignes = []
    for a in amplitudes:
        swe, prod = essai_fonte_reel(a, f, temps)
        avr = float(np.median(swe[garde & (mois == 4) & (idx.day.to_numpy() == 1)]))
        v = dict(amp=a, swe_max=float(swe[garde].max()), swe_avr=avr,
                 hiv=float(prod[hiv].sum()) / n_ans, pri=float(prod[pri].sum()) / n_ans,
                 ete=float(prod[ete].sum()) / n_ans, aut=float(prod[aut].sum()) / n_ans,
                 tot=float(prod[garde].sum()) / n_ans)
        lignes.append(v)
        print(f"{a:10.2f} {v['swe_max']:11.0f}mm {avr:15.0f}mm {v['hiv']:7.0f}mm "
              f"{v['pri']:9.0f}mm {v['ete']:6.0f}mm {v['aut']:7.0f}mm {v['tot']:6.0f}mm")

    base = lignes[0]
    print()
    print(f"Ecarts par rapport a l'amplitude {base['amp']:.2f}, celle de la recette :")
    for v in lignes[1:]:
        print(f"  amplitude {v['amp']:.2f} : hiver {v['hiv']-base['hiv']:+.0f} mm/an | "
              f"printemps {v['pri']-base['pri']:+.0f} | ete {v['ete']-base['ete']:+.0f} | "
              f"automne {v['aut']-base['aut']:+.0f} | manteau au 1er avril "
              f"{v['swe_avr']-base['swe_avr']:+.0f} mm")
    d = lignes[-1]["hiv"] - base["hiv"]
    print()
    if d > 10:
        print(f"MECANISME CONFIRME : passer de {base['amp']:.2f} a {lignes[-1]['amp']:.2f} "
              f"libere {d:.0f} mm/an en hiver. La question merite une passe sur bassin.")
    elif abs(d) <= 10:
        print(f"MECANISME NEGLIGEABLE : {d:+.0f} mm/an d'ecart en hiver. L'amplitude "
              f"n'est pas le levier du deficit hivernal ; chercher ailleurs avant de "
              f"depenser du GPU.")
    else:
        print(f"SENS INVERSE : baisser l'amplitude RETIRE {-d:.0f} mm/an a l'hiver.")


def banc_bilan(reg="gasp", n_noeuds=8, modes=("mcguinness", "penman", "oudin")):
    """Ou part la precipitation ? Bilan de la colonne isolee, sur forcage reel.

    Motif (2026-09-03) : le banc de fonte a rendu 109 mm/an de production pour une
    Gaspesie qui recoit environ 1100 mm/an, soit un coefficient d'ecoulement de 0.10 quand
    la realite quebecoise tourne autour de 0.5. Avant de chercher un levier hivernal, il
    faut savoir si la colonne perd son eau, et par ou.
    """
    import pandas as pd
    f, temps = forcage_reel(reg, n_noeuds)
    idx = pd.DatetimeIndex(temps)
    g4 = idx.year < idx.year.min() + 4
    f, temps = f[g4], temps[g4]
    idx = pd.DatetimeIndex(temps)
    an = idx.year.to_numpy()
    garde = an > an.min()
    n_ans = len(np.unique(an[garde]))

    torch.set_default_dtype(torch.float64)
    n = f.shape[1]
    print(f"Bilan de la colonne isolee, {reg.upper()}, {n_ans} annees jugees, en mm/an")
    print(f"{'mode ETP':>12s} {'pluie':>7s} {'ETR':>7s} {'part ETR':>9s} "
          f"{'production':>11s} {'coef ecoul.':>12s}")
    for _mode in modes:
        _bilan_un_mode(f, temps, garde, n_ans, n, _mode)
    print()
    print("  ETR boreale reelle : 400 a 500 mm/an (litterature). Coefficient")
    print("  d'ecoulement au Quebec meridional : 0.45 a 0.65.")
    return


def _bilan_un_mode(f, temps, garde, n_ans, n, et_mode):
    import pandas as pd
    idx = pd.DatetimeIndex(temps)
    col = _colonne(n=n, et_mode=et_mode)
    st = col.init_state(n, theta_init=(0.30, 0.30, 0.30))
    doy = idx.dayofyear.to_numpy()
    tt = lambda a: torch.tensor(a, dtype=torch.float64)
    prod, etr = [], []
    with torch.no_grad():
        for i in range(f.shape[0]):
            p, st, dg = col(tt(f[i, :, 0]), tt(f[i, :, 1]), tt(f[i, :, 2]),
                            tt(f[i, :, 3]), tt(f[i, :, 4]), tt(f[i, :, 5]),
                            float(doy[i]), st)
            prod.append(float(p.mean()))
            etr.append(float(dg["etr"].mean()) if "etr" in dg else float("nan"))
    prod, etr = np.array(prod), np.array(etr)
    P = f[:, :, 0].mean(axis=1)
    pr = P[garde].sum() / n_ans
    pd_ = prod[garde].sum() / n_ans
    ev = etr[garde].sum() / n_ans
    print(f"{et_mode:>12s} {pr:6.0f}mm {ev:6.0f}mm {100*ev/max(pr,1e-9):8.0f}% "
          f"{pd_:10.0f}mm {pd_/max(pr,1e-9):12.2f}")


def main():
    quoi = sys.argv[1] if len(sys.argv) > 1 else "fonte"
    if quoi == "bilan":
        banc_bilan(sys.argv[2] if len(sys.argv) > 2 else "gasp")
        return
    if quoi != "fonte":
        raise SystemExit("bancs disponibles : fonte, bilan")
    amps = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 \
        else [0.5, 0.25, 0.0]
    banc_fonte(amps)


if __name__ == "__main__":
    main()
