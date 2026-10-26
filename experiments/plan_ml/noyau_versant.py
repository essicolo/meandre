"""Candidate 5.6, épreuve 3.3 : la forme de la réponse rapide est-elle commune, et
transfère-t-elle aux stations, aux années et aux territoires jamais vus ?

Le diagnostic du matin se lisait sur des parts cumulées par orage, sensibles à la
pluie des jours suivants. Ici la forme est mesurée de deux façons, sur les débits du
modèle provincial gelé et le forçage de bassin, sans simulation nouvelle :

A. Profil moyen de la hausse sur sept jours, normalisé par orage. On mesure la dispersion
   entre stations et entre orages, puis on éprouve le transfert du profil commun aux
   stations, aux années et aux territoires retirés.
B. Déconvolution pluie-débit. Chaque orage a une amplitude libre c_i et un noyau commun k
   positif de somme un : r_i = c_i (k * P_i). k porte la forme, c_i le volume.

    .venv/Scripts/python.exe experiments/plan_ml/noyau_versant.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
FENETRE = 7
PLIS = 5
GRAINE = 0


def evenements(p):
    """Un enregistrement par orage d'été : pluies des sept jours qui suivent."""
    sec = (p.shift(1) < 2.0) & (p.shift(2) < 2.0)
    ev = (p >= 10.0) & sec & p.index.month.isin((6, 7, 8, 9))
    lignes = []
    for t in p.index[ev]:
        i = p.index.get_loc(t)
        if i < 4 or i + FENETRE >= len(p):
            continue
        pluie = p.iloc[i:i + FENETRE].to_numpy(dtype=float)
        if not np.all(np.isfinite(pluie)) or pluie.sum() < 10.0:
            continue
        lignes.append({"date": t, "pluie": pluie})
    return lignes


def hausse(q, date, base):
    """Hausse sur sept jours au-dessus du débit de la veille, bornée à zéro."""
    i = q.index.get_loc(date)
    if i < 1:
        return None
    r = (q.iloc[i:i + FENETRE].to_numpy(dtype=float) - float(base))
    if not np.all(np.isfinite(r)):
        return None
    return np.clip(r, 0.0, None)


def normalise(profils):
    somme = profils.sum(1, keepdims=True)
    return profils / np.where(somme > 0, somme, 1.0)


def distance(a, b):
    """Distance L1 entre deux profils de somme un, de 0 à 2."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def profil_moyen(profils):
    """Profil moyen d'un groupe d'orages, chaque orage pesant pareil."""
    return profils.mean(0) / profils.mean(0).sum()


def retard(profil):
    return float((np.asarray(profil) * np.arange(FENETRE)).sum())


def dependance(nom, retard_orage, intensite, api, stations):
    """Corrélation du retard de la forme avec la taille de la pluie et l'humidité antécédente."""
    ok = np.isfinite(retard_orage) & np.isfinite(intensite) & np.isfinite(api)
    rho_intensite = spearmanr(retard_orage[ok], intensite[ok]).statistic
    rho_api = spearmanr(retard_orage[ok], api[ok]).statistic
    rhos, pentes = [], []
    for st in sorted(set(stations)):
        m = (stations == st) & ok
        if m.sum() < 20:
            continue
        rhos.append(spearmanr(retard_orage[m], intensite[m]).statistic)
        x = intensite[m] - intensite[m].mean()
        y = retard_orage[m] - retard_orage[m].mean()
        if x @ x > 0:
            pentes.append(float(x @ y / (x @ x)))
    print(f"  {nom:20s} corrélation avec la pluie {rho_intensite:+.2f} (dans la station "
          f"{np.median(rhos):+.2f}, pente {np.median(pentes):+.2f} j par décade de pluie)"
          if rhos else
          f"  {nom:20s} corrélation avec la pluie {rho_intensite:+.2f} (trop peu d'orages par station)",
          end="")
    print(f" | avec l'humidité antécédente {rho_api:+.2f}")


def convolue(pluie, noyau):
    return np.convolve(pluie, noyau)[:FENETRE]


def ajuste(pluies, hausses, iterations=40):
    """Alterne l'amplitude par orage et le noyau commun, positif et de somme un."""
    n = len(pluies)
    if n == 0:
        return None
    noyau = np.full(FENETRE, 1.0 / FENETRE)
    for _ in range(iterations):
        amp = np.zeros(n)
        for i in range(n):
            cp = convolue(pluies[i], noyau)
            den = float(cp @ cp)
            amp[i] = max(0.0, float(hausses[i] @ cp) / den) if den > 0 else 0.0
        lignes, cible = [], []
        for i in range(n):
            if amp[i] <= 0:
                continue
            for j in range(FENETRE):
                ligne = np.zeros(FENETRE)
                for lag in range(FENETRE):
                    if j - lag >= 0:
                        ligne[lag] = amp[i] * pluies[i][j - lag]
                lignes.append(ligne)
                cible.append(hausses[i][j])
        if not lignes:
            return None
        a, b = np.vstack(lignes), np.asarray(cible)
        res = minimize(lambda x: float(np.sum((a @ x - b) ** 2)), noyau, method="SLSQP",
                       bounds=[(0.0, 1.0)] * FENETRE,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                       options={"maxiter": 200, "ftol": 1e-10})
        nouveau = np.clip(res.x, 0.0, None)
        nouveau = nouveau / nouveau.sum()
        if np.max(np.abs(nouveau - noyau)) < 1e-6:
            return nouveau
        noyau = nouveau
    return noyau


def erreur_forme(noyau, pluies, hausses):
    """Erreur de forme sur des orages, amplitude réajustée par orage ; médiane."""
    vals = []
    for p, r in zip(pluies, hausses):
        cp = convolue(p, noyau)
        den = float(cp @ cp)
        total = float(r.sum())
        if den <= 0 or total <= 1e-6:
            continue
        c = max(0.0, float(r @ cp) / den)
        vals.append(float(np.abs(r - c * cp).sum()) / total)
    return float(np.median(vals)) if vals else float("nan")


def affiche_noyau(nom, k):
    k = np.asarray(k, dtype=float)
    k = k / k.sum()
    print(f"  {nom:22s} " + " ".join(f"{v:.3f}" for v in k) +
          f" | retard {retard(k):.2f} j | pic {int(np.argmax(k))} j")


def principale():
    rng = np.random.default_rng(GRAINE)
    q = pd.read_parquet(f"{REF}/debits.parquet")
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")

    pluies, obs, sim, stations, regions, annees = [], [], [], [], [], []
    api_rel = []
    for st in q["q_obs"].columns:
        o, s = q["q_obs"][st], q["q_sim"][st]
        if o.notna().sum() < 365 or st not in at.index or at.loc[st, "aire_km2"] <= 0:
            continue
        if st not in fx.columns.get_level_values(0):
            continue
        conv = 86.4 / float(at.loc[st, "aire_km2"])
        p = fx[st]["P"].reindex(q.index)
        api = p.ewm(alpha=0.1, adjust=False).mean().shift(1) * 10.0
        reference = np.nanmedian(api[p.index.month.isin((6, 7, 8, 9))])
        for e in evenements(p):
            ro = hausse(o, e["date"], o.shift(1).loc[e["date"]])
            rs = hausse(s, e["date"], s.shift(1).loc[e["date"]])
            if ro is None or rs is None:
                continue
            ro, rs = ro * conv, rs * conv
            if ro.sum() < 1.5 or rs.sum() < 1.5:
                continue
            pluies.append(e["pluie"])
            obs.append(ro)
            sim.append(rs)
            stations.append(st)
            regions.append(st.split(":")[0])
            annees.append(e["date"].year)
            valeur = float(api.loc[e["date"]])
            api_rel.append(valeur / reference if reference and np.isfinite(reference) and reference > 0 else np.nan)

    pluies = np.asarray(pluies)
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    stations = np.asarray(stations)
    regions = np.asarray(regions)
    annees = np.asarray(annees)
    api_rel = np.asarray(api_rel)
    no, ns = normalise(obs), normalise(sim)
    print(f"{len(pluies)} orages, {len(set(stations))} stations, {len(set(regions))} territoires, "
          f"{annees.min()}-{annees.max()}")

    # A. Forme mesurée sans déconvolution.
    print("\nprofil moyen des sept jours, part de la pluie et de la hausse :")
    affiche_noyau("pluie", pluies.mean(0))
    affiche_noyau("hausse observée", obs.mean(0))
    affiche_noyau("hausse simulée", sim.mean(0))
    dominant = (pluies / pluies.sum(1, keepdims=True))[:, 0] > 0.8
    if dominant.sum() >= 50:
        print(f"  orages à pluie du jour 0 majoritaire, {int(dominant.sum())} sur {len(pluies)} :")
        affiche_noyau("  hausse observée", obs[dominant].mean(0))
        affiche_noyau("  hausse simulée", sim[dominant].mean(0))
    uniques = sorted(set(stations))
    prof_obs, prof_sim = {}, {}
    for st in uniques:
        m = stations == st
        if m.sum() < 20:
            continue
        prof_obs[st] = profil_moyen(obs[m])
        prof_sim[st] = profil_moyen(sim[m])
    sts = sorted(prof_obs)
    if len(sts) >= 10:
        po = np.vstack([prof_obs[s] for s in sts])
        ps = np.vstack([prof_sim[s] for s in sts])
        commun_o = po.mean(0) / po.mean(0).sum()
        commun_s = ps.mean(0) / ps.mean(0).sum()
        print(f"\ndispersion des profils, {len(sts)} stations d'au moins vingt orages :")
        print(f"  distance L1 médiane à la forme commune : observé "
              f"{np.median([distance(p, commun_o) for p in po]):.3f}, "
              f"simulé {np.median([distance(p, commun_s) for p in ps]):.3f}")
        ecart_intra = [float(np.median([distance(no[i], prof_obs[s])
                                       for i in np.where(stations == s)[0]])) for s in sts]
        print(f"  dispersion interne à une station, orage à orage : "
              f"{np.median(ecart_intra):.3f}")
        print(f"  dispersion orage à orage sur tout l'échantillon : observé "
              f"{np.median([distance(p, commun_o) for p in no]):.3f}, "
              f"simulé {np.median([distance(p, commun_s) for p in ns]):.3f}")
        print(f"  retard moyen observé {np.median([retard(p) for p in po]):.2f} j, "
              f"simulé {np.median([retard(p) for p in ps]):.2f} j")
        plus_rapide = sum(retard(prof_obs[s]) < retard(prof_sim[s]) for s in sts)
        print(f"  l'observé est plus rapide que le simulé à {plus_rapide} stations sur {len(sts)}")
        print(f"  distance entre la forme commune observée et la simulée : "
              f"{distance(commun_o, commun_s):.3f}, contre {np.median([distance(p, commun_o) for p in po]):.3f} "
              "de dispersion entre stations et "
              f"{np.median([distance(p, commun_o) for p in no]):.3f} entre orages")

        # La forme dépend-elle de l'état de l'orage, et le modèle la fait-il suivre ?
        intensite = np.log10(pluies.sum(1))
        retard_obs = (no * np.arange(FENETRE)).sum(1)
        retard_sim = (ns * np.arange(FENETRE)).sum(1)
        print("\ndépendance de la forme à l'état de l'orage, corrélations de rang :")
        dependance("retard observé", retard_obs, intensite, api_rel, stations)
        dependance("retard simulé", retard_sim, intensite, api_rel, stations)
        if dominant.sum() >= 50:
            print("  même mesure sur les seuls orages à pluie d'un jour, où la durée de pluie ne confond plus :")
            dependance("  retard observé", retard_obs[dominant], intensite[dominant],
                       api_rel[dominant], stations[dominant])
            dependance("  retard simulé", retard_sim[dominant], intensite[dominant],
                       api_rel[dominant], stations[dominant])

        # Transfert du profil commun, stations et territoires retirés.
        pli = {s: int(i % PLIS) for i, s in enumerate(rng.permutation(sts))}
        app = np.array([pli[s] for s in sts])
        resultats = {"commun observé": [], "commun simulé": [], "plat": []}
        for f in range(PLIS):
            tr = app != f
            te = ~tr
            co = po[tr].mean(0) / po[tr].mean(0).sum()
            cs = ps[tr].mean(0) / ps[tr].mean(0).sum()
            plat = np.full(FENETRE, 1.0 / FENETRE)
            for i in np.where(te)[0]:
                resultats["commun observé"].append(distance(co, po[i]))
                resultats["commun simulé"].append(distance(cs, po[i]))
                resultats["plat"].append(distance(plat, po[i]))
        print("\ntransfert du profil commun aux stations retirées, distance L1 à l'observé :")
        for nom, vals in resultats.items():
            print(f"  {nom:16s} médiane {np.median(vals):.3f}")
        tr = annees <= 2021
        te = annees >= 2022
        if tr.any() and te.any():
            mo = [profil_moyen(obs[tr & (stations == s)]) for s in sts if (tr & (stations == s)).sum() >= 10]
            co = np.mean(mo, axis=0)
            co = co / co.sum()
            print("  années retirées, 2019-2021 vers 2022-2024 : distance L1 médiane "
                  f"{np.median([distance(co, prof_obs[s]) for s in sts if te[(stations == s)].sum() >= 10]):.3f}")
        print("  territoire retiré entier :")
        for reg in sorted(set(regions)):
            m_te = regions == reg
            if m_te.sum() < 50:
                continue
            m_tr = ~m_te
            autres = [profil_moyen(obs[m_tr & (stations == s)]) for s in sts
                      if (m_tr & (stations == s)).sum() >= 10]
            autres_sim = [profil_moyen(sim[m_tr & (stations == s)]) for s in sts
                          if (m_tr & (stations == s)).sum() >= 10]
            if not autres:
                continue
            co = np.mean(autres, axis=0)
            co = co / co.sum()
            cs = np.mean(autres_sim, axis=0)
            cs = cs / cs.sum()
            locaux = [s for s in sts if regions[np.where(stations == s)[0][0]] == reg]
            if not locaux:
                continue
            print(f"    {reg:5s} {len(locaux):3d} stations | commun observé "
                  f"{np.median([distance(co, prof_obs[s]) for s in locaux]):.3f} | "
                  f"commun simulé {np.median([distance(cs, prof_obs[s]) for s in locaux]):.3f}")

    # B. Déconvolution pluie-débit.
    print("\ndéconvolution, noyau commun à somme un, amplitude libre par orage :")
    k_obs = ajuste(pluies, obs)
    k_sim = ajuste(pluies, sim)
    k_un = np.zeros(FENETRE)
    k_un[1] = 1.0
    affiche_noyau("observé", k_obs)
    affiche_noyau("simulé", k_sim)
    affiche_noyau("retard d'un jour", k_un)
    print("\nretrait de stations, cinq plis, erreur de forme sur les orages retirés :")
    pli = {s: int(i % PLIS) for i, s in enumerate(rng.permutation(uniques))}
    app = np.array([pli[s] for s in stations])
    resultats = {"observé": [], "simulé": [], "un jour": []}
    for f in range(PLIS):
        tr = app != f
        te = ~tr
        resultats["observé"].append(erreur_forme(ajuste(pluies[tr], obs[tr]), pluies[te], obs[te]))
        resultats["simulé"].append(erreur_forme(ajuste(pluies[tr], sim[tr]), pluies[te], obs[te]))
        resultats["un jour"].append(erreur_forme(k_un, pluies[te], obs[te]))
    for nom, vals in resultats.items():
        print(f"  noyau {nom:10s} médiane {np.median(vals):.3f} | plis " +
              " ".join(f"{v:.3f}" for v in vals))
    tr = annees <= 2021
    te = annees >= 2022
    print("\nretrait d'années, forme apprise 2019-2021, éprouvée 2022-2024 :")
    print(f"  observé {erreur_forme(ajuste(pluies[tr], obs[tr]), pluies[te], obs[te]):.3f} | "
          f"simulé {erreur_forme(ajuste(pluies[tr], sim[tr]), pluies[te], obs[te]):.3f} | "
          f"un jour {erreur_forme(k_un, pluies[te], obs[te]):.3f}")
    print("\nretrait d'un territoire entier, erreur de forme sur ses orages :")
    for reg in sorted(set(regions)):
        te = regions == reg
        if te.sum() < 50:
            continue
        tr = ~te
        print(f"  {reg:5s} {int(te.sum()):4d} orages | observé "
              f"{erreur_forme(ajuste(pluies[tr], obs[tr]), pluies[te], obs[te]):.3f} | "
              f"simulé {erreur_forme(ajuste(pluies[tr], sim[tr]), pluies[te], obs[te]):.3f}")


if __name__ == "__main__":
    principale()
