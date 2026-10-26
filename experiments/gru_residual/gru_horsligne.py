"""Étape 3 en version rapide : un GRU hors ligne en aval du modèle provincial gelé, aux
stations, mêmes prédicteurs et même validation B que plafond.py (apprentissage sur les
stations des autres plis, évaluation 2022-2024 aux stations retirées). Les attributs
statiques et les paramètres du champ spatial au tronçon exutoire jouent le rôle du
conditionnement par le champ spatial : ce qui marche ici est un plafond de ce qu'une
correction spatialisée peut transférer aux stations non jaugées.

Arrêt précoce : apprentissage sur 2019-2020, arrêt sur 2021 aux stations d'apprentissage ;
2022-2024 n'est jamais vu avant l'évaluation.

Deux sorties :
  additive : y = (Q_obs - Q_sim) / moy(Q_sim), comme les arbres ;
  bornee : Q = Q_sim · (1 + 0,3 tanh(h)), le GRU minimal de l'été, perte sur l'écart
           divisé par le débit moyen simulé.

    .venv/Scripts/python.exe experiments/gru_residual/gru_horsligne.py [graines]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plafond

plafond.CIBLE = "lin"
DOSSIER = plafond.DOSSIER
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Correcteur(nn.Module):
    def __init__(self, n_in, cache=32, sortie="additive"):
        super().__init__()
        self.gru = nn.GRU(n_in, cache, batch_first=True)
        self.tete = nn.Linear(cache, 1)
        nn.init.zeros_(self.tete.weight)
        nn.init.zeros_(self.tete.bias)
        self.sortie = sortie

    def forward(self, x):
        h, _ = self.gru(x)
        return self.tete(h).squeeze(-1)


def tenseurs(D, cols, stations, temps):
    """(S, T, F) entrées, cible, q_sim, moyenne simulée, masque de mesure."""
    S, T = len(stations), len(temps)
    X = np.zeros((S, T, len(cols)), np.float32)
    Y = np.full((S, T), np.nan, np.float32)
    QS = np.zeros((S, T), np.float32)
    QM = np.ones(S, np.float32)
    pos = {t: k for k, t in enumerate(temps)}
    for i, st in enumerate(stations):
        g = D[D.station == st]
        k = np.array([pos[t] for t in g.index])
        X[i, k] = g[cols].to_numpy(np.float32)
        Y[i, k] = g["y"].to_numpy(np.float32)
        QS[i, k] = g["q_sim"].to_numpy(np.float32)
        QM[i] = g["qm"].iloc[0]
    return X, Y, QS, QM


def perte(m, x, y, qs, qm, masque):
    h = m(x)
    if m.sortie == "additive":
        pred = h
    else:
        pred = qs * 0.3 * torch.tanh(h) / qm[:, None]
    e = (pred - y) ** 2
    return e[masque].mean(), pred


def entrainer(Xtr, Ytr, QStr, QMtr, idx_app, idx_arret, sortie, graine, epoques=400):
    torch.manual_seed(graine)
    m = Correcteur(Xtr.shape[-1], sortie=sortie).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    x, y = torch.tensor(Xtr, device=DEV), torch.tensor(np.nan_to_num(Ytr), device=DEV)
    qs, qm = torch.tensor(QStr, device=DEV), torch.tensor(QMtr, device=DEV)
    mes = torch.tensor(np.isfinite(Ytr), device=DEV)
    m_app = mes.clone()
    m_app[:, idx_arret] = False
    m_arr = mes.clone()
    m_arr[:, idx_app] = False
    meilleur, etat, patience = np.inf, None, 0
    for ep in range(epoques):
        m.train()
        opt.zero_grad()
        l, _ = perte(m, x, y, qs, qm, m_app)
        l.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        m.eval()
        with torch.no_grad():
            la, _ = perte(m, x, y, qs, qm, m_arr)
        if la.item() < meilleur - 1e-5:
            meilleur, etat, patience = la.item(), {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            patience += 1
            if patience > 40:
                break
    m.load_state_dict(etat)
    return m, ep


def main():
    graines = [int(g) for g in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    D = plafond.construire()
    G = plafond.groupes(D)
    temps = D.index.unique().sort_values()
    stations = np.array(sorted(D.station.unique()))
    i_2021 = np.where(temps.year == 2021)[0]
    i_app = np.where(temps.year <= 2020)[0]
    fin_app = np.where(temps.year <= 2021)[0][-1] + 1
    res = []
    for nom_g in ("minimal", "minimal+forçage+neige", "tout+statiques"):
        cols = G[nom_g]
        for sortie in ("additive", "bornee"):
            for graine in graines:
                yhat = pd.Series(np.nan, index=pd.MultiIndex.from_arrays([D.station, D.index]))
                for _, ite in GroupKFold(n_splits=5).split(stations, groups=stations):
                    tenues = stations[ite]
                    app = stations[np.setdiff1d(np.arange(len(stations)), ite)]
                    Dapp = D[D.station.isin(app) & (D.annee <= 2021)]
                    mu, sd = Dapp[cols].mean(), Dapp[cols].std().replace(0, 1)
                    Dn = D.copy()
                    Dn[cols] = ((D[cols] - mu) / sd).fillna(0)
                    X, Y, QS, QM = tenseurs(Dn[Dn.station.isin(app)], cols, app, temps)
                    m, ep = entrainer(X[:, :fin_app], Y[:, :fin_app], QS[:, :fin_app], QM, i_app, i_2021, sortie, graine)
                    Xt, _, QSt, QMt = tenseurs(Dn[Dn.station.isin(tenues)], cols, tenues, temps)
                    with torch.no_grad():
                        h = m(torch.tensor(Xt, device=DEV)).cpu().numpy()
                    pred = h if sortie == "additive" else QSt * 0.3 * np.tanh(h) / QMt[:, None]
                    for i, st in enumerate(tenues):
                        yhat.loc[st] = pred[i][np.searchsorted(temps, yhat.loc[st].index)]
                te = (D.annee >= 2022).to_numpy()
                yh = yhat.to_numpy()
                r, L = plafond.evaluer(D[te], yh[te])
                r.update(predicteurs=nom_g, sortie=sortie, graine=graine)
                res.append(r)
                print(f"  {nom_g:22s} {sortie:8s} graine {graine} : R² {r['r2_global']:+.3f} ; KGE médian {r['kge_av']:.3f} -> {r['kge_ap']:.3f}, {100 * r['part_ameliorees']:.0f} % améliorées ; |volume-1| {r['vol_av']:.3f} -> {r['vol_ap']:.3f} ; |log étiage| {r['etiage_av']:.3f} -> {r['etiage_ap']:.3f}", flush=True)
    pd.DataFrame(res).to_csv(f"{DOSSIER}/gru_horsligne.csv", index=False)


if __name__ == "__main__":
    main()
