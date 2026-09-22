"""Ce que chaque observable contraint dans l'espace des PARAMÈTRES, et non des séries.

Toutes les mesures d'information faites jusqu'ici portent sur la redondance des SÉRIES : elles
disent si une observation est prédictible à partir des autres. Ce n'est pas la question. Une
observation prédictible à partir du débit peut contraindre une combinaison de paramètres que
le débit ne contraint pas, puisqu'elle regarde une autre sortie du modèle. C'est ce qui s'est
produit en mai, quand l'évapotranspiration et la gravimétrie ont décollapsé la partition
verticale d'un facteur six à huit alors qu'elles n'ajoutent aucune direction de série.

La grandeur qui répond est le rang de la matrice de SENSIBILITÉ : la dérivée de chaque
observable par rapport à chaque paramètre. Le modèle étant différentiable, elle se calcule.
C'est le seul endroit où la revendication centrale du projet se vérifie en acte.

MÉTHODE. Un facteur multiplicatif par champ du réseau spatial, quarante-trois en tout, valant
un au départ : la dérivée d'un observable à ce facteur est sans dimension, c'est sa réponse à
une variation relative uniforme du champ. Une seule passe avant, puis une rétropropagation par
résumé d'observable, le graphe étant retenu. Les résumés sont saisonniers, quatre par
observable, ce qui suffit à distinguer un mécanisme d'hiver d'un mécanisme d'été.

CE QUE LE RANG SIGNIFIE ICI. Les valeurs singulières de la matrice d'un observable disent
combien de combinaisons de paramètres il distingue. Un observable de rang un ne contraint
qu'une direction, quel que soit le nombre de ses mesures.

    ETL_REGION=outv .venv/bin/python .runs/quebec/sensibilite_des_observables.py --jours 365
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SAISONS = {"hiver": (12, 1, 2), "printemps": (3, 4, 5), "ete": (6, 7, 8), "automne": (9, 10, 11)}


def _rang_effectif(j, seuil=0.95):
    """Nombre de directions de paramètres qu'un bloc de sensibilité distingue.

    Chaque ligne est normalisée avant la décomposition : sans cela, un observable dont
    l'amplitude est grande dominerait la matrice pour une raison d'unités et non
    d'information. Retourne (directions à 95 %, rapport de participation).
    """
    j = np.asarray(j, dtype="float64")
    j = j[np.isfinite(j).all(axis=1)]
    if j.size == 0 or j.shape[0] < 1:
        return 0, np.nan
    n = np.linalg.norm(j, axis=1, keepdims=True)
    j = j[n[:, 0] > 1e-12] / np.clip(n[n[:, 0] > 1e-12], 1e-12, None)
    if j.shape[0] < 1:
        return 0, np.nan
    vp = np.linalg.svd(j, compute_uv=False) ** 2
    vp = vp[vp > vp.max() * 1e-12] if vp.size else vp
    if vp.size == 0:
        return 0, np.nan
    part = np.cumsum(vp) / vp.sum()
    return int(np.searchsorted(part, seuil) + 1), float(vp.sum() ** 2 / (vp ** 2).sum())


def _resumes(nom, serie, mois, noeuds=None):
    """Quatre moyennes saisonnières d'un observable, éventuellement restreint à des nœuds."""
    out = {}
    x = serie if noeuds is None else serie[:, noeuds]
    for s, m in SAISONS.items():
        masque = torch.tensor(np.isin(mois, m), device=x.device)
        if not bool(masque.any()):
            continue
        out[f"{nom}:{s}"] = x[masque].mean()
    return out


def mesurer(model, td, times, device, jours=365, sortie=None):
    """Mesure appelée PAR LE PILOTE, une fois le modèle et les données construits.

    L'inversion est voulue : le pilote lit soixante-dix-neuf variables d'environnement et
    c'est lui qui définit le modèle. Reconstruire un modèle à côté mesurerait autre chose,
    piège que ce projet paie depuis des mois. Le pilote se termine par un arrêt brutal du
    processus, il ne peut donc pas rendre la main : c'est lui qui appelle.
    """
    import dataclasses

    import pandas as pd

    from meandre.utils.state import HydroState

    # Les diagnostics doivent porter le graphe : on annule le deplacement sur le processeur,
    # qui est pose pour l'entrainement.
    os.environ["MEANDRE_DIAG_CPU"] = "0"
    mois = pd.DatetimeIndex(times[:jours]).month.to_numpy()
    sp0 = model.spatial_encoder(td.node_coords, td.territorial.to_tensor())
    champs = [f.name for f in dataclasses.fields(type(sp0))
              if torch.is_tensor(getattr(sp0, f.name))]
    mults = {c: torch.ones((), device=device, requires_grad=True) for c in champs}
    model.spatial_encoder.multiplicateurs = mults
    print(f"[sensibilite] passe avant sur {jours} jours, {len(champs)} champs", flush=True)

    Q, _etat, diag = model.simulate(
        forcing=td.forcing[:jours],
        initial_state=HydroState.zeros(td.forcing.shape[1], device=device),
        graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
        withdrawals=td.withdrawals, day_of_year=td.day_of_year[:jours],
        return_diagnostics=True)

    obs = {}
    obs.update(_resumes("debit", torch.log(Q.clamp(min=1e-3)), mois,
                        td.station_idx.detach().cpu().numpy()))
    for nom, att in (("evapotranspiration", "etr"), ("neige", "swe"),
                     ("nappe", "s_gw"), ("recharge", "recharge")):
        v = getattr(diag, att, None)
        if v is not None and torch.is_tensor(v) and v.requires_grad:
            obs.update(_resumes(nom, v, mois))
    print(f"[sensibilite] {len(obs)} resumes, retropropagation", flush=True)

    lignes, noms = [], []
    cles = list(mults)
    for k, (nom, valeur) in enumerate(obs.items()):
        g = torch.autograd.grad(valeur, [mults[c] for c in cles],
                                retain_graph=(k < len(obs) - 1), allow_unused=True)
        lignes.append([float(x) if x is not None else 0.0 for x in g])
        noms.append(nom)
    J = np.array(lignes)
    _rapport(J, noms, cles)
    if sortie:
        np.savez_compressed(sortie, J=J, observables=np.array(noms), champs=np.array(cles))
        print(f"ecrit : {sortie}")
    return J, noms, cles


def _rapport(J, noms, champs):
    """Tableau des directions de paramètres, par observable puis en apport propre."""
    print("")
    print(f"{'observable':<22s} {'resumes':>8s} {'directions':>11s} {'participation':>14s}")
    familles = sorted({n.split(":")[0] for n in noms})
    for fam in familles:
        idx = [i for i, n in enumerate(noms) if n.startswith(fam + ":")]
        d, part = _rang_effectif(J[idx])
        part_t = f"{part:>14.1f}" if np.isfinite(part) else f"{'—':>14s}"
        print(f"{fam:<22s} {len(idx):>8d} {d:>11d} {part_t}")
    d_tout, part_tout = _rang_effectif(J)
    print("")
    print(f"ensemble : {d_tout} directions de parametres sur {len(champs)}, "
          f"participation {part_tout:.1f}")
    print("")
    print(f"{'observable retire':<22s} {'directions restantes':>21s} {'apport propre':>15s}")
    for fam in familles:
        idx = [i for i, n in enumerate(noms) if not n.startswith(fam + ":")]
        d_sans, _ = _rang_effectif(J[idx])
        print(f"{fam:<22s} {d_sans:>21d} {d_tout - d_sans:>15d}")
    # LES CHAMPS QUE PERSONNE NE VOIT. Une colonne nulle de la matrice designe un parametre
    # qu'aucune observation ne touche : il n'est pas mal identifie, il n'est pas identifie du
    # tout, et l'entrainement ne peut que le laisser ou le prior le met.
    norme = np.linalg.norm(J, axis=0)
    seuil = norme.max() * 1e-6 if norme.size and norme.max() > 0 else 0.0
    muets = [champs[i] for i in np.argsort(norme)[:10] if norme[i] <= max(seuil, 1e-30)]
    print("")
    if muets:
        print(f"champs qu'AUCUN observable ne touche ({len(muets)}) : {', '.join(muets)}")
    else:
        ordre = np.argsort(norme)[:6]
        print("champs les moins vus : "
              + ", ".join(f"{champs[i]} {norme[i]:.1e}" for i in ordre))


def main():
    """Lance le PILOTE, qui construira le modele puis rappellera `mesurer`."""
    p = argparse.ArgumentParser()
    p.add_argument("--jours", type=int, default=365)
    p.add_argument("--sortie", default=None)
    a = p.parse_args()
    os.environ["MEANDRE_SENSIBILITE"] = "1"
    os.environ["MEANDRE_SENSIBILITE_JOURS"] = str(a.jours)
    if a.sortie:
        os.environ["MEANDRE_SENSIBILITE_SORTIE"] = a.sortie
    os.environ.setdefault("ETL_EPOCHS", "0")
    pilote = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etl_run.py")
    import runpy

    runpy.run_path(pilote, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
