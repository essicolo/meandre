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


def _largeur_et_pente(model, td, device):
    """Largeur du tronçon et pente, pour tirer un niveau d'eau du débit.

    La largeur vient de `physio/troncon_width_depth.csv` du projet Hydrotel, la pente du
    calage. Le modèle n'a aucune géométrie de lit : sa table des tronçons ne porte que
    l'identifiant, les coordonnées, un drapeau de lac et l'ordre topologique.
    """
    import pandas as pd

    plat = os.environ.get("ETL_MELT_DIR")
    if not plat:
        return None
    f = os.path.join(plat, "physio", "troncon_width_depth.csv")
    if not os.path.exists(f):
        print(f"[sensibilite] pas de geometrie de tronçon : {f}")
        return None
    t = pd.read_csv(f, sep=";", skipinitialspace=True)
    w = torch.tensor(t.iloc[:, 2].to_numpy(dtype="float32"), device=device)
    n = td.forcing.shape[1]
    if w.numel() < n:
        return None
    w = w[:n].clamp(min=1.0)
    pente = getattr(model.vertical_column, "slope", None)
    if pente is None or not torch.is_tensor(pente):
        pente = torch.full((n,), 0.01, device=device)
    return w, pente.to(device).clamp(min=1e-5)


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


def mesurer(model, td, times, device, jours=365, sortie=None, bloc=None):
    """Mesure appelée PAR LE PILOTE, une fois le modèle et les données construits.

    L'inversion est voulue : le pilote lit soixante-dix-neuf variables d'environnement et
    c'est lui qui définit le modèle. Reconstruire un modèle à côté mesurerait autre chose,
    piège que ce projet paie depuis des mois. Le pilote se termine par un arrêt brutal du
    processus, il ne peut donc pas rendre la main : c'est lui qui appelle.

    PAR BLOCS, comme l'entraînement. Le graphe d'une année entière sur trois mille tronçons
    ne tient pas dans huit gigaoctets, et celui de cent vingt jours non plus : mesuré le
    2026-09-22, la colonne déborde dans la conductivité de Campbell. On simule donc par blocs
    de quarante-cinq jours en détachant l'état entre eux, exactement l'approximation que la
    rétropropagation tronquée fait déjà pendant l'entraînement. Ce qui se perd est la part du
    gradient qui passerait par l'état porté d'un bloc au suivant.
    """
    import dataclasses

    import pandas as pd

    from meandre.utils.state import HydroState

    # NE RETENIR QUE LES QUATRE DIAGNOSTICS MESURES. Poser MEANDRE_DIAG_CPU a zero rend
    # toutes les listes ordinaires, donc une vingtaine de variables gardent leur graphe sur la
    # carte : bien PIRE que l'entrainement, et la colonne deborde des le premier bloc. On garde
    # le mecanisme de restriction et on nomme exactement ce dont on derive.
    os.environ["MEANDRE_DIAG_CPU"] = "1"
    # Les teneurs en eau des trois couches servent au stockage total, ce que la gravimetrie
    # mesure : sans elles ce terme sortait absent du tableau.
    os.environ["MEANDRE_DIAG_DERIVES"] = "etr,swe,s_gw,recharge,theta1,theta2,theta3"
    bloc = int(bloc or os.environ.get("MEANDRE_SENSIBILITE_BLOC", "15"))
    # Le pilote a deja fait tourner la colonne : on rend la memoire avant de construire un
    # graphe, faute de quoi le premier bloc part avec la carte a moitie pleine.
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mois_tout = pd.DatetimeIndex(times[:jours]).month.to_numpy()
    sp0 = model.spatial_encoder(td.node_coords, td.territorial.to_tensor())
    champs = [f.name for f in dataclasses.fields(type(sp0))
              if torch.is_tensor(getattr(sp0, f.name))]
    mults = {c: torch.ones((), device=device, requires_grad=True) for c in champs}
    model.spatial_encoder.multiplicateurs = mults
    cles = list(mults)
    n_blocs = (jours + bloc - 1) // bloc
    print(f"[sensibilite] {jours} jours en {n_blocs} blocs de {bloc}, {len(champs)} champs",
          flush=True)

    familles = [("debit", None), ("evapotranspiration", "etr"), ("neige", "swe"),
                ("nappe", "s_gw"), ("recharge", "recharge")]
    # GRACE et SWOT n'etaient pas dans le banc : le premier faute d'un resume de stockage
    # total, le second faute d'un niveau d'eau, que le modele ne produit pas. Les deux se
    # calculent ici sans toucher au modele.
    # Le champ spatial se recalcule A CHAQUE BLOC. Calcule une seule fois hors de la boucle,
    # son graphe est libere par la derniere retropropagation du premier bloc, et le bloc
    # suivant echoue en voulant y repasser.
    _geom = _largeur_et_pente(model, td, device)
    cumul, compte = {}, {}
    etat = HydroState.zeros(td.forcing.shape[1], device=device)
    stations = td.station_idx.detach().cpu().numpy()
    for b in range(n_blocs):
        a0, a1 = b * bloc, min((b + 1) * bloc, jours)
        Q, etat, diag = model.simulate(
            forcing=td.forcing[a0:a1], initial_state=etat,
            graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
            withdrawals=td.withdrawals, day_of_year=td.day_of_year[a0:a1],
            return_diagnostics=True)
        mois = mois_tout[a0:a1]
        sommes = {}
        # STOCKAGE TOTAL, ce que la gravimetrie mesure : sol, manteau, souterrain, canopee.
        _stock = None
        for _att, _ep in (("theta1", 0.3), ("theta2", None), ("theta3", None)):
            _v = getattr(diag, _att, None)
            if _v is None or not torch.is_tensor(_v) or not _v.requires_grad:
                _stock = None
                break
            _z = (_ep if _ep is not None
                  else float(getattr(model.vertical_column, "z2_ref", 1.0)))
            _stock = _v * _z * 1000.0 if _stock is None else _stock + _v * _z * 1000.0
        if _stock is not None:
            for _att in ("swe", "s_gw", "canopy"):
                _v = getattr(diag, _att, None)
                if _v is not None and torch.is_tensor(_v) and _v.requires_grad:
                    _stock = _stock + _v
            familles_sup = [("gravimetrie", _stock)]
        else:
            familles_sup = []
        # NIVEAU D'EAU, ce que SWOT mesure. Manning en section large : la profondeur vaut
        # (Q n / (w racine(S)))^(3/5). Le niveau absolu demanderait l'altitude du lit, que
        # nous n'avons pas ; SWOT se comparerait donc en ANOMALIES, comme les puits.
        if _geom is not None:
            _w, _pente = _geom
            _n_man = getattr(model.spatial_encoder(td.node_coords, td.territorial.to_tensor()),
                             "manning_n", None)
            if _n_man is not None:
                _h = (Q.clamp(min=1e-3) * _n_man / (_w * _pente.clamp(min=1e-5).sqrt())) ** 0.6
                familles_sup.append(("niveau", _h))
        for nom, att in familles + familles_sup:
            if torch.is_tensor(att):
                serie = att
            elif att is None:
                serie = torch.log(Q.clamp(min=1e-3))[:, stations]
            else:
                v = getattr(diag, att, None)
                if v is None or not torch.is_tensor(v) or not v.requires_grad:
                    continue
                serie = v
            for sais, m in SAISONS.items():
                masque = torch.tensor(np.isin(mois, m), device=serie.device)
                n = int(masque.sum())
                if not n:
                    continue
                cle = f"{nom}:{sais}"
                sommes[cle] = serie[masque].sum()
                compte[cle] = compte.get(cle, 0) + n * serie.shape[1]
        for k, (cle, valeur) in enumerate(sommes.items()):
            g = torch.autograd.grad(valeur, [mults[c] for c in cles],
                                    retain_graph=(k < len(sommes) - 1), allow_unused=True)
            v = np.array([float(x) if x is not None else 0.0 for x in g])
            cumul[cle] = cumul.get(cle, 0.0) + v
        etat = etat.detach()
        print(f"[sensibilite] bloc {b + 1}/{n_blocs}, {len(sommes)} resumes", flush=True)

    noms = sorted(cumul)
    J = np.array([cumul[n] / max(compte[n], 1) for n in noms])
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
    # TEMOIN NUL. Une matrice de meme forme, de memes normes de ligne, remplie de directions
    # tirees au hasard : c'est ce que le rang donnerait sans aucune structure. Sans lui, le
    # chiffre n'a pas d'echelle, defaut qui a produit deux conclusions fausses le meme jour.
    rng = np.random.default_rng(1234)
    faux = rng.standard_normal(J.shape)
    faux *= np.linalg.norm(J, axis=1, keepdims=True) / np.clip(
        np.linalg.norm(faux, axis=1, keepdims=True), 1e-30, None)
    d_bruit, part_bruit = _rang_effectif(faux)
    print("")
    print(f"ensemble : {d_tout} directions de parametres sur {len(champs)}, "
          f"participation {part_tout:.1f}")
    print(f"temoin de bruit : {d_bruit} directions, participation {part_bruit:.1f}")
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
