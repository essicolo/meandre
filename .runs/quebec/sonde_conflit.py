"""Conflit crue contre étiage, champ par champ (2026-10-27).

Question : un entraînement en deux étapes, la crue d'abord puis l'étiage sous garde, défera-t-il
la première étape ? Cela dépend des champs que les deux objectifs veulent déplacer en sens
contraires. La sonde le mesure sur le modèle chargé, sans entraîner : pour chaque champ du réseau
spatial, un facteur multiplicatif valant un, et la dérivée de deux erreurs par rapport à lui.

- Erreur de crue : écart absolu du débit, rapporté au débit moyen observé de la station, sur les
  jours où le débit observé dépasse son 75e centile.
- Erreur d'étiage : écart absolu des logarithmes du débit sur les jours où le débit observé est
  sous son 30e centile.

Les deux s'additionnent jour par jour, donc se calculent par blocs en détachant l'état entre eux,
la même approximation que l'entraînement. Une année de mise en régime sans gradient précède.
Appelée par le pilote (`MEANDRE_SONDE_CONFLIT=1`), qui définit le modèle.
"""
import os

import numpy as np
import torch


def mesurer(model, td, device, spinup=365, bloc=30, sortie=None):
    from meandre.utils.state import HydroState
    # Deux retropropagations dans le meme graphe : le sol compile les refuse avec ses tampons donnes.
    import torch._functorch.config as _fc
    _fc.donated_buffer = False
    sp0 = model.spatial_encoder(td.node_coords, td.territorial.to_tensor())
    import dataclasses
    champs = [f.name for f in dataclasses.fields(type(sp0)) if torch.is_tensor(getattr(sp0, f.name))]
    mults = {c: torch.ones((), device=device, requires_grad=True) for c in champs}
    model.spatial_encoder.multiplicateurs = mults
    T = td.forcing.shape[0]
    t0 = int(td.train_slice.start)
    q_obs = td.q_obs.to(device)   # q_obs[0] correspond a forcing[t0]
    st = td.station_idx.to(device)
    ok = torch.isfinite(q_obs)
    qo = torch.nan_to_num(q_obs)
    hi = torch.stack([torch.quantile(q_obs[ok[:, j], j], 0.75) if ok[:, j].sum() > 30 else torch.tensor(float("inf"), device=device) for j in range(q_obs.shape[1])])
    lo = torch.stack([torch.quantile(q_obs[ok[:, j], j], 0.30) if ok[:, j].sum() > 30 else torch.tensor(-1.0, device=device) for j in range(q_obs.shape[1])])
    moy = torch.stack([q_obs[ok[:, j], j].mean() if ok[:, j].sum() > 30 else torch.tensor(1.0, device=device) for j in range(q_obs.shape[1])]).clamp(min=1e-3)
    etat = HydroState.zeros(td.forcing.shape[1], device=device)
    with torch.no_grad():
        _, etat = model.simulate(forcing=td.forcing[:spinup], initial_state=etat, graph=td.graph, node_coords=td.node_coords, territorial=td.territorial, withdrawals=td.withdrawals, day_of_year=td.day_of_year[:spinup])
    g = {"crue": np.zeros(len(champs)), "etiage": np.zeros(len(champs))}
    n = {"crue": 0, "etiage": 0}
    a = max(spinup, t0)
    n_blocs = (T - a + bloc - 1) // bloc
    print(f"[conflit] {T - a} jours apres mise en regime, {n_blocs} blocs de {bloc}, {len(champs)} champs", flush=True)
    for b in range(n_blocs):
        a0, a1 = a + b * bloc, min(a + (b + 1) * bloc, T)
        Q, etat = model.simulate(forcing=td.forcing[a0:a1], initial_state=etat, graph=td.graph, node_coords=td.node_coords, territorial=td.territorial, withdrawals=td.withdrawals.slice(a0, a1), day_of_year=td.day_of_year[a0:a1])
        qs = Q[:, st]
        o = qo[a0 - t0:a1 - t0]
        m = ok[a0 - t0:a1 - t0]
        crue = m & (o > hi)
        etiage = m & (o < lo) & (o > 0)
        termes = {}
        if crue.any():
            termes["crue"] = ((qs - o).abs() / moy)[crue].sum()
            n["crue"] += int(crue.sum())
        if etiage.any():
            termes["etiage"] = (torch.log(qs.clamp(min=1e-4)) - torch.log(o.clamp(min=1e-4))).abs()[etiage].sum()
            n["etiage"] += int(etiage.sum())
        for k, (nom, val) in enumerate(termes.items()):
            gr = torch.autograd.grad(val, [mults[c] for c in champs], retain_graph=(k < len(termes) - 1), allow_unused=True)
            g[nom] += np.array([float(x) if x is not None else 0.0 for x in gr])
        etat = etat.detach()
        if (b + 1) % 6 == 0 or b == n_blocs - 1:
            print(f"[conflit] bloc {b + 1}/{n_blocs}", flush=True)
    gc = g["crue"] / max(n["crue"], 1)
    ge = g["etiage"] / max(n["etiage"], 1)
    sc, se = np.abs(gc).max() or 1.0, np.abs(ge).max() or 1.0
    cos = float(gc @ ge / (np.linalg.norm(gc) * np.linalg.norm(ge) + 1e-30))
    print(f"\n[conflit] {n['crue']} jours de crue, {n['etiage']} jours d'etiage (station-jours) ; cosinus des deux gradients : {cos:+.2f}")
    print(f"{'champ':<24s} {'crue':>10s} {'etiage':>10s}  lecture")
    ordre = np.argsort(-(np.abs(gc) / sc + np.abs(ge) / se))
    for i in ordre:
        rc, re = gc[i] / sc, ge[i] / se
        if max(abs(rc), abs(re)) < 0.01:
            continue
        if abs(rc) >= 0.05 and abs(re) >= 0.05:
            lecture = "CONFLIT" if np.sign(rc) != np.sign(re) else "accord"
        else:
            lecture = "crue seule" if abs(rc) >= 0.05 else ("etiage seul" if abs(re) >= 0.05 else "faible")
        print(f"{champs[i]:<24s} {rc:>+10.3f} {re:>+10.3f}  {lecture}")
    print("(derivees rapportees au maximum de chaque objectif ; une derivee positive veut dire qu'augmenter le champ AUGMENTE l'erreur)")
    if sortie:
        np.savez_compressed(sortie, champs=np.array(champs), crue=gc, etiage=ge, n_crue=n["crue"], n_etiage=n["etiage"])
        print(f"ecrit : {sortie}")
