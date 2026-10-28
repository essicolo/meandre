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

Groupes de stations : `groups` associe un identifiant de station à un groupe (par exemple naturel
ou influencé, tiré du régime des stations). Les deux gradients sont alors calculés aussi pour
chaque groupe, pour savoir si un conflit vient des stations régularisées.
"""
import numpy as np
import torch

OBJECTIVES = ("flood", "low_flow")
LABELS = {"flood": "crue", "low_flow": "etiage"}


def _print_table(fields, g_flood, g_low):
    s_flood, s_low = np.abs(g_flood).max() or 1.0, np.abs(g_low).max() or 1.0
    print(f"{'champ':<24s} {'crue':>10s} {'etiage':>10s}  lecture")
    order = np.argsort(-(np.abs(g_flood) / s_flood + np.abs(g_low) / s_low))
    for i in order:
        rf, rl = g_flood[i] / s_flood, g_low[i] / s_low
        if max(abs(rf), abs(rl)) < 0.01:
            continue
        if abs(rf) >= 0.05 and abs(rl) >= 0.05:
            reading = "CONFLIT" if np.sign(rf) != np.sign(rl) else "accord"
        else:
            reading = "crue seule" if abs(rf) >= 0.05 else ("etiage seul" if abs(rl) >= 0.05 else "faible")
        print(f"{fields[i]:<24s} {rf:>+10.3f} {rl:>+10.3f}  {reading}")


def measure(model, td, device, spinup=365, block=30, output=None, station_ids=None, groups=None):
    from meandre.utils.state import HydroState
    # Deux rétropropagations dans le même graphe : le sol compilé les refuse avec ses tampons donnés.
    import torch._functorch.config as _fc
    _fc.donated_buffer = False
    sp0 = model.spatial_encoder(td.node_coords, td.territorial.to_tensor())
    import dataclasses
    fields = [f.name for f in dataclasses.fields(type(sp0)) if torch.is_tensor(getattr(sp0, f.name))]
    multipliers = {c: torch.ones((), device=device, requires_grad=True) for c in fields}
    model.spatial_encoder.multipliers = multipliers
    n_steps = td.forcing.shape[0]
    t0 = int(td.train_slice.start)
    q_obs = td.q_obs.to(device)   # q_obs[0] correspond à forcing[t0]
    stations = td.station_idx.to(device)
    valid = torch.isfinite(q_obs)
    q_filled = torch.nan_to_num(q_obs)

    def per_station(fn, default):
        return torch.stack([fn(q_obs[valid[:, j], j]) if valid[:, j].sum() > 30 else torch.tensor(default, device=device) for j in range(q_obs.shape[1])])

    high = per_station(lambda x: torch.quantile(x, 0.75), float("inf"))
    low = per_station(lambda x: torch.quantile(x, 0.30), -1.0)
    mean = per_station(lambda x: x.mean(), 1.0).clamp(min=1e-3)
    # Masques de colonnes par groupe : « tout », puis un par groupe connu.
    masks = {"tout": torch.ones(q_obs.shape[1], dtype=torch.bool, device=device)}
    if groups and station_ids is not None:
        for name in sorted(set(groups.values())):
            m = torch.tensor([groups.get(str(sid)) == name for sid in station_ids], dtype=torch.bool, device=device)
            if bool(m.any()):
                masks[name] = m
                print(f"[conflit] groupe {name} : {int(m.sum())} stations", flush=True)
    grads = {(g, o): np.zeros(len(fields)) for g in masks for o in OBJECTIVES}
    counts = {(g, o): 0 for g in masks for o in OBJECTIVES}
    state = HydroState.zeros(td.forcing.shape[1], device=device)
    with torch.no_grad():
        _, state = model.simulate(forcing=td.forcing[:spinup], initial_state=state, graph=td.graph, node_coords=td.node_coords, territorial=td.territorial, withdrawals=td.withdrawals, day_of_year=td.day_of_year[:spinup])
    start = max(spinup, t0)
    n_blocks = (n_steps - start + block - 1) // block
    print(f"[conflit] {n_steps - start} jours apres mise en regime, {n_blocks} blocs de {block}, {len(fields)} champs", flush=True)
    for b in range(n_blocks):
        a0, a1 = start + b * block, min(start + (b + 1) * block, n_steps)
        q, state = model.simulate(forcing=td.forcing[a0:a1], initial_state=state, graph=td.graph, node_coords=td.node_coords, territorial=td.territorial, withdrawals=td.withdrawals.slice(a0, a1), day_of_year=td.day_of_year[a0:a1])
        q_sim = q[:, stations]
        obs = q_filled[a0 - t0:a1 - t0]
        ok = valid[a0 - t0:a1 - t0]
        flood_days = ok & (obs > high)
        low_days = ok & (obs < low) & (obs > 0)
        err_flood = (q_sim - obs).abs() / mean
        err_low = (torch.log(q_sim.clamp(min=1e-4)) - torch.log(obs.clamp(min=1e-4))).abs()
        terms = {}
        for g, mask in masks.items():
            fd, ld = flood_days & mask, low_days & mask
            if fd.any():
                terms[(g, "flood")] = err_flood[fd].sum()
                counts[(g, "flood")] += int(fd.sum())
            if ld.any():
                terms[(g, "low_flow")] = err_low[ld].sum()
                counts[(g, "low_flow")] += int(ld.sum())
        for k, (key, value) in enumerate(terms.items()):
            gr = torch.autograd.grad(value, [multipliers[c] for c in fields], retain_graph=(k < len(terms) - 1), allow_unused=True)
            grads[key] += np.array([float(x) if x is not None else 0.0 for x in gr])
        state = state.detach()
        if (b + 1) % 6 == 0 or b == n_blocks - 1:
            print(f"[conflit] bloc {b + 1}/{n_blocks}", flush=True)
    saved = {"champs": np.array(fields)}
    for g in masks:
        g_flood = grads[(g, "flood")] / max(counts[(g, "flood")], 1)
        g_low = grads[(g, "low_flow")] / max(counts[(g, "low_flow")], 1)
        cosine = float(g_flood @ g_low / (np.linalg.norm(g_flood) * np.linalg.norm(g_low) + 1e-30))
        print(f"\n[conflit] stations : {g} ; {counts[(g, 'flood')]} jours de crue, {counts[(g, 'low_flow')]} jours d'etiage (station-jours) ; cosinus des deux gradients : {cosine:+.2f}")
        _print_table(fields, g_flood, g_low)
        for o, vec in (("flood", g_flood), ("low_flow", g_low)):
            saved[f"{LABELS[o]}_{g}"] = vec
            saved[f"n_{LABELS[o]}_{g}"] = counts[(g, o)]
    print("(derivees rapportees au maximum de chaque objectif ; une derivee positive veut dire qu'augmenter le champ AUGMENTE l'erreur)")
    if output:
        np.savez_compressed(output, **saved)
        print(f"ecrit : {output}")
