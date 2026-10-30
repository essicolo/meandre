"""Bac à sable : une colonne de sol continue produit-elle d'elle-même les formes observées ?

Question (R447, R449, R451). Les débits observés montrent un coefficient de ruissellement d'épisode
qui croît avec la pluie (pente log-log 1,2 à 1,5) et avec l'humidité antérieure (0,6), et un étiage
qui tient l'été. Le modèle donne 0,8 et 0,25, parce que son profil (0,95 m) est entièrement à portée
des racines et ne sèche jamais. Ce bac à sable compare, sans calage et avec les mêmes lois partout,
deux discrétisations d'un même sol :

- « trois couches » : 0,15, 0,30 et 0,50 m, racines uniformes sur tout le profil (la situation du
  modèle en forêt) ;
- « continu » : vingt mailles jusqu'à 2,5 m, densité racinaire continue de Jackson et al. (1996),
  fraction cumulée 1 − β^z (z en cm).

Une seule physique, sans catégorie d'eau : rétention et conductivité de Campbell, Darcy vertical avec
K(θ), vidange latérale de chaque maille le long du versant par K(θ), prélèvement racinaire selon la
densité de racines et l'eau utile de la maille, débordement de surface quand la maille du haut sature.
Entrée : apport au sol du modèle gelé (pluie et fonte, moyenne du bassin), ETP de Priestley-Taylor.
Sortie : écoulement en mm/j, comparé à l'observé par les pentes d'épisode de R447 et la tenue de l'étiage.

Usage : python sandbox.py <dossier des dumps> <territoire> [beta] [variantes : base,continu,retention,lateral,vsa,tout]
"""
import sys

import numpy as np
import polars as pl
import xarray as xr

# Loam de Clapp et Hornberger (1978) : porosité, potentiel d'entrée d'air (m), exposant b, Ks (m/j).
THETA_S, PSI_S, B, K_S = 0.451, -0.478, 5.39, 0.60
SLOPE, HILLSLOPE_M = 0.05, 50.0
SUBSTEPS = 96


def theta_at(psi):
    return THETA_S * (psi / PSI_S) ** (-1.0 / B)


THETA_FC, THETA_WP = theta_at(-3.3), theta_at(-150.0)


def conductivity(theta):
    return K_S * np.clip(theta / THETA_S, 1e-6, 1.0) ** (2 * B + 3)


def potential(theta):
    return PSI_S * np.clip(theta / THETA_S, 1e-3, 1.0) ** (-B)


def root_fractions(edges_m, beta, max_depth_m=None):
    """Part des racines dans chaque maille, Jackson et al. (1996) ; `max_depth_m` coupe et renormalise."""
    z_cm = edges_m * 100.0
    cumulative = 1.0 - beta ** z_cm
    r = np.diff(cumulative)
    if max_depth_m is not None:
        r = np.where(edges_m[1:] <= max_depth_m + 1e-9, r, 0.0)
    return r / r.sum()


def run(apport_mm, etp_mm, thickness_m, roots, theta_s=None, theta_fc=None, theta_wp=None, lateral_threshold=True, vsa_b=None, direct_fraction=0.0):
    """Options, une à la fois, pour réintroduire les éléments de la colonne de meandre : rétention par
    maille (`theta_s`, `theta_fc`, `theta_wp`), écoulement latéral sans seuil de capacité au champ
    (Darcy sur toute l'eau, forme BASE_LINEAR), aire contributive (θ1/θs)^b sur l'apport."""
    n_days, n_cells = len(apport_mm), len(thickness_m)
    ts = np.full(n_cells, THETA_S) if theta_s is None else np.asarray(theta_s, float)
    fc = np.full(n_cells, THETA_FC) if theta_fc is None else np.asarray(theta_fc, float)
    wp = np.full(n_cells, THETA_WP) if theta_wp is None else np.asarray(theta_wp, float)
    theta = 0.5 * (fc + ts)
    centers = np.cumsum(thickness_m) - thickness_m / 2
    gap = np.diff(centers)
    dt = 1.0 / SUBSTEPS
    q_out = np.zeros(n_days)
    et_out = np.zeros(n_days)
    for day in range(n_days):
        p = apport_mm[day] / 1000.0
        demand = etp_mm[day] / 1000.0
        runoff = lateral = et = 0.0
        for _ in range(SUBSTEPS):
            k = K_S * np.clip(theta / ts, 1e-6, 1.0) ** (2 * B + 3)
            psi = PSI_S * np.clip(theta / ts, 1e-3, 1.0) ** (-B)
            k_face = np.sqrt(k[:-1] * k[1:])
            q_down = k_face * ((psi[:-1] - psi[1:]) / gap + 1.0) * dt
            store = theta * thickness_m
            # un flux ne prend pas plus que l'eau au-dessus du point de flétrissement de la maille donneuse
            q_down = np.where(q_down > 0, np.minimum(q_down, np.maximum(store[:-1] - wp[:-1] * thickness_m[:-1], 0) * 0.5), np.maximum(q_down, -np.maximum(store[1:] - wp[1:] * thickness_m[1:], 0) * 0.5))
            q_lat = k * SLOPE * thickness_m / HILLSLOPE_M * dt
            floor = fc if lateral_threshold else wp
            q_lat = np.minimum(q_lat, np.maximum(store - floor * thickness_m, 0) * 0.5 + 1e-12)
            stress = np.clip((theta - wp) / (fc - wp), 0, 1)
            uptake = demand * dt * roots * stress
            uptake = np.minimum(uptake, np.maximum(store - wp * thickness_m, 0))
            store = store - uptake - q_lat
            store[:-1] -= q_down
            store[1:] += q_down
            p_in = p * dt
            if direct_fraction > 0:
                # pluie sur les surfaces imperméables et l'eau libre, rendue directement à la rivière
                runoff += p_in * direct_fraction
                p_in -= p_in * direct_fraction
            if vsa_b is not None:
                r_vsa = p_in * np.clip(theta[0] / ts[0], 0, 1) ** vsa_b
                runoff += r_vsa
                p_in -= r_vsa
            store[0] += p_in
            excess = np.maximum(store - ts * thickness_m, 0)
            # le surplus d'une maille saturée remonte vers la surface, puis ruisselle
            for i in range(n_cells - 1, 0, -1):
                store[i - 1] += excess[i]
                store[i] -= excess[i]
                excess[i - 1] = max(store[i - 1] - ts[i - 1] * thickness_m[i - 1], 0)
            runoff += excess[0]
            store[0] -= excess[0]
            theta = store / thickness_m
            lateral += q_lat.sum()
            et += uptake.sum()
        q_out[day] = (runoff + lateral) * 1000.0
        et_out[day] = et * 1000.0
    return q_out, et_out


def event_slopes(P, Q, Qprev_series, months):
    rows = []
    for k in range(Q.shape[1]):
        for t in range(3, len(P) - 7):
            if not (6 <= months[t] <= 10) or P[t, k] < 10 or P[t - 3:t, k].sum() > 2:
                continue
            q, qp = Q[:, k], Qprev_series[:, k]
            if np.isnan(q[t - 1:t + 7]).any() or q[t - 1] <= 0:
                continue
            v = np.clip(q[t:t + 7] - q[t - 1], 0, None).sum()
            if v > 0:
                rows.append((k, np.log(v / P[t:t + 7, k].sum()), np.log(P[t:t + 7, k].sum()), np.log(qp[t - 1])))
    r = np.array(rows)
    y, X = r[:, 1].copy(), r[:, 2:].copy()
    for k in np.unique(r[:, 0]):
        sel = r[:, 0] == k
        y[sel] -= y[sel].mean()
        X[sel] -= X[sel].mean(axis=0)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b[0], b[1], len(r)


def main(folder, territory, beta=0.97):
    s = np.load(f"{folder}/stations-{territory}.npz", allow_pickle=True)
    qz = np.load(f"{folder}/q-{territory}.npz", allow_pickle=True)
    meta = pl.read_parquet(f"{folder}/stations-meta-{territory}.parquet")
    dates = np.array(s["dates"], dtype="datetime64[D]")
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    P = s["basin_snowmelt"].astype(float)
    f = xr.open_dataset(f"D:/meandre-data/quebec/forcing-{territory}-casr-brut.nc")["forcing"].sel(time=slice(str(dates[0]), str(dates[-1])))
    nodes = meta["node_idx"].to_numpy()
    rn = f.sel(var="R_n").values[:, nodes]
    tmean = 0.5 * (f.sel(var="Tmin").values[:, nodes] + f.sel(var="Tmax").values[:, nodes])
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098 * es / (tmean + 237.3) ** 2
    etp = np.clip(1.26 * delta / (delta + 0.066) * rn / 2.45, 0, None)
    obs = qz["q_obs"] * 86.4 / meta["drainage_area_km2"].to_numpy()[None, :]

    three = np.array([0.15, 0.30, 0.50])
    meandre_retention = dict(theta_s=[0.60, 0.60, 0.60], theta_fc=[0.49, 0.41, 0.51], theta_wp=[0.15, 0.17, 0.22])
    chosen = sys.argv[4].split(",") if len(sys.argv) > 4 else ["base", "continu"]
    catalogue = {
        "base": ("trois couches, 0,95 m, racines partout", three, None, {}),
        "continu": ("continu, 2,5 m, racines de Jackson", np.full(20, 0.125), beta, {}),
        "retention": ("trois couches + rétention de meandre", three, None, meandre_retention),
        "lateral": ("trois couches + latéral sans seuil", three, None, dict(lateral_threshold=False)),
        "vsa": ("trois couches + aire contributive b=8", three, None, dict(vsa_b=8.0)),
        "direct": ("trois couches + 8 % direct (imperméable, eau)", three, None, dict(direct_fraction=0.08)),
        "continu-direct": ("continu + 8 % direct", np.full(20, 0.125), beta, dict(direct_fraction=0.08)),
        "tout": ("trois couches + les trois", three, None, dict(meandre_retention, lateral_threshold=False, vsa_b=8.0)),
    }
    variants = {catalogue[c][0]: catalogue[c][1:] for c in chosen}
    summer = np.isin(months, (7, 8, 9))
    bo, wo, n = event_slopes(P, obs, obs, months)
    print(f"{territory} : {P.shape[1]} stations ; OBSERVÉ pente pluie {bo:+.2f}, pente humidité {wo:+.2f} ({n} épisodes)")
    for label, (thick, b, options) in variants.items():
        edges = np.concatenate([[0.0], np.cumsum(thick)])
        roots = root_fractions(edges, b) if b is not None else thick / thick.sum()
        Q = np.zeros_like(P)
        ET = np.zeros_like(P)
        for k in range(P.shape[1]):
            # mise en régime : la première année deux fois
            n0 = 365
            run_in = np.concatenate([P[:n0, k], P[:, k]])
            e_in = np.concatenate([etp[:n0, k], etp[:, k]])
            q, e = run(run_in, e_in, thick, roots, **options)
            Q[:, k], ET[:, k] = q[n0:], e[n0:]
        bs, ws, n = event_slopes(P, Q, Q, months)
        low = [np.nanmedian(Q[summer & (months == mo)]) / max(np.nanmedian(Q[months == 6]), 1e-9) for mo in (7, 8, 9)]
        low_obs = [np.nanmedian(obs[summer & (months == mo)]) / max(np.nanmedian(obs[months == 6]), 1e-9) for mo in (7, 8, 9)]
        print(f"  {label:40s} pente pluie {bs:+.2f}, pente humidité {ws:+.2f} | débit médian juil/août/sept rapporté à juin {' '.join(f'{x:.2f}' for x in low)} (observé {' '.join(f'{x:.2f}' for x in low_obs)}) | ET d'été {np.nanmean(ET[summer]):.2f} mm/j, volume sim/obs {np.nansum(np.where(np.isfinite(obs), Q, 0)) / np.nansum(obs):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.97)
