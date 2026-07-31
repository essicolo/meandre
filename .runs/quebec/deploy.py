"""Déploiement provincial reproductible (release v0.1) : la carte est une fonction
pure de (données, code). Étape MEASURE : dérive les adaptateurs depuis les sources
(ratio bilan/MOD16, récessions de queue, règle de forçage) avec provenance ;
étape INFER : champion gelé + adaptateurs dérivés -> carte + rapport.

  python .runs/quebec/deploy.py             # pipeline complet
  python .runs/quebec/deploy.py --measure   # mesures seulement
"""
import os, sys, json
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib
import numpy as np
import pandas as pd
import torch
import duckdb
import xarray as xr
from collections import defaultdict

CFG = tomllib.load(open(".runs/quebec/deploy-v0.1.toml", "rb"))
P = CFG["protocole"]; D = CFG["donnees"]
REGIONS = ["gasp", "sagu", "mont", "labi", "abit", "cnda", "cndb", "cndc", "cndd", "cnde", "outm", "outv", "slno", "slso", "vaud"]


def dbp(reg):
    return D["base_slso"] if reg == "slso" else D["bases"].format(reg=reg)


def measure():
    """Dérive tous les adaptateurs depuis les données. Retourne dict + provenance JSON."""
    out = {}
    z = xr.open_zarr(D["grille_krigee"]); lat_max = float(z["latitude"].values.max()); z.close()
    from meandre.data.basin_cache import BasinCache
    for reg in REGIONS:
        prov = {"region": reg}
        cache = BasinCache(dbp(reg))
        h = cache.load(device="cpu")
        nc = h["node_coords"].numpy()
        lat_col = 0 if 40 < float(np.nanmean(nc[:, 0])) < 62 else 1
        north = float(np.quantile(nc[:, lat_col], 0.9))
        hyb = D["forcage_hyb"].format(reg=reg)
        con = duckdb.connect(dbp(reg), read_only=True)
        st = con.execute("select station_id, node_idx from stations").fetchdf()
        e = con.execute("select src, dst from edges").fetchdf()
        area = h["territorial"].get_physical("area_km2_local").numpy()
        up = defaultdict(list)
        for s_, d_ in zip(e["src"].values, e["dst"].values):
            up[int(d_)].append(int(s_))

        def anc(node):
            seen, stk = {node}, [node]
            while stk:
                for x in up.get(stk.pop(), []):
                    if x not in seen:
                        seen.add(x); stk.append(x)
            return list(seen)

        et = None
        try:
            et = cache.load_modis_et("2001-01-01", "2024-12-31", device="cpu").numpy()
        except Exception:
            pass

        def bilan_stats(path):
            """Écart de fermeture |P-(Q+ET)|/P (médiane stations) + ratios (P-Q)/ET."""
            if not os.path.exists(path) or et is None:
                return None, [], []
            dsl = xr.open_dataset(path); Pl = dsl["forcing"].values[:, :, 0]; dsl.close()
            errs, rat, usd = [], [], []
            for _, rr2 in st.iterrows():
                o2 = con.execute(f"select discharge from observations where station_id={rr2.station_id}").fetchdf()
                if len(o2) < P["debias_et_stations_min_annees"] * 365:
                    continue
                a2 = anc(int(rr2.node_idx)); akm2 = area[a2].sum()
                if akm2 < 100:
                    continue
                q2 = float(np.nanmean(o2["discharge"])) * 86400 * 365.25 / (akm2 * 1e6) * 1000
                w2 = area[a2] / area[a2].sum()
                p2 = float((Pl[:, a2] * w2).sum(axis=1).mean()) * 365.25
                etn2 = et[:, a2]; land2 = np.isfinite(etn2).sum(axis=0) >= 200
                if land2.sum() == 0:
                    continue
                wl2 = area[a2][land2] / area[a2][land2].sum()
                e2 = float((np.nanmean(etn2[:, land2], axis=0) * 365.25 * wl2).sum())
                if e2 > 50:
                    errs.append(abs(p2 - (q2 + e2)) / max(p2, 1))
                    rat.append((p2 - q2) / e2); usd.append(int(rr2.station_id))
            return (float(np.median(errs)) if errs else None), rat, usd

        # PRODUIT MÉTÉO : CaSR BRUT partout (décision Essi 2026-07-31). Un seul produit,
        # zéro prétraitement, zéro règle de sélection à défendre. Les variantes (SIMAT
        # hybride, CaSR corrigé budyko) restent dans le dépôt mais ne sont plus utilisées :
        # trois critères de sélection ont été essayés et réfutés ou jugés non concluants
        # (latitude, volume, fermeture du bilan) — cf. journal 2026-07-31.
        fx = D["forcage_casr_slso"] if reg == "slso" else D["forcage_casr"].format(reg=reg)
        prov["forcage"] = "casr_brut"
        prov["forcage_fichier"] = os.path.basename(fx)
        err_bilan, _, _ = bilan_stats(fx)
        prov["bilan_ecart_pct"] = round(err_bilan * 100, 1) if err_bilan is not None else None
        _, ratios_sel, used_sel = bilan_stats(fx)
        dsx = xr.open_dataset(fx); Pmm = dsx["forcing"].values[:, :, 0]
        times = pd.to_datetime(dsx["time"].values); dsx.close()
        ratios, used = ratios_sel, used_sel
        lo, hi = P["debias_et_bornes"]
        ds_val = float(np.clip(np.median(ratios), lo, hi)) if ratios else 1.0
        # (le choix de produit ci-dessus est confirmé ou infirmé par le bilan : voir
        #  prov["bilan_ecart_pct"], calculé sur le fichier retenu)
        if ratios:
            prov["bilan_ecart_pct"] = round(float(np.median([abs(1 - x) for x in ratios])) * 100, 1)
        prov["debias_et"] = round(ds_val, 3)
        prov["debias_et_stations"] = used
        m0, m1 = P["recession_fenetre_mois"]
        kt = []
        for _, rr in st.iterrows():
            o = con.execute(f"select date, discharge from observations where station_id={rr.station_id} order by date").fetchdf()
            if len(o) < 1000:
                continue
            o["date"] = pd.to_datetime(o["date"])
            q = o.set_index("date")["discharge"].asfreq("D")
            qmed = q.median()
            mask = q.index.month.isin(range(m0, m1 + 1))
            dq = q.diff(); dec = (dq < 0) & mask & q.notna()
            ks, seg = [], []
            for day, isd in dec.items():
                if isd:
                    seg.append(day)
                else:
                    if len(seg) >= P["recession_segment_min_jours"]:
                        qs = q.loc[seg].values
                        if np.all(qs > 0) and qs[0] < qmed:
                            k = -np.polyfit(np.arange(len(qs)), np.log(qs), 1)[0]
                            if 0.001 < k < 0.5:
                                ks.append(k)
                    seg = []
            if len(ks) >= 5:
                kt.append(np.median(ks))
        prov["k_recession_queue"] = round(float(np.median(kt)), 4) if kt else None
        prov["k_recession_n_stations"] = len(kt)
        con.close()
        out[reg] = prov
        print(f"[measure] {reg}: forcage={prov['forcage']} | debias_et={prov['debias_et']} ({len(used)} st) | k_queue={prov['k_recession_queue']}", flush=True)
    with open("reports/deploy_adapters.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out


def infer(adapters):
    from meandre.model import HydroModel
    from meandre.utils.state import HydroState
    from et_module import compute_demand
    DEVICE = "cuda"
    base = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
    rows = []
    for reg in REGIONS:
        ad = adapters[reg]
        _t = ad.get("forcage", "casr")
        os.environ["JOINT_FX_SUFFIX"] = "-none" if _t == "casr" else _t
        from importlib import reload
        import joint_data
        reload(joint_data)
        try:
            r = joint_data.load_region(reg, dict(base["loss"]), device=DEVICE)
        except Exception as ex:
            print(f"[infer] {reg} ECHEC chargement : {ex}", flush=True)
            continue
        td = r["train_data"]; n = r["n_nodes"]
        demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) * ad["debias_et"]
        f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
        times_all = pd.to_datetime(r["times"])
        _i0 = int(np.searchsorted(times_all, np.datetime64(P["spinup_debut"])))
        times = times_all[_i0:]
        sl = np.asarray((times >= P["heldout"][0]) & (times <= P["heldout"][1]))
        t0 = td.train_slice.start
        w = np.flatnonzero((r["times"] >= P["heldout"][0]) & (r["times"] <= P["heldout"][1]))
        qo = td.q_obs[w[0] - t0: w[-1] - t0 + 1].cpu().numpy()
        m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
                       use_temporal=False, use_residual=False, use_travel_time_attn=False,
                       use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                       column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                       use_latent_codes=False, latent_mode="additive", spatial_melt=True,
                       routing_mode="operator-lagged", predict_lake_params=True,
                       compile_soil=False, use_aquifer=True).to(DEVICE)
        m.load(CFG["champion"]["checkpoint"]); m.eval(); m.vertical_column.etp_channel = 6
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7[_i0:], initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year[_i0:])
        Qs = Q[torch.tensor(sl, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60:
                continue
            rr2 = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean() / o[v].mean()
            g = (si[v].std() / si[v].mean()) / (o[v].std() / o[v].mean())
            ks.append(1 - np.sqrt((rr2 - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))
        med = float(np.median(ks)) if ks else float("nan")
        rows.append(dict(region=reg, kge_med=round(med, 4), n=len(ks),
                         forcage=ad["forcage"], debias_et=ad["debias_et"]))
        print(f"[infer] {reg}: KGE {med:.3f} (n={len(ks)})", flush=True)
        del m, Q, f7, demand
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv("reports/deploy_map.csv", index=False)
    print("-> reports/deploy_map.csv + reports/deploy_adapters.json (provenance)")


if __name__ == "__main__":
    ad = measure()
    if "--measure" not in sys.argv:
        infer(ad)
