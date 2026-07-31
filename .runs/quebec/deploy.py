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
        # la couverture du forçage hybride se VÉRIFIE (les nœuds hors grille krigée
        # reçoivent une pluie tronquée), elle ne se suppose pas : on compare le volume
        # annuel hyb vs budyko et on rejette hyb s'il ampute (>5% de déficit).
        use_hyb = (north <= lat_max - P["forcage_marge_nord_deg"]) and os.path.exists(hyb)
        if use_hyb:
            bud = D["forcage_budyko"].format(reg=reg)
            if os.path.exists(bud):
                import xarray as _xr
                _a = _xr.open_dataset(hyb); _pa = float(_a["forcing"].values[:, :, 0].mean()); _a.close()
                _b = _xr.open_dataset(bud); _pb = float(_b["forcing"].values[:, :, 0].mean()); _b.close()
                if _pa < 0.95 * _pb:
                    use_hyb = False
                    prov["forcage_rejet_hyb"] = f"P_hyb={_pa*365.25:.0f} < 95% de P_budyko={_pb*365.25:.0f} mm/an
        prov["forcage"] = "hyb" if use_hyb else "budyko"
        prov["forcage_motif"] = f"q90_lat={north:.2f} vs grille_max={lat_max:.2f}" + ("" if os.path.exists(hyb) else " (hyb absent)")
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

        fx = hyb if use_hyb else D["forcage_budyko"].format(reg=reg)
        if not os.path.exists(fx):
            fx = f"D:/meandre-data/quebec/forcing-{reg}.nc"
        prov["forcage_fichier"] = os.path.basename(fx)
        if reg == "slso":
            fx = "D:/meandre-data/slso/forcing-casr-corr.nc"
        dsx = xr.open_dataset(fx); Pmm = dsx["forcing"].values[:, :, 0]
        times = pd.to_datetime(dsx["time"].values); dsx.close()
        et = None
        try:
            et = cache.load_modis_et("2001-01-01", "2024-12-31", device="cpu").numpy()
        except Exception:
            pass
        ratios, used = [], []
        min_d = P["debias_et_stations_min_annees"] * 365
        for _, rr in st.iterrows():
            o = con.execute(f"select date, discharge from observations where station_id={rr.station_id}").fetchdf()
            if len(o) < min_d or et is None:
                continue
            a = anc(int(rr.node_idx)); akm = area[a].sum()
            if akm < 100:
                continue
            q_mm = float(np.nanmean(o["discharge"])) * 86400 * 365.25 / (akm * 1e6) * 1000
            w = area[a] / area[a].sum()
            p_mm = float((Pmm[:, a] * w).sum(axis=1).mean()) * 365.25
            etn = et[:, a]; land = np.isfinite(etn).sum(axis=0) >= 200
            if land.sum() == 0:
                continue
            wl = area[a][land] / area[a][land].sum()
            et16 = float((np.nanmean(etn[:, land], axis=0) * 365.25 * wl).sum())
            if et16 > 50:
                ratios.append((p_mm - q_mm) / et16); used.append(int(rr.station_id))
        lo, hi = P["debias_et_bornes"]
        ds_val = float(np.clip(np.median(ratios), lo, hi)) if ratios else 1.0
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
        _f = ad.get("forcage_fichier", "")
        if "-hyb" in _f: _sfx = "-hyb"
        elif "-budyko" in _f: _sfx = "-budyko"
        else: _sfx = "-none"   # forcing-{reg}.nc : joint_data retombe dessus si absent
        os.environ["JOINT_FX_SUFFIX"] = _sfx
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
