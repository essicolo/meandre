"""Banc de SOUS-BASSIN : le palier qui manquait entre la colonne et la region.

POURQUOI (idee d'Essi, 2026-09-03). Deux echelles existaient et aucune ne convenait pour
tester une hypothese. La colonne isolee repond en trois minutes mais n'a ni reseau, ni
routage, ni station : elle ne peut rien dire d'un retard ni d'un hydrogramme. La region
entiere a tout mais coute 31 minutes par epoque sur la plus petite, mesure le 2026-09-03,
donc cinq heures pour dix epoques.

Entre les deux : un sous-bassin jauge de quelques dizaines de troncons. Inventaire du
depot : 59 sous-bassins de 25 a 130 troncons portent plus de 3000 jours d'observations,
et 115 en ont moins de 120 avec plus de 2000 jours. Un sous-bassin de 33 troncons a tout
ce qu'il faut -- reseau, lacs, routage, une station reelle -- pour un centieme du cout.

Le banc extrait le sous-bassin en amont d'une station, reindexe le graphe, decoupe le
forcage et le territorial, et fait tourner le modele COMPLET. Il rend le KGE sur la
periode d'evaluation, la repartition mensuelle contre l'observe, et la platitude.

  python .runs/quebec/banc_sousbassin.py liste gasp outv mont
  python .runs/quebec/banc_sousbassin.py gasp 021702
  python .runs/quebec/banc_sousbassin.py gasp 021702 --epoques 5
"""
import argparse
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from meandre.utils import paths as _p


def _amont(graph, depart):
    """Indices des noeuds en amont d'un noeud, lui compris."""
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()
    pere = defaultdict(list)
    for s, t in zip(src, dst):
        pere[int(t)].append(int(s))
    vus, q = {int(depart)}, deque([int(depart)])
    while q:
        u = q.popleft()
        for v in pere[u]:
            if v not in vus:
                vus.add(v)
                q.append(v)
    return np.array(sorted(vus))


def extraire(reg, station):
    """Sous-bassin en amont d'une station : graphe reindexe, territorial, coords, obs."""
    import duckdb
    from meandre.data.basin_cache import BasinCache
    from meandre.routing.graph import RiverGraph
    from meandre.spatial.territorial import TerritorialFeatures

    base = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
    d = BasinCache(base).load(device=torch.device("cpu"))
    g = d["graph"]
    con = duckdb.connect(base, read_only=True)
    ligne = con.execute("select node_idx, drainage_area_km2 from stations "
                        "where station_id = ?", [station]).fetchone()
    con.close()
    if ligne is None:
        raise SystemExit(f"{reg}: station {station} inconnue")
    n_exut, aire = int(ligne[0]), float(ligne[1] or 0.0)

    idx = _amont(g, n_exut)
    rang = {int(v): k for k, v in enumerate(idx)}
    src = g.edge_index[0].numpy()
    dst = g.edge_index[1].numpy()
    garde = np.array([(int(s) in rang) and (int(t) in rang) for s, t in zip(src, dst)])
    e_src = np.array([rang[int(s)] for s in src[garde]], dtype=np.int64)
    e_dst = np.array([rang[int(t)] for t in dst[garde]], dtype=np.int64)

    # Ordre topologique restreint, dans l'ordre d'origine.
    topo = np.array([rang[int(v)] for v in g.topo_order.numpy() if int(v) in rang],
                    dtype=np.int64)
    sous = RiverGraph(
        edge_index=torch.tensor(np.stack([e_src, e_dst])),
        edge_attr=g.edge_attr[torch.tensor(garde)],
        topo_order=torch.tensor(topo),
        is_lake=g.is_lake[torch.tensor(idx)],
        travel_time_days=g.travel_time_days[torch.tensor(garde)],
    )
    terr = d["territorial"]
    _ti = torch.tensor(idx)
    _n0 = terr.data.shape[0]
    sous_terr = TerritorialFeatures(
        data=terr.data[_ti],
        columns=list(terr.columns),
        physical={k: v[_ti] for k, v in terr.physical.items()
                  if torch.is_tensor(v) and v.shape[:1] == (_n0,)},
    )
    return dict(graph=sous, territorial=sous_terr,
                node_coords=d["node_coords"][torch.tensor(idx)],
                node_ids=[d["node_ids"][i] for i in idx],
                idx=idx, exutoire=rang[n_exut], aire=aire, base=base)


def lister(regions, n_min=25, n_max=130, jours_min=3000):
    import duckdb
    from meandre.data.basin_cache import BasinCache
    print(f"{'region':7s} {'station':9s} {'troncons':>9s} {'aire km2':>10s} {'jours obs':>10s}")
    out = []
    for reg in regions:
        base = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
        try:
            d = BasinCache(base).load(device=torch.device("cpu"))
        except Exception:
            continue
        con = duckdb.connect(base, read_only=True)
        st = con.execute("select station_id, node_idx, drainage_area_km2 "
                         "from stations").fetchall()
        ob = dict(con.execute("select station_id, count(*) from observations "
                              "where discharge is not null group by station_id").fetchall())
        con.close()
        for sid, nidx, aire in st:
            k = len(_amont(d["graph"], int(nidx)))
            nj = ob.get(sid, 0)
            if n_min <= k <= n_max and nj >= jours_min:
                out.append((k, reg, str(sid), aire or 0.0, nj))
    for k, reg, sid, aire, nj in sorted(out):
        print(f"{reg:7s} {sid:9s} {k:9d} {aire:10.0f} {nj:10d}")
    print(f"\n{len(out)} sous-bassins de {n_min} a {n_max} troncons avec au moins "
          f"{jours_min} jours d'observations")



def simuler(reg, station, annees=6, ancrer=True, kc=None, kmusk=None,
            melt_saison=None, seuil_neige=None, debut=None, sol=None,
            aquifere=True, charger=None, verbeux=True):
    """Modele COMPLET sur le sous-bassin : colonne, reseau, routage, une station reelle.

    Retourne (dates, debit simule a l'exutoire, debit observe). Aucun entrainement : le
    champ part de son initialisation de litterature et des ancrages de la plateforme,
    exactement comme une passe a zero epoque.
    """
    import os
    import duckdb
    import pandas as pd
    import xarray as xr
    from meandre.model import HydroModel
    from meandre.routing.withdrawals import WithdrawalData
    from meandre.utils.state import HydroState
    from meandre.data.hydrotel_calib import (load_linacre_nodes, load_melt_nodes,
                                             load_passage_pluie_neige)

    s = extraire(reg, station)
    idx, g, terr = s["idx"], s["graph"], s["territorial"]
    n = len(idx)

    _sfx = os.environ.get("JOINT_FX_SUFFIX", "-hyb")
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}{_sfx}.nc")
    temps = pd.DatetimeIndex(ds["time"].values)
    if debut is None:
        fin = temps.year < temps.year.min() + annees
    else:
        # Fenetre explicite : indispensable pour comparer a Hydrotel, dont le
        # post-traitement ne couvre que 2020 a 2026.
        fin = (temps.year >= int(debut)) & (temps.year < int(debut) + annees)
    F = torch.tensor(ds["forcing"].isel(node=idx).values[fin], dtype=torch.float32)
    ds.close()
    temps = temps[fin]
    doy = torch.tensor(temps.dayofyear.to_numpy(), dtype=torch.long)

    m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False,
                   use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   use_latent_codes=False, spatial_melt=True,
                   routing_mode="operator-lagged", predict_lake_params=True,
                   compile_soil=False, use_aquifer=aquifere)
    m.eval()
    if ancrer:
        maj = reg.upper()
        plat = f"{_p.PLATFORMS_ROOT}/LN24HA/{maj}_LN24HA_2020"
        col = m.vertical_column
        col.et_mode = "linacre"
        col.etp_channel = None
        col.set_linacre_params(*load_linacre_nodes(plat, s["node_ids"],
                                                   device=torch.device("cpu")))
        try:
            col.set_melt_params(load_melt_nodes(plat, s["node_ids"],
                                                device=torch.device("cpu")))
        except Exception as e:
            print(f"  [ancrage] fonte non chargee : {type(e).__name__}")
        sn = load_passage_pluie_neige(plat)
        if sn:
            col.t_neige_seuil = sn
        # OCCUPATION DU SOL. Sans elle la colonne traite tout le territoire en sol nu
        # decouvert, donc la classe de neige la plus fondante et aucun ruissellement sur
        # l'eau libre : elle l'avertit d'ailleurs. Aucun chiffre de volume ni de
        # calendrier n'est lisible sans elle (meme lecon que R72 sur l'ETP).
        from meandre.data.hydrotel_calib import load_occupation_sol
        try:
            col.set_land_cover(load_occupation_sol(plat, s["node_ids"],
                                                   device=torch.device("cpu")))
        except Exception as e:
            print(f"  [ancrage] occupation non chargee : {type(e).__name__} {e}")
    if sol is not None:
        # SOL D'HYDROTEL IMPOSE. sol="complet" : tout le calage bv3c, krec et recharge
        # compris, ce qui reproduit Hydrotel qui n'a pas de voie profonde (a coupler avec
        # aquifere=False). sol="sauf_ks" : tout sauf les conductivites a saturation, qui
        # restent au champ -- c'est la loi des ancrages.
        from meandre.data.hydrotel_calib import load_calibrated_soil
        maj = reg.upper()
        plat = f"{_p.PLATFORMS_ROOT}/LN24HA/{maj}_LN24HA_2020"
        z1 = float(getattr(m.vertical_column, "z1", 0.15))
        calib = load_calibrated_soil(plat, s["node_ids"], z1, device=torch.device("cpu"))
        if sol == "sauf_ks":
            for k in ("ks1", "ks2", "ks3"):
                calib.pop(k, None)
        m.vertical_column.set_calibrated_soil(calib)
    if melt_saison is not None:
        m.vertical_column.melt_seasonal_amp = float(melt_saison)
    if seuil_neige is not None:
        # Seuil de partage pluie/neige au bulbe humide, en degres. Plus il est HAUT,
        # plus la precipitation est comptee en neige, donc stockee au lieu de ruisseler.
        m.vertical_column.t_neige_seuil = float(seuil_neige)
    if kc is not None or kmusk is not None:
        _o = m.spatial_encoder.forward

        def _mod(*a, _o=_o, **kw):
            sp = _o(*a, **kw)
            if kc is not None:
                sp.K_c = torch.full_like(sp.K_c, float(kc))
            if kmusk is not None:
                sp.K_musk_hours = torch.full_like(sp.K_musk_hours, float(kmusk))
            return sp
        m.spatial_encoder.forward = _mod

    if charger:
        # Evaluer un point de reprise avec CE protocole (continu, independant du trainer).
        m.load(charger)
        m.eval()
    w = WithdrawalData(net=torch.zeros(F.shape[0], n))
    with torch.no_grad():
        Q, _ = m.simulate(forcing=F, initial_state=HydroState.zeros(n),
                          graph=g, node_coords=s["node_coords"], territorial=terr,
                          withdrawals=w, day_of_year=doy)
    q_sim = Q[:, s["exutoire"]].numpy()

    con = duckdb.connect(s["base"], read_only=True)
    obs = con.execute("select date, discharge from observations where station_id = ? "
                      "order by date", [station]).fetchdf()
    con.close()
    o = pd.Series(obs["discharge"].values,
                  index=pd.DatetimeIndex(obs["date"])).reindex(temps).to_numpy(dtype=float)
    return temps, q_sim, o



def forme(temps, q, o, garde=None):
    """Deux questions que gamma ne separe pas (Essi, 2026-09-04) :
      1. l'hydrogramme est-il RABOTE ? -> rapport des pointes annuelles et des hauts
         quantiles, simule sur observe ;
      2. l'hydrogramme est-il PLAT par moments ? -> part de jours ou le debit varie de
         moins de 1 % d'un jour a l'autre, en hiver et sur l'annee, et plus longue suite.
    Un gamma bas peut venir de l'un ou de l'autre ; ils ne se corrigent pas pareil.
    """
    import pandas as pd
    if garde is None:
        garde = np.ones(len(temps), bool)
    an = temps.year.to_numpy()
    mois = temps.month.to_numpy()
    m = garde & np.isfinite(q) & np.isfinite(o)
    # 1. rabotage
    pics = []
    for a in np.unique(an[m]):
        sel = m & (an == a)
        if sel.sum() > 300 and np.nanmax(o[sel]) > 0:
            pics.append(np.nanmax(q[sel]) / np.nanmax(o[sel]))
    q95 = np.nanquantile(q[m], 0.95) / max(np.nanquantile(o[m], 0.95), 1e-9)
    q99 = np.nanquantile(q[m], 0.99) / max(np.nanquantile(o[m], 0.99), 1e-9)
    # 2. platitude
    def _plat(x):
        r = np.abs(np.diff(x)) / np.maximum(x[:-1], 1e-9)
        return r < 0.01
    mm = m[:-1] & m[1:]
    ps, po = _plat(q), _plat(o)
    hiv = np.isin(mois[:-1], (12, 1, 2, 3)) & mm
    n_ = mx = 0
    for c in ps[mm]:
        n_ = n_ + 1 if c else 0
        mx = max(mx, n_)
    return dict(pic=float(np.median(pics)) if pics else float("nan"), q95=float(q95),
                q99=float(q99), plat=100 * float(ps[mm].mean()),
                plat_obs=100 * float(po[mm].mean()),
                plat_hiv=100 * float(ps[hiv].mean()) if hiv.any() else float("nan"),
                plat_hiv_obs=100 * float(po[hiv].mean()) if hiv.any() else float("nan"),
                suite=int(mx))


def _ligne_forme(etiquette, f):
    print(f"  {etiquette:28s} pointes annuelles sim/obs {f['pic']:5.2f} | q95 {f['q95']:5.2f} "
          f"| q99 {f['q99']:5.2f} || plat {f['plat']:4.1f}% (obs {f['plat_obs']:4.1f}%) "
          f"| hiver {f['plat_hiv']:4.1f}% (obs {f['plat_hiv_obs']:4.1f}%) "
          f"| plus longue suite {f['suite']:3d} j", flush=True)


def _kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 100:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sim, obs = sim[m], obs[m]
    r = float(np.corrcoef(sim, obs)[0, 1])
    beta = float(sim.mean() / obs.mean())
    gamma = float((sim.std() / sim.mean()) / (obs.std() / obs.mean()))
    return 1 - float(np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)), r, beta, gamma


def rapport(reg, station, **kw):
    import pandas as pd
    temps, q, o = simuler(reg, station, **kw)
    garde = temps.year > temps.year.min()      # premiere annee = mise en regime
    k, r, b, gm = _kge(q[garde], o[garde])
    print(f"{reg.upper()} / {station} : KGE {k:.3f} | r {r:.3f} | beta {b:.3f} | "
          f"gamma {gm:.3f}")
    _ligne_forme("forme", forme(temps, q, o, garde))
    mo = temps.month.to_numpy()
    ps, po = [], []
    for mm in range(1, 13):
        sel = garde & (mo == mm)
        ps.append(np.nansum(q[sel])); po.append(np.nansum(o[sel]))
    ps = 100 * np.array(ps) / max(np.sum(ps), 1e-9)
    po = 100 * np.array(po) / max(np.sum(po), 1e-9)
    print("  parts mensuelles, simule contre observe :")
    print("   " + " ".join(f"{x:5.1f}" for x in ps))
    print("   " + " ".join(f"{x:5.1f}" for x in po))
    hiv_s, hiv_o = ps[[11, 0, 1, 2]].sum(), po[[11, 0, 1, 2]].sum()
    print(f"  hiver : simule {hiv_s:.1f} % contre observe {hiv_o:.1f} % "
          f"({hiv_s - hiv_o:+.1f} points)")
    return k



def entrainer(reg, station, epoques=20, lr=5e-4, sol="sauf_ks", aquifere=True,
              debut_train=2010, fin_train=2017, fin_val=2019, debut_eval=2020,
              kge_continu=True, etat_continu=True, device=None, tag="",
              pas_par_bloc=True, amorce=False, aux=True, fin_charge=None,
              substeps=None, chunk=45, w_et=0.4):
    """LE TEST QUI DECIDE : un champ entraine sous une boucle JUSTE rend-il les
    hydrogrammes plus nets ou plus plats ?

    Point de depart : la recette du socle (tout le sol d'Hydrotel impose sauf K_sat,
    qui reste au champ). Au zero epoque, la variabilite vaut 0.795 sur ce sous-bassin
    contre 0.863 pour Hydrotel. Si l'entrainement de K_sat sous la boucle corrigee fait
    MONTER gamma au-dessus d'Hydrotel, le champ bat Hydrotel sur ce qui compte. S'il le
    fait DESCENDRE, la reponse est definitive dans l'autre sens.

    Protocole : entrainement 2010-2017, validation 2018-2019, evaluation 2020-2024
    (la periode couverte par les sorties d'Hydrotel, pour un duel a trois).
    """
    import os
    import duckdb
    import pandas as pd
    import xarray as xr
    from meandre.model import HydroModel
    from meandre.routing.withdrawals import WithdrawalData
    from meandre.utils.state import HydroState
    from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
    from meandre.training.loss import HydroLoss
    from meandre.data.hydrotel_calib import (load_linacre_nodes, load_melt_nodes,
                                             load_passage_pluie_neige,
                                             load_occupation_sol, load_calibrated_soil)

    os.environ["MEANDRE_KGE_CONTINU"] = "1" if kge_continu else "0"
    os.environ["MEANDRE_ETAT_CONTINU"] = "1" if etat_continu else "0"
    os.environ["MEANDRE_PAS_PAR_BLOC"] = "1" if pas_par_bloc else "0"
    os.environ["MEANDRE_HISTORIQUE_AMORCE"] = "1" if amorce else "0"
    if substeps is not None:
        # ESSAI DE MECANISME (Essi, 2026-09-04 : « pourquoi pas des essais de trente
        # secondes »). Seize sous-pas au lieu de soixante-quatre : la physique n'est plus
        # celle de la recette, mais la question posee ici est « l'optimiseur ameliore-t-il
        # l'objectif, et dans quel sens », pas la fidelite au dixieme.
        os.environ["MEANDRE_NSUBSTEP"] = str(int(substeps))
    # GPU (2026-09-04, les epoques etaient extremement trop longues). Le banc tournait
    # sur processeur : 13 min par epoque pour 33 troncons. Tous les chargeurs d'ancrage
    # acceptent un device ; modele, donnees et ancrages y vont ensemble, sinon un seul
    # tenseur reste sur l'autre carte et tout s'arrete.
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # GRAINE FIXE (2026-09-04). Le champ spatial est initialise au hasard : trois zero
    # epoque successifs ont donne 0.811, 0.817 et 0.850 de gamma pour la meme
    # configuration, soit l'ordre de grandeur de l'ecart a Hydrotel. Sans graine, un
    # verdict absolu n'a pas de sens ; on lit un ecart APPARIE dans un meme run, et on
    # rend la reference reproductible.
    torch.manual_seed(int(os.environ.get("ETL_SEED", "1234")))
    np.random.seed(int(os.environ.get("ETL_SEED", "1234")))

    s = extraire(reg, station)
    idx, g, terr = s["idx"], s["graph"], s["territorial"]
    n = len(idx)
    _sfx = os.environ.get("JOINT_FX_SUFFIX", "-hyb")
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}{_sfx}.nc")
    temps = pd.DatetimeIndex(ds["time"].values)
    # MISE EN REGIME. Le trainer ne spinne que sur les jours qui PRECEDENT le debut de
    # la tranche d'entrainement (spinup_end = min(730, train_slice.start)). Si le
    # forcage commence le jour meme, il n'y a AUCUN spinup : chaque epoque repart d'un
    # manteau vide un premier janvier, le premier bloc rend des gradients NaN, et le
    # premier hiver de chaque epoque est faux -- la faute de R64 sous une autre forme.
    # On charge donc deux annees de plus en amont, reservees a la mise en regime.
    fen = (temps.year >= debut_train - 2)
    if fin_charge is not None:
        fen = fen & (temps.year <= int(fin_charge))
    F = torch.tensor(ds["forcing"].isel(node=idx).values[fen], dtype=torch.float32).to(dev)
    ds.close()
    temps = temps[fen]
    doy = torch.tensor(temps.dayofyear.to_numpy(), dtype=torch.long).to(dev)

    con = duckdb.connect(s["base"], read_only=True)
    obs = con.execute("select date, discharge from observations where station_id = ? "
                      "order by date", [station]).fetchdf()
    con.close()
    o = pd.Series(obs["discharge"].values,
                  index=pd.DatetimeIndex(obs["date"])).reindex(temps).to_numpy(dtype=float)
    q_obs = torch.tensor(o, dtype=torch.float32)[:, None].to(dev)
    # CIBLES AUXILIAIRES (objection d'Essi, 2026-09-04). La recette du socle ne
    # s'entraine PAS sur le KGE seul : elle pese l'evapotranspiration MOD16 a 0.4, GRACE a
    # 0.2 et 0.05, le biais de volume a 0.5 et une MSE a 0.1, pour rendre le modele
    # IDENTIFIABLE. Un banc au KGE seul mesure la compensation non identifiable que ces
    # termes existent pour empecher. MOD16 est charge ici pour les noeuds du sous-bassin ;
    # GRACE ne l'est pas : son empreinte (un mascon de ~300 km) n'a aucun sens pour un
    # sous-bassin de 200 km2, et le socle le dit lui-meme pour les regions sans cible.
    et_obs = None
    if aux:
        from meandre.data.basin_cache import BasinCache as _BC
        _et = _BC(s["base"]).load_modis_et(str(temps[0].date()), str(temps[-1].date()),
                                          device=torch.device("cpu"))
        if _et is not None:
            _et = _et[:, torch.tensor(idx)]
            if _et.shape[0] == len(temps):
                et_obs = _et.to(torch.float32).to(dev)
                print(f"  MOD16 : {int(torch.isfinite(et_obs).sum())} valeurs finies sur "
                      f"{et_obs.numel()} (ET 8 jours, mm/j)", flush=True)
            else:
                print(f"  MOD16 ignore : {_et.shape[0]} pas de temps contre {len(temps)}")
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    mask[s["exutoire"]] = True
    st_idx = torch.tensor([s["exutoire"]], device=dev)
    g = g.to(dev)
    from meandre.spatial.territorial import TerritorialFeatures as _TF
    terr = _TF(data=terr.data.to(dev), columns=list(terr.columns),
               physical={k: v.to(dev) for k, v in terr.physical.items()})
    coords = s["node_coords"].to(dev)

    def _construire():
        m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6,
                       use_temporal=False, use_residual=False, use_travel_time_attn=False,
                       use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                       column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                       use_latent_codes=False, spatial_melt=True,
                       routing_mode="operator-lagged", predict_lake_params=True,
                       compile_soil=False, use_aquifer=aquifere)
        maj = reg.upper()
        plat = f"{_p.PLATFORMS_ROOT}/LN24HA/{maj}_LN24HA_2020"
        m = m.to(dev)
        col = m.vertical_column
        col.et_mode = "linacre"
        col.etp_channel = None
        col.set_linacre_params(*load_linacre_nodes(plat, s["node_ids"], device=dev))
        col.set_melt_params(load_melt_nodes(plat, s["node_ids"], device=dev))
        sn = load_passage_pluie_neige(plat)
        if sn:
            col.t_neige_seuil = sn
        col.set_land_cover(load_occupation_sol(plat, s["node_ids"], device=dev))
        if sol:
            z1 = float(getattr(col, "z1", 0.15))
            calib = load_calibrated_soil(plat, s["node_ids"], z1, device=dev)
            if sol == "sauf_ks":
                for k in ("ks1", "ks2", "ks3"):
                    calib.pop(k, None)
            col.set_calibrated_soil(calib)
        return m

    def _tranche(a, b):
        i0 = int(np.argmax(temps.year >= a))
        i1 = int(np.argmax(temps.year > b))
        return slice(i0, i1 if i1 > 0 else len(temps))

    w = WithdrawalData(net=torch.zeros(F.shape[0], n, device=dev))
    commun = dict(forcing=F, q_obs=q_obs, station_mask=mask, station_idx=st_idx, graph=g,
                  node_coords=coords, territorial=terr, withdrawals=w,
                  day_of_year=doy)
    # CONVENTION DU TRAINER, payee le 2026-09-04 : q_obs[0] correspond a
    # forcing[train_slice.start], PAS a forcing[0]. Le forcage porte la mise en regime
    # en amont, les observations commencent au premier jour juge. Pour la validation,
    # q_obs[0] correspond a forcing[val_slice.start]. En passant q_obs complet, la
    # boucle comparait la simulation de 2010 aux observations de 2008, et la validation
    # celle de 2018 a celles de 2008 : le trainer notait 0.399 puis 0.560 un modele qui
    # vaut 0.82. J'avais attribue cet ecart au trainer (R75) ; il etait dans ce banc.
    tr_sl = _tranche(debut_train, fin_train)
    va_sl = _tranche(fin_train + 1, fin_val)
    td = TrainingData(train_slice=tr_sl, val_slice=va_sl,
                      **{**commun, "q_obs": q_obs[tr_sl.start:],
                         "et_obs": (et_obs[tr_sl.start:] if et_obs is not None else None)})
    vd = TrainingData(train_slice=va_sl, val_slice=va_sl,
                      **{**commun, "q_obs": q_obs[va_sl.start:],
                         "et_obs": (et_obs[va_sl.start:] if et_obs is not None else None)})

    def _evaluer(m, etiquette):
        m.eval()
        with torch.no_grad():
            Q, _, _d = m.simulate(forcing=F, initial_state=HydroState.zeros(n, device=dev),
                                  graph=g, node_coords=coords, territorial=terr,
                                  withdrawals=w, day_of_year=doy, return_diagnostics=True)
        q = Q[:, s["exutoire"]].detach().cpu().numpy()
        ev = (temps.year >= debut_eval)
        # Bilan d'evapotranspiration simulee contre MOD16, la ou MOD16 est fini : dit si
        # le terme d'ET tire le volume vers le bas (ET simulee < MOD16) ou vers le haut.
        _etr = getattr(_d, "etr", None)
        if et_obs is not None and _etr is not None and _etr.shape[0] == len(temps):
            _fin = torch.isfinite(et_obs) & torch.tensor(ev, device=et_obs.device)[:, None]
            if bool(_fin.any()):
                _s = float(_etr.to(et_obs.device)[_fin].mean())
                _o = float(et_obs[_fin].mean())
                print(f"  {etiquette:28s} ET simulee {_s:5.2f} mm/j contre MOD16 {_o:5.2f} mm/j "
                      f"({100 * (_s / max(_o, 1e-9) - 1):+.0f} %) sur les jours MOD16 finis "
                      f"de la periode d'evaluation", flush=True)
        k, r, b, gm = _kge(q[ev], o[ev])
        print(f"  {etiquette:28s} KGE vs observe {k:6.3f} | r {r:5.3f} | beta {b:5.3f} "
              f"| gamma {gm:5.3f}", flush=True)
        _ligne_forme(etiquette, forme(temps, q, o, ev))
        # Le meme KGE sur la fenetre d'ENTRAINEMENT : a comparer au terme KGE de la
        # perte (1 - KGE), qui semblait deja presque nul a la premiere epoque.
        tr = (temps.year >= debut_train) & (temps.year <= fin_train)
        if tr.any() and np.isfinite(o[tr]).sum() > 100:
            k2, r2, b2, g2 = _kge(q[tr], o[tr])
            print(f"  {etiquette:28s} KGE entrainement {debut_train}-{fin_train} {k2:6.3f} "
                  f"| r {r2:5.3f} | beta {b2:5.3f} | gamma {g2:5.3f}", flush=True)
        return q, ev

    oui_non = lambda v: "oui" if v else "NON"
    print(f"{reg.upper()} / {station} : {n} troncons | entrainement {debut_train}-{fin_train}"
          f" | validation {fin_train+1}-{fin_val} | evaluation {debut_eval}-{temps.year.max()}")
    print(f"  boucle : etat continu={oui_non(etat_continu)}, "
          f"KGE continu={oui_non(kge_continu)} | sol={sol} | aquifere={aquifere} "
          f"| device={dev} | pas par bloc={oui_non(pas_par_bloc)} "
          f"| historique amorce={oui_non(amorce)}", flush=True)
    print(f"  mise en regime : {td.train_slice.start} jours avant {debut_train} "
          f"(le trainer en spinne au plus 730)", flush=True)
    m = _construire()
    q0, ev = _evaluer(m, "zero epoque")

    if aux and et_obs is not None:
        # Poids de la recette du socle (gasp-v4 [loss] + ETL_WET=0.4), sans GRACE.
        # per_station=True est OBLIGATOIRE (trouve le 2026-09-04 a 14 h 35) : le defaut
        # de HydroLoss est la branche « pooled », qui calcule le KGE sur le seul bloc
        # courant SANS l'historique detache. Tous les essais du banc jusqu'ici ont donc
        # optimise un KGE de quinze ou quarante-cinq jours, la faute meme que R67
        # corrige ; slso.py passe per_station=True, le banc ne le faisait pas.
        loss_fn = HydroLoss(w_kge=1.0, w_pbias=0.5, w_mse=0.1, w_nse=0.0, w_nrmse=0.0,
                            w_log_nse=0.0, w_log_mse=0.0, w_et=float(w_et), per_station=True,
                            # TENDANCE, pas niveau (R24, socle.toml et_mode = "anomaly") :
                            # MOD16 donne la forme de l'ET, jamais son volume. Le banc
                            # laissait le defaut « level » jusqu'a 15 h 30 le 2026-09-04,
                            # et un essai a conclu a tort que MOD16 vidait la riviere.
                            et_mode="anomaly")
        print(f"  perte : KGE 1.0 + biais 0.5 + MSE 0.1 + ET MOD16 {float(w_et):.2f} en tendance "
              "(recette du socle, sans GRACE)", flush=True)
    else:
        loss_fn = HydroLoss(w_kge=1.0, w_pbias=0.0, w_nse=0.0, w_mse=0.0, w_nrmse=0.0,
                            w_log_nse=0.0, w_log_mse=0.0, per_station=True)
        print("  perte : KGE seul (PAS la recette du socle)", flush=True)
    # warmup_epochs=0 : le defaut de cinq epoques de rechauffement rendait un essai
    # court entierement nul (cinq pas d'Adam a taux presque nul).
    tconf = TrainingConfig(n_epochs=epoques, lr=lr, chunk_steps=int(chunk), tbptt_steps=365,
                           grad_clip=1.0, w_prior=0.005, w_latent_reg=0.0,
                           best_metric="kge_median", autopilot=False, warmup_epochs=0)
    ck = f"{_p.DATA_ROOT}/quebec/sousbassin/best-{reg}-{station}{tag}.pt"
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    tr = Trainer(model=m, loss_fn=loss_fn, train_data=td, val_data=vd, config=tconf,
                 run_name=f"sb-{reg}-{station}{tag}", checkpoint_path=ck)
    tr.fit()
    if os.path.exists(ck):
        m.load(ck)
    q1, _ = _evaluer(m, f"apres {epoques} epoques")

    try:
        rqh = os.environ.get("MEANDRE_RQH", "D:/rqh")
        z = xr.open_zarr(f"{rqh}/rqh_2026-04/data/06_posttraitement/posttraitement_LN24HA.zarr")
        ids = z["troncon_id"].values.astype(str)
        tid = f"{reg.upper()}{int(s['node_ids'][s['exutoire']]):05d}"
        wq = np.flatnonzero(ids == tid)
        if len(wq):
            h = pd.Series(z["Dis"].values[int(wq[0]), :],
                          index=pd.to_datetime(z["time"].values)).reindex(temps).to_numpy(float)
            k, r, b, gm = _kge(h[ev], o[ev])
            print(f"  {'Hydrotel':28s} KGE vs observe {k:6.3f} | r {r:5.3f} | beta {b:5.3f} "
                  f"| gamma {gm:5.3f}")
            _ligne_forme("Hydrotel", forme(temps, h, o, ev))
            for etiq, q in (("zero epoque", q0), (f"apres {epoques} epoques", q1)):
                k, r, b, gm = _kge(q[ev], h[ev])
                print(f"  {etiq + ' vs Hydrotel':28s} KGE {k:6.3f} | r {r:5.3f}")
    except Exception as e:
        print(f"  [hydrotel] indisponible : {type(e).__name__} {e}")
    return q0, q1, o, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("station", nargs="?")
    ap.add_argument("--simuler", action="store_true")
    ap.add_argument("--charger", default=None,
                    help="point de reprise (.pt) a charger avant de simuler : compare un modele entraine au zero epoque")
    ap.add_argument("--sans-ancrage", action="store_true")
    ap.add_argument("--kc", type=float, default=None)
    ap.add_argument("--kmusk", type=float, default=None)
    ap.add_argument("--annees", type=int, default=6)
    ap.add_argument("--fonte", type=float, default=None)
    ap.add_argument("--seuil", type=float, default=None)
    ap.add_argument("--debut", type=int, default=None)
    ap.add_argument("--sol", choices=["complet", "sauf_ks"], default=None)
    ap.add_argument("--sans-aquifere", action="store_true")
    ap.add_argument("--entrainer", type=int, default=0, metavar="EPOQUES")
    ap.add_argument("--device", default=None, help="cuda ou cpu (defaut : cuda si dispo)")
    ap.add_argument("--tag", default="", help="suffixe du point de reprise et du run")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--w-et", type=float, default=0.4,
                    help="poids du terme MOD16 (0 = meme perte sans le terme d ET)")
    ap.add_argument("--chunk", type=int, default=45,
                    help="longueur des blocs en jours : plus court = plus de pas par epoque")
    ap.add_argument("--rapide", action="store_true",
                    help="essai de mecanisme en ~30 s par epoque : un an d entrainement, un an de validation, 16 sous-pas")
    ap.add_argument("--kge-seul", action="store_true",
                    help="perte au KGE seul, sans MOD16 : PAS la recette du socle")
    ap.add_argument("--amorce", action="store_true",
                    help="passe sans gradient en debut d epoque : chaque bloc voit le KGE de toute la periode")
    ap.add_argument("--pas-par-epoque", action="store_true",
                    help="un seul pas par epoque, comme avant le 2026-09-04")
    ap.add_argument("--ancienne-boucle", action="store_true",
                    help="etat et KGE NON continus, comme avant le 2026-09-03")
    ap.add_argument("--regions", nargs="*", default=["gasp", "outv", "mont", "sagu",
                                                     "slso", "slno", "cnde", "abit"])
    a = ap.parse_args()
    if a.region == "liste":
        lister(a.regions)
        return
    if not a.station:
        raise SystemExit("usage : banc_sousbassin.py <region> <station>")
    if a.entrainer and a.rapide:
        entrainer(a.region, a.station, epoques=a.entrainer,
                  sol=a.sol or "sauf_ks", aquifere=not a.sans_aquifere,
                  kge_continu=not a.ancienne_boucle, etat_continu=not a.ancienne_boucle,
                  device=a.device, tag=a.tag, pas_par_bloc=not a.pas_par_epoque, lr=a.lr,
                  amorce=a.amorce, aux=not a.kge_seul,
                  debut_train=2012, fin_train=2012, fin_val=2013, debut_eval=2013,
                  fin_charge=2013, substeps=16, chunk=a.chunk, w_et=a.w_et)
        return
    if a.entrainer:
        entrainer(a.region, a.station, epoques=a.entrainer,
                  sol=a.sol or "sauf_ks", aquifere=not a.sans_aquifere,
                  kge_continu=not a.ancienne_boucle, etat_continu=not a.ancienne_boucle,
                  device=a.device, tag=a.tag, pas_par_bloc=not a.pas_par_epoque, lr=a.lr,
                  amorce=a.amorce, aux=not a.kge_seul)
        return
    if a.simuler:
        rapport(a.region, a.station, ancrer=not a.sans_ancrage,
                kc=a.kc, kmusk=a.kmusk, annees=a.annees, melt_saison=a.fonte,
                seuil_neige=a.seuil, debut=a.debut, sol=a.sol,
                aquifere=not a.sans_aquifere, charger=a.charger)
        return
    s = extraire(a.region, a.station)
    g = s["graph"]
    print(f"{a.region.upper()} / station {a.station}")
    print(f"  troncons          {g.is_lake.shape[0]}")
    print(f"  liens             {g.edge_index.shape[1]}")
    print(f"  lacs              {int(g.is_lake.sum())}")
    print(f"  aire declaree     {s['aire']:.0f} km2")
    print(f"  exutoire (indice) {s['exutoire']}")


if __name__ == "__main__":
    main()
