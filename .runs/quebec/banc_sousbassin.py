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
            aquifere=True, charger=None, verbeux=True, melt_diurne=False):
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

    # GRAINE FIXE, comme dans `entrainer`. Elle manquait ici jusqu'au 2026-09-13, si bien
    # que deux passes de la meme configuration differaient d'autant qu'un traitement : le
    # champ spatial est tire au hasard et trois zero epoque successifs avaient donne
    # 0.811, 0.817 et 0.850 de variabilite. Toute comparaison faite par ce chemin sans
    # graine mesurait ce tirage. La variable ETL_SEED permet de balayer plusieurs tirages
    # pour situer un effet par rapport a cette dispersion.
    torch.manual_seed(int(os.environ.get("ETL_SEED", "1234")))
    np.random.seed(int(os.environ.get("ETL_SEED", "1234")))

    s = extraire(reg, station)
    idx, g, terr = s["idx"], s["graph"], s["territorial"]
    n = len(idx)

    # LE FORCAGE N'EST PAS CELUI DE LA RECETTE PAR DEFAUT. Le socle emploie -budyko ;
    # ce banc part de -hyb, herite de son premier usage. L'ecart n'est pas anodin : sur ce
    # sous-bassin, mesure le 2026-09-13, le KGE a zero epoque vaut 0,782 sous -budyko et
    # 0,644 sous -hyb, soit sept fois la dispersion due au tirage du champ. Le forcage est
    # donc annonce a chaque passe, faute de quoi une comparaison se fait sans le savoir
    # entre deux modeles differents.
    _sfx = os.environ.get("JOINT_FX_SUFFIX", "-hyb")
    if verbeux:
        print(f"  [forcage] forcing-{reg}{_sfx}.nc"
              + ("" if _sfx == "-budyko" else "   ATTENTION : la recette emploie -budyko"))
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
        # MEANDRE_BANC_ETP_MODE (2026-09-28) : formule d'ETP autre que Linacre calée, par
        # exemple penman, dont la forme saisonnière suit MOD16 (avril / octobre 1,58 contre 1,63).
        col.et_mode = os.environ.get("MEANDRE_BANC_ETP_MODE", "linacre")
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
    if melt_diurne:
        m.vertical_column.melt_diurnal = True
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
        Q, _, _dg = m.simulate(forcing=F, initial_state=HydroState.zeros(n),
                               graph=g, node_coords=s["node_coords"], territorial=terr,
                               withdrawals=w, day_of_year=doy, return_diagnostics=True)
    q_sim = Q[:, s["exutoire"]].numpy()
    # PARTITION DES CHEMINS (2026-09-05). La question ouverte de R83 et R85 : pendant un
    # plateau d'ete, quel chemin porte le debit ? Surface, hypodermique et nappe sont
    # moyennes sur les noeuds du sous-bassin, en millimetres par jour.
    _part = {}
    for _k in ("prod_surf", "prod_hypo", "prod_base"):
        _v = getattr(_dg, _k, None)
        if _v is None and isinstance(_dg, dict):
            _v = _dg.get(_k)
        if _v is not None:
            _part[_k] = _v.detach().cpu().numpy().mean(axis=1)
    globals()["_DERNIERE_PARTITION"] = _part

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
    ete = np.isin(mois[:-1], (6, 7, 8, 9)) & mm
    n_ = mx = 0
    for c in ps[mm]:
        n_ = n_ + 1 if c else 0
        mx = max(mx, n_)
    return dict(pic=float(np.median(pics)) if pics else float("nan"), q95=float(q95),
                q99=float(q99), plat=100 * float(ps[mm].mean()),
                plat_obs=100 * float(po[mm].mean()),
                plat_hiv=100 * float(ps[hiv].mean()) if hiv.any() else float("nan"),
                plat_hiv_obs=100 * float(po[hiv].mean()) if hiv.any() else float("nan"),
                plat_ete=100 * float(ps[ete].mean()) if ete.any() else float("nan"),
                plat_ete_obs=100 * float(po[ete].mean()) if ete.any() else float("nan"),
                suite=int(mx))


def _ligne_forme(etiquette, f):
    print(f"  {etiquette:28s} pointes annuelles sim/obs {f['pic']:5.2f} | q95 {f['q95']:5.2f} "
          f"| q99 {f['q99']:5.2f} || plat {f['plat']:4.1f}% (obs {f['plat_obs']:4.1f}%) "
          f"| hiver {f['plat_hiv']:4.1f}% (obs {f['plat_hiv_obs']:4.1f}%) "
          f"| ete {f['plat_ete']:4.1f}% (obs {f['plat_ete_obs']:4.1f}%) "
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
              substeps=None, chunk=45, w_et=0.4, w_kge=1.0, w_pbias=0.5, w_mse=0.1,
              w_dq=0.0, w_fdc=0.0, w_dq_log=0.0, quantile=False, charger=None,
              w_log_mse=0.0, w_peak=0.0,
              w_nappe=0.0, nappe_valid=0.5):
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
    # Drapeau de reconstruction du Centre d'expertise hydrique : sous la glace, presque tout
    # janvier et fevrier est reconstruit. Un jour sans drapeau n'est pas compte comme mesure.
    try:
        _fl = con.execute("select date, reconstructed from observations where station_id = ? "
                          "and reconstructed is not null", [station]).fetchdf()
        mesure = (pd.Series(_fl["reconstructed"].values, index=pd.DatetimeIndex(_fl["date"])).reindex(temps) == False).to_numpy()  # noqa: E712
    except Exception:
        mesure = None
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
    if os.environ.get("MEANDRE_BANC_SANS_LACS") == "1":
        # Diagnostic : les lacs deviennent des troncons ordinaires. Dit si l'etalement d'un
        # bassin a lacs vient de leur routage (pseudo-lacs importes en reservoirs actifs).
        print(f"  lacs neutralises : {int(g.is_lake.sum())} noeuds routes comme des rivieres", flush=True)
        g.is_lake = torch.zeros_like(g.is_lake)
    from meandre.spatial.territorial import TerritorialFeatures as _TF
    terr = _TF(data=terr.data.to(dev), columns=list(terr.columns),
               physical={k: v.to(dev) for k, v in terr.physical.items()})
    coords = s["node_coords"].to(dev)

    # Forme du champ spatial, pour l'epreuve NeRF contre perceptron (2026-09-25) :
    # MEANDRE_CHAMP_FREQS, bandes de Fourier sur la position, -1 pour AUCUNE position ;
    # MEANDRE_CHAMP_MODE, "nerf" ou "static" (un seul jeu de parametres pour tout le domaine).
    _freqs = int(os.environ.get("MEANDRE_CHAMP_FREQS", "6"))
    _mode = os.environ.get("MEANDRE_CHAMP_MODE", "nerf")
    print(f"  champ spatial : mode {_mode}, {'aucune position' if _freqs < 0 else f'{_freqs} bandes de Fourier'}", flush=True)

    # Bornes du champ : celles du TOML du pilote quand ETL_CONFIG est pose (2026-10-01).
    _field_bounds = {}
    if os.environ.get("ETL_CONFIG"):
        import tomllib
        from meandre.spatial.field_network import field_bounds_from_toml
        with open(os.environ["ETL_CONFIG"], "rb") as _fcfg:
            _field_bounds = field_bounds_from_toml(tomllib.load(_fcfg))
        if _field_bounds:
            print(f"  bornes du champ prises du TOML : {_field_bounds}", flush=True)

    def _construire():
        m = HydroModel(field_bounds=_field_bounds, n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6,
                       use_temporal=False, use_residual=False, use_travel_time_attn=False,
                       use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode=_mode, n_coord_freqs=_freqs,
                       column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                       use_latent_codes=False, spatial_melt=True,
                       routing_mode="operator-lagged", predict_lake_params=True,
                       # SOUS-PAS DU ROUTAGE (2026-09-30) : a deux sous-pas de 12 h, tout temps de
                       # transfert sous 7,5 h annule le stockage de Muskingum (c2 borne a 0) ; la
                       # borne basse de 4 h etait donc sans effet. MEANDRE_ROUTAGE_SOUSPAS l'affine.
                       routing_substeps=int(os.environ.get("MEANDRE_ROUTAGE_SOUSPAS", "2")),
                       compile_soil=False, use_aquifer=aquifere,
                       use_phenology_modulator=os.environ.get("MEANDRE_PHENOLOGIE", "0") == "1")
        maj = reg.upper()
        plat = f"{_p.PLATFORMS_ROOT}/LN24HA/{maj}_LN24HA_2020"
        m = m.to(dev)
        # SURFACE ET ANCRE DES LACS (2026-10-02), comme le pilote : sans surface posee, le
        # routage rapporte le stock du lac a l'aire DRAINEE du noeud et la loi de vidange
        # retient la fonte au printemps pour la rendre l'ete (passe sans lacs : avril 5,42 ->
        # 6,40 mm/j pour 6,20 observe). Surface : HydroLAKES, sinon fraction de lac x aire
        # locale ; ancre d'exutoire k0 (A_ref / A)^alpha. MEANDRE_BANC_LACS_SURFACE=0 restitue
        # l'ancien banc.
        if os.environ.get("MEANDRE_BANC_LACS_SURFACE", "1") == "1":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import pandas as _pdl
            from recipe import RAW_QC as _RAWQC, HYDROLAKES as _HL
            _idx = np.asarray(s["idx"])
            _rang = {int(v): k for k, v in enumerate(_idx)}
            _A = terr.get_physical("area_km2_local").cpu().numpy()
            _rw = _pdl.read_parquet(_RAWQC)
            _rw = _rw[_rw.region == reg].reset_index(drop=True)
            _alac = _A * _rw["lake_fraction"].values[_idx].clip(0, 1)
            _src = "fraction de lac x aire locale"
            try:
                _hl = _pdl.read_parquet(_HL)
                _hl = _hl[(_hl.region == reg) & _hl.node_idx.isin(_rang)]
                _alac[[_rang[int(v)] for v in _hl.node_idx.values]] = _hl["lake_area_km2"].values
                _src = f"HydroLAKES ({len(_hl)} noeuds) + repli"
            except Exception as _el:
                _src += f" (HydroLAKES illisible : {type(_el).__name__})"
            m.set_lake_area(torch.tensor(_alac, dtype=torch.float32))
            m.spatial_encoder.set_lake_anchor(torch.tensor(_alac, dtype=torch.float32),
                                              a_ref_km2=float(os.environ.get("ETL_LAKE_AREF", "20")),
                                              alpha=float(os.environ.get("ETL_LAKE_ALPHA", "1.0")))
            _isl = g.is_lake.bool().cpu().numpy()
            if _isl.any():
                print(f"  lacs : surface {_src}, mediane {np.median(_alac[_isl]):.2f} km2 sur {int(_isl.sum())} lacs (aire drainee mediane {np.median(_A[_isl]):.1f} km2)", flush=True)
        col = m.vertical_column
        # MEANDRE_BANC_ETP_MODE (2026-09-28) : formule d'ETP autre que Linacre calée, par
        # exemple penman, dont la forme saisonnière suit MOD16 (avril / octobre 1,58 contre 1,63).
        col.et_mode = os.environ.get("MEANDRE_BANC_ETP_MODE", "linacre")
        col.etp_channel = None
        col.set_linacre_params(*load_linacre_nodes(plat, s["node_ids"], device=dev))
        col.set_melt_params(load_melt_nodes(plat, s["node_ids"], device=dev))
        sn = load_passage_pluie_neige(plat)
        if sn:
            col.t_neige_seuil = sn
        # PARTAGE PLUIE-NEIGE ET FONTE SAISONNIERE DU PILOTE (2026-10-01). Le banc gardait le
        # seuil AIR du projet et une fonte constante, alors que la recette du pilote (socle)
        # pose le bulbe humide a -0,8 degre et l'amplitude de fonte 0,5, systeme nival repare
        # par R37. Memes cles que le pilote ; absentes = ancien banc.
        if "ETL_SEUIL_TWB" in os.environ:
            col.split_mode = "wet_bulb"
            col.t_neige_seuil = float(os.environ["ETL_SEUIL_TWB"])
            print(f"  partage pluie-neige au bulbe humide, seuil {col.t_neige_seuil:+.2f} degres", flush=True)
        if "ETL_MELT_SAISON" in os.environ:
            col.melt_seasonal_amp = float(os.environ["ETL_MELT_SAISON"])
            print(f"  fonte saisonniere, amplitude {col.melt_seasonal_amp:g}", flush=True)
        # MILIEUX HUMIDES ET PHENOLOGIE DU PROJET (2026-10-02), comme le pilote : le banc ne
        # posait que l'occupation, si bien que le reservoir de milieu humide n'existait pas.
        # Memes cles que le pilote : ETL_SANS_MH=1 l'eteint, ETL_MH_FIDELE=1 restitue le
        # couplage d'Hydrotel ; MEANDRE_BANC_PHENO_PROJET=0 garde la phenologie par defaut.
        _lc = load_occupation_sol(plat, s["node_ids"], device=dev)
        if os.environ.get("ETL_SANS_MH", "0") != "1":
            from meandre.data.hydrotel_calib import load_milieux_humides
            _mh = load_milieux_humides(plat, s["node_ids"], device=dev)
            if os.environ.get("ETL_MH_FIDELE", "0") == "1":
                col.mh_conservatif = False
            # EPREUVE DE SENSIBILITE (2026-10-02) : un projet sans fichier de milieux humides
            # (Saint-Laurent nord-ouest) n'a aucun reservoir. MEANDRE_BANC_MH_OCCUPATION=r
            # le construit depuis l'occupation : surface = fraction humide x aire locale,
            # fraction drainee = r x fraction humide (rapport median observe sur l'Outaouais
            # : 0,30 drainee pour 0,055 humide, soit r de 5 a 6). Autres parametres : les
            # defauts SWAT de la colonne, ceux de tous les fichiers d'Hydrotel.
            if not _mh and os.environ.get("MEANDRE_BANC_MH_OCCUPATION"):
                _r = float(os.environ["MEANDRE_BANC_MH_OCCUPATION"])
                _fw = _lc["f_wetland_raw"]
                _al = terr.get_physical("area_km2_local").to(_fw.device)
                _mh = {"wet_a_raw": _fw * _al, "wet_dra_fr_raw": torch.clamp(_r * _fw, 0.0, 1.0)}
                print(f"  milieux humides construits depuis l'occupation, fraction drainee = {_r:g} x fraction humide", flush=True)
            _lc.update(_mh)
            if _mh:
                _wa = _mh["wet_a_raw"]
                print(f"  milieux humides isoles : {int((_wa > 0).sum())} noeuds sur {_wa.numel()}, fraction humide moyenne {float(_lc['f_wetland_raw'].mean()):.3f}", flush=True)
        col.set_land_cover(_lc)
        if os.environ.get("MEANDRE_BANC_LACS_DIFFUS"):
            # LACS HORS RESEAU (2026-10-02), epreuve "c,T,beta,h_ref" : surface = eau libre de
            # l'occupation moins les lacs routes, part drainee rd = 1 - exp(-c x fraction), temps
            # de sejour T (jours) a la hauteur h_ref (m), loi de seuil en h^beta. Bornes douces :
            # MEANDRE_BANC_LACS_DIFFUS_BORNES="c:min:max,T:min:max". Voir HydroModel.set_distributed_lakes.
            _c, _Td, _bd, _hr = (float(x) for x in os.environ["MEANDRE_BANC_LACS_DIFFUS"].split(","))
            _bornes = {x.split(":")[0]: (float(x.split(":")[1]), float(x.split(":")[2])) for x in os.environ["MEANDRE_BANC_LACS_DIFFUS_BORNES"].split(",")}
            _Al = terr.get_physical("area_km2_local").to(dev)
            _route = torch.as_tensor(_alac, dtype=torch.float32, device=dev) if "_alac" in locals() else torch.zeros_like(_Al)
            _route = torch.where(g.is_lake.bool().to(dev), _route, torch.zeros_like(_route))
            _fd = torch.clamp(_lc["f_water_raw"].to(dev) - _route / _Al.clamp(min=1e-6), 0.0, 1.0)
            _appris = tuple(x for x in os.environ.get("MEANDRE_BANC_LACS_DIFFUS_LEARN", "").split(",") if x)
            m.set_distributed_lakes(_fd, _c, _Td, _bd, _hr, _bornes, learn=_appris)
            _rdd = 1.0 - torch.exp(-_c * _fd)
            _wa = (_Al / _Al.sum())
            print(f"  lacs hors reseau : eau libre {float((_fd * _wa).sum()):.3f} du bassin, part drainee {float((_rdd * _wa).sum()):.2f}, c {_c:g}, temps de sejour {_Td:g} j a {_hr:g} m, beta {_bd:g}, bornes {_bornes}, appris : {', '.join(_appris) or 'aucun'}", flush=True)
        if os.environ.get("MEANDRE_BANC_PHENO_PROJET", "1") == "1":
            from meandre.data.hydrotel_calib import load_phenologie
            _ph = load_phenologie(plat)
            if _ph:
                col.set_phenology(_ph)
                print(f"  phenologie du projet : {len(_ph)} classes", flush=True)
        if sol:
            z1 = float(getattr(col, "z1", 0.15))
            calib = load_calibrated_soil(plat, s["node_ids"], z1, device=dev)
            if sol == "sauf_ks":
                for k in ("ks1", "ks2", "ks3"):
                    calib.pop(k, None)
            col.set_calibrated_soil(calib)
        # CORRECTIFS DE LA COUCHE 3 ET NAPPE LIBRE (2026-09-28), lus depuis les MEMES
        # variables que le pilote regional : sans eux le banc tournait avec le clone
        # d'origine, couche 3 engorgee, 86 % de ruissellement de surface, alors que la
        # ronde les porte. ETL_L3_KSUB n'accepte ici qu'une constante en mm/jour.
        if os.environ.get("MEANDRE_GEL_CONTINU") == "1":
            # Infiltration reduite en proportion de la fraction gelee de la couche de surface,
            # au lieu du tout ou rien du clone (2026-09-28).
            col.soil.frozen_gate_continuous = True
        if os.environ.get("MEANDRE_GEL_SURFACIQUE") == "1":
            # Part gelee de la couche de surface = part impermeable de l'aire (2026-09-28).
            col.soil.frozen_gate_areal = True
        if os.environ.get("MEANDRE_BANC_SOIL_TOML"):
            # Profil de sol declare par couche, comme le pilote regional (section [soil] d'un
            # TOML) : remplace les branches ETL_L3_* par une declaration complete (2026-09-29).
            import tomllib
            from meandre.vertical import soil_processes as _soil_proc
            with open(os.environ["MEANDRE_BANC_SOIL_TOML"], "rb") as _f:
                _cfg = tomllib.load(_f)
            _profile = _soil_proc.from_toml(_cfg.get("soil"))
            if _profile is None:
                raise SystemExit("MEANDRE_BANC_SOIL_TOML : aucun processus declare dans [soil]")
            _appris = col.declare_soil_profile(_profile)
            print(f"  profil de sol declare : {_profile.layers} couches, {len(_profile.processes)} processus, appris : {', '.join(_appris) or 'aucun'}", flush=True)
            for _pr in _profile.processes:
                print(f"    couche {_pr.layer} {_pr.kind} {_pr.form} {_pr.params}", flush=True)
        if "ETL_L3_TAU" in os.environ:
            col.l3_tau_fc = float(os.environ["ETL_L3_TAU"]) * 24.0
        if "ETL_L3_KSUB" in os.environ:
            col.l3_k_sub = float(os.environ["ETL_L3_KSUB"]) / 1000.0 / 24.0
        if "ETL_L3_TAULAT" in os.environ:
            # SANS EFFET quand un profil de sol est declare par MEANDRE_BANC_SOIL_TOML : la
            # sortie laterale vient alors du profil (mesure 2026-10-01, passe identique a 3 et 8 j).
            col.l3_tau_lat = float(os.environ["ETL_L3_TAULAT"]) * 24.0
        if os.environ.get("PROV_DRAIN", "0") == "1":
            # Drainage souterrain agricole (Hooghoudt), memes cles que le pilote provincial.
            col.drainage_agricole = dict(espacement_m=float(os.environ.get("PROV_DRAIN_L", "15")),
                                         profondeur_m=float(os.environ.get("PROV_DRAIN_Z", "1.0")),
                                         part_cultive=float(os.environ.get("PROV_DRAIN_F", "0.6")))
            print(f"  drainage agricole : {col.drainage_agricole}", flush=True)
        if "MEANDRE_RACINES_ECHELLE" in os.environ:
            col.root_depth_scale = float(os.environ["MEANDRE_RACINES_ECHELLE"])
            print(f"  racines : profondeur x {col.root_depth_scale:g}", flush=True)
        if os.environ.get("ETL_NAPPE_LIBRE", "0") == "1":
            col.activer_nappe_libre(sy=float(os.environ.get("ETL_NAPPE_SY", 0.05)), k_b=float(os.environ.get("ETL_NAPPE_KB", 2.0e-3)), z_riv=float(os.environ.get("ETL_NAPPE_ZRIV", 8.0)), h_ref=float(os.environ.get("ETL_NAPPE_HREF", 4.0)), e_frac=float(os.environ.get("ETL_NAPPE_EFRAC", 0.35)), z_ext=float(os.environ.get("ETL_NAPPE_ZEXT", 9.0)), exposant=float(os.environ.get("ETL_NAPPE_EXP", 2.0)), couplage=float(os.environ.get("ETL_NAPPE_COUPLAGE", 0.0)), apprise=os.environ.get("ETL_NAPPE_APPRIS") == "1")
        return m

    def _tranche(a, b):
        i0 = int(np.argmax(temps.year >= a))
        i1 = int(np.argmax(temps.year > b))
        return slice(i0, i1 if i1 > 0 else len(temps))

    w = WithdrawalData(net=torch.zeros(F.shape[0], n, device=dev))
    if os.environ.get("MEANDRE_BANC_PRELEVEMENTS", "1") == "1":
        # PRELEVEMENTS REELS, PAR DEFAUT depuis le 2026-09-30 (decision d'Essi : les debits
        # observes les contiennent, le banc doit les retirer). MEANDRE_BANC_PRELEVEMENTS=0
        # restitue le banc sans prelevements, celui de tous les resultats anterieurs. Le banc
        # tournait sans prelevements alors que
        # les debits observes les contiennent : sur les deux sous-bassins ils valent 2 a
        # 6 % du debit d'ete, l'ordre de grandeur que la sensibilite doit resoudre.
        from meandre.data.basin_cache import BasinCache as _BC
        _wt = _BC(s["base"]).load_withdrawals(str(temps[0].date()), str(temps[-1].date()), device=dev)
        _ix = torch.tensor(idx, device=dev)
        w = WithdrawalData(net=_wt.net[:, _ix].clone(), net_gw=_wt.net_gw[:, _ix].clone())
        print(f"  prelevements reels : surface {float(w.net.sum(dim=1).mean()):+.3f} m3/s, souterrain {float(w.net_gw.sum(dim=1).mean()):+.3f} m3/s en moyenne (positif = ajoute)", flush=True)
    if os.environ.get("MEANDRE_BANC_PRELEVEMENT_TEST"):
        # SENSIBILITE (2026-09-30) : « surface:0.05 » ou « gw:0.05 » retire chaque jour une
        # fraction du debit moyen observe a l'exutoire, repartie sur les troncons par aire
        # locale. Dit si un prelevement de cette taille laisse sur l'etiage simule une
        # signature plus grande que la dispersion entre graines.
        _kind, _frac = os.environ["MEANDRE_BANC_PRELEVEMENT_TEST"].split(":")
        _qm = float(np.nanmean(o))
        _al = terr.get_physical("area_km2_local").to(dev).float()
        _part = (_al / _al.sum()).reshape(1, -1)
        _retrait = -float(_frac) * _qm * _part.expand(F.shape[0], -1)
        if _kind == "gw":
            w = WithdrawalData(net=w.net, net_gw=w.net_gw + _retrait)
        else:
            w = WithdrawalData(net=w.net + _retrait, net_gw=w.net_gw)
        print(f"  prelevement de test : {_kind} {float(_frac) * 100:.0f} % du debit moyen observe ({float(_frac) * _qm:.3f} m3/s), reparti par aire", flush=True)
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
    # PUITS DU SOUS-BASSIN (2026-09-23). La question posee au banc : le terme des puits
    # achete-t-il de la nappe INDEPENDANTE une fois l'optimiseur converge, ce que huit
    # epoques regionales ne peuvent pas dire. Les puits du domaine sont restreints aux
    # nœuds du sous-bassin, puis partages par un ordre stable : la moitie dans la perte, la
    # moitie tenue de cote et evaluee en fin d'entrainement.
    nappe_obs = nappe_idx = None
    puits_garde, nappe_garde = [], None
    # Les puits sont extraits des que la moitie tenue de cote doit etre evaluee, y compris
    # pour le temoin a poids nul : sans cela le temoin n'avait pas de ligne de puits a
    # comparer (2026-09-23). Seul le terme de perte reste conditionne au poids.
    if float(w_nappe) > 0 or float(nappe_valid) > 0:
        import pandas as _pdn
        from meandre.data.rsesq_loader import _chemin_defaut, read_rsesq

        _cn = read_rsesq(reg, temps)
        _pos = {int(n): i for i, n in enumerate(s["idx"])}
        _dans = [j for j, n in enumerate(_cn.node_idx) if int(n) in _pos]
        if _dans:
            _pu = [str(_cn.puits[j]) for j in _dans]
            _niv = _pdn.read_parquet(f"{_chemin_defaut()}/rsesq-niveaux-journaliers.parquet",
                                     columns=["puits", "date", "niveau_m"])
            _niv = _niv[_niv.puits.isin(_pu)]
            _tab = (_niv.pivot_table(index="date", columns="puits", values="niveau_m",
                                     aggfunc="mean")
                    .reindex(index=_pdn.DatetimeIndex(temps), columns=_pu))
            _val = _tab.to_numpy(dtype="float32")
            _rang = sorted(range(len(_pu)), key=lambda i: _pu[i])
            _n_val = max(1, int(round(float(nappe_valid) * len(_pu)))) if len(_pu) >= 4 else 0
            puits_garde = sorted(_rang[::max(len(_pu) // max(_n_val, 1), 1)][:_n_val]) if _n_val else []
            _entraine = [i for i in range(len(_pu)) if i not in set(puits_garde)]
            _sb_idx = [_pos[int(_cn.node_idx[_dans[i]])] for i in range(len(_pu))]
            if float(w_nappe) > 0:
                nappe_obs = torch.tensor(_val[:, _entraine], device=dev)
                nappe_idx = torch.tensor([_sb_idx[i] for i in _entraine], dtype=torch.long, device=dev)
            nappe_garde = (_val[:, puits_garde], [_sb_idx[i] for i in puits_garde],
                           [_pu[i] for i in puits_garde])
            print(f"  puits : {len(_pu)} dans le sous-bassin, {len(_entraine)} "
                  f"{'dans la perte' if float(w_nappe) > 0 else 'hors perte (poids nul)'}, "
                  f"{len(puits_garde)} tenus de cote "
                  f"({', '.join(_pu[i] for i in puits_garde) or 'aucun'})", flush=True)
        else:
            print("  puits : aucun puits recevable dans ce sous-bassin, terme inactif", flush=True)
            w_nappe = 0.0
    td = TrainingData(train_slice=tr_sl, val_slice=va_sl,
                      **{**commun, "q_obs": q_obs[tr_sl.start:],
                         "et_obs": (et_obs[tr_sl.start:] if et_obs is not None else None),
                         "nappe_obs": (nappe_obs[tr_sl.start:] if nappe_obs is not None else None),
                         "nappe_idx": nappe_idx})
    vd = TrainingData(train_slice=va_sl, val_slice=va_sl,
                      **{**commun, "q_obs": q_obs[va_sl.start:],
                         "et_obs": (et_obs[va_sl.start:] if et_obs is not None else None),
                         "nappe_obs": (nappe_obs[va_sl.start:] if nappe_obs is not None else None),
                         "nappe_idx": nappe_idx})

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
        # PUITS TENUS DE COTE : correlation entre profondeur simulee et niveau mesure, en
        # moyennes mensuelles, sur toute la periode. Seule preuve que le terme achete de la
        # nappe et pas seulement du debit perdu.
        if nappe_garde is not None:
            _z = getattr(_d, "profondeur_nappe", None)
            _z = _z if _z is not None else getattr(_d, "s_gw", None)
            if _z is not None:
                import pandas as _pde
                _zs = _z.detach().cpu().numpy()[:, nappe_garde[1]]
                _ax = _pde.DatetimeIndex(temps)
                _rs = []
                for _j in range(_zs.shape[1]):
                    _o = _pde.Series(nappe_garde[0][:, _j], index=_ax).resample("MS").mean()
                    _s = _pde.Series(_zs[:, _j], index=_ax).resample("MS").mean()
                    _ok = _o.notna() & _s.notna()
                    _rs.append(float(np.corrcoef(-_s[_ok], -_o[_ok])[0, 1]) if int(_ok.sum()) >= 24 else float("nan"))
                print(f"  {etiquette:28s} puits tenus de cote : " + ", ".join(
                    f"{nappe_garde[2][j]} r {_rs[j]:5.2f}" for j in range(len(_rs))), flush=True)
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
    if os.environ.get("MEANDRE_BANC_MULT_FIXE") and not os.environ.get("MEANDRE_BANC_MENSUEL"):
        # Multiplicateur FIXE sur des champs, actif aussi pendant l'entrainement (2026-09-28) :
        # par exemple K_c:0.5 pour ramener Penman au volume de Linacre calee, le champ
        # apprenant autour.
        m.spatial_encoder.multiplicateurs = {a.split(":")[0]: torch.tensor(float(a.split(":")[1]), device=dev) for a in os.environ["MEANDRE_BANC_MULT_FIXE"].split(",")}
        print(f"  champ spatial : multiplicateurs fixes {os.environ['MEANDRE_BANC_MULT_FIXE']}", flush=True)
    if os.environ.get("MEANDRE_KMUSK"):
        # INITIALISATION DU TEMPS DE TRANSFERT (2026-09-30). Le banc n'appelle pas
        # l'initialisation par la litterature : chaque sortie du champ part du MILIEU de ses
        # bornes, et la valeur d'initialisation de MEANDRE_KMUSK etait ignoree (deux
        # entrainements initialises a 6 et 24 h sont sortis identiques au bit pres). On pose
        # ici le biais de cette seule sortie pour qu'elle parte de la valeur demandee.
        import math as _mk
        from dataclasses import fields as _dcf
        from meandre.spatial.field_network import SpatialParams as _SPk, _KMUSK_MIN as _k0, _KMUSK_MAX as _k1, _KMUSK_INIT as _ki
        _row = [f.name for f in _dcf(_SPk)].index("K_musk_hours")
        _fr = min(max((_ki - _k0) / (_k1 - _k0), 1e-4), 1 - 1e-4)
        with torch.no_grad():
            m.spatial_encoder.fc_out.bias[_row] = _mk.log(_fr / (1 - _fr))
        print(f"  temps de transfert initialise a {_ki:.1f} h (bornes {_k0:.0f} a {_k1:.0f} h)", flush=True)
    if os.environ.get("MEANDRE_CHAMP_GELE"):
        # Sorties du champ gelees a leur valeur d'initialisation, uniforme (2026-09-28).
        _gel = [x for x in os.environ["MEANDRE_CHAMP_GELE"].split(",") if x]
        m.spatial_encoder.freeze_outputs(_gel)
        print(f"  champ spatial : sorties gelees {', '.join(_gel)}", flush=True)
    if os.environ.get("MEANDRE_BANC_MENSUEL"):
        # BILAN MENSUEL (2026-09-28) : pour chaque point de reprise nomme, moyennes sur le
        # bassin par mois de l'annee, en mm/j : precipitation, ET simulee, MOD16, debit
        # observe et simule, et residu du bilan P - Q observe. Dit dans quelle saison se
        # trouve le surplus de precipitation, et quelle ET l'absorbe.
        _a = terr.get_physical("area_km2_local").to(dev).float()
        _w = _a / _a.sum()
        _aire_m2 = float(s["aire"]) * 1e6 if s.get("aire") else float(_a.sum()) * 1e6
        _mois = pd.DatetimeIndex(temps).month
        _ok_t = pd.DatetimeIndex(temps).year >= 2011
        if os.environ.get("MEANDRE_BANC_ETP_FORMES") == "1":
            # FORME SAISONNIÈRE DE L'ETP (2026-09-28) : chaque formule de la colonne calculée
            # jour par jour sur le forçage du sous-bassin, moyennée sur le bassin et par mois,
            # contre MOD16. Aucune simulation : ne dépend que du forçage et des ancrages.
            col = m.vertical_column
            lat_n = coords[:, 1].to(dev)
            res = {}
            for _mode in ("linacre", "mcguinness", "oudin", "penman"):
                col.et_mode = _mode
                vals = []
                with torch.no_grad():
                    for t in range(len(temps)):
                        f = F[t]
                        z = torch.zeros(n, device=dev)
                        try:
                            e = col._etp(f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5], lat_n, doy[t:t + 1].to(dev).float().expand(n), couv=z, albn=z)
                        except TypeError:
                            e = col._etp(f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5], lat_n, doy[t:t + 1].to(dev).float().expand(n))
                        vals.append(float((e * _w).sum()))
                res[_mode] = np.array(vals)
            col.et_mode = "linacre"
            if et_obs is not None:
                _vo = torch.isfinite(et_obs)
                MODs = (torch.nan_to_num(et_obs) * _w * _vo).sum(dim=1) / (_w * _vo).sum(dim=1).clamp(min=1e-9)
                MODs = torch.where(_vo.any(dim=1), MODs, torch.full_like(MODs, float("nan"))).cpu().numpy()
            else:
                MODs = np.full(len(temps), np.nan)
            dfe = pd.DataFrame({"mois": _mois, "MOD16": MODs, **res})[_ok_t].groupby("mois").mean()
            print("  ETP mensuelle par formule, mm/j, moyennes du bassin, contre MOD16 (evapotranspiration REELLE) :", flush=True)
            print(dfe.round(2).to_string(), flush=True)
            print("  rapport avril / octobre : " + ", ".join(f"{c} {dfe.loc[4, c] / dfe.loc[10, c]:.2f}" for c in dfe.columns), flush=True)
            return
        for _ck in os.environ["MEANDRE_BANC_MENSUEL"].split(";"):
            if _ck != "initial":
                m.load(_ck)
            if os.environ.get("MEANDRE_BANC_MULT_FIXE"):
                m.spatial_encoder.multiplicateurs = {a.split(":")[0]: torch.tensor(float(a.split(":")[1]), device=dev) for a in os.environ["MEANDRE_BANC_MULT_FIXE"].split(",")}
            m.eval()
            with torch.no_grad():
                Q, _, _d = m.simulate(forcing=F, initial_state=HydroState.zeros(n, device=dev), graph=g, node_coords=coords, territorial=terr, withdrawals=w, day_of_year=doy, return_diagnostics=True)
            P = (F[:, :, 0] * _w).sum(dim=1).cpu().numpy()
            ET = (_d.etr.to(dev) * _w).sum(dim=1).cpu().numpy()
            if et_obs is not None:
                _vo = torch.isfinite(et_obs)
                MOD = (torch.nan_to_num(et_obs) * _w * _vo).sum(dim=1) / (_w * _vo).sum(dim=1).clamp(min=1e-9)
                MOD = torch.where(_vo.any(dim=1), MOD, torch.full_like(MOD, float("nan"))).cpu().numpy()
            else:
                MOD = np.full(len(temps), np.nan)
            conv = 86400.0 * 1000.0 / _aire_m2
            QS = Q[:, s["exutoire"]].cpu().numpy() * conv
            QO = o * conv
            # Composantes de la production verticale moyennees sur le bassin, en mm/j :
            # ruissellement de surface, ecoulement hypodermique, et sortie de la nappe.
            def _moy(x):
                return (x.to(dev) * _w).sum(dim=1).cpu().numpy() if x is not None else np.full(len(temps), np.nan)
            _comp = {"surf": _moy(getattr(_d, "prod_surf", None)), "hypo": _moy(getattr(_d, "prod_hypo", None)), "nappe": _moy(getattr(_d, "q_baseflow", None))}
            # Humidite des couches, en fraction de la porosite imposee par le calage.
            _sp = m.spatial_encoder(coords, terr.data)
            for _nom, (_val, _u) in m.vertical_column.soil_learned_values().items():
                print(f"  profil appris {_nom} : {_val:.4g} {_u}", flush=True)
            if getattr(m, "_distributed_lakes", None) is not None:
                _vl = m.distributed_lakes_values()
                _rdv = 1.0 - torch.exp(-_vl["c"] * m._distributed_lakes["f"].to(_w.device).reshape(-1))
                print(f"  lacs hors reseau appris : c {float(_vl['c']):.4g} (part drainee {float((_rdv * _w.reshape(-1)).sum()):.2f}), temps de sejour {float(_vl['T']):.4g} j", flush=True)
            _ths = getattr(m.vertical_column, "_static", {}).get("soil", {})
            if os.environ.get("MEANDRE_BANC_PARAMS_SOL") and isinstance(_ths, dict):
                # Proprietes du sol en vigueur, medianes et quartiles sur les noeuds (2026-10-01).
                for _nom in sorted(_ths):
                    _v = _ths[_nom]
                    if torch.is_tensor(_v) and _v.numel() > 1:
                        _q = torch.quantile(_v.float().flatten().cpu(), torch.tensor([0.25, 0.5, 0.75]))
                        print(f"  sol {_nom} : mediane {_q[1]:.4g}, quartiles {_q[0]:.4g} a {_q[2]:.4g}", flush=True)
                    elif torch.is_tensor(_v) or isinstance(_v, (int, float)):
                        print(f"  sol {_nom} : {float(_v):.4g}", flush=True)
                # Audit des 43 sorties du champ (2026-10-01) : une sortie UNIFORME (centiles 1 et
                # 99 a moins de 1 % l'un de l'autre) est soit gelee, soit collee a une borne.
                from dataclasses import fields as _dcf2
                for _f in _dcf2(type(_sp)):
                    _v = getattr(_sp, _f.name)
                    if not torch.is_tensor(_v) or _v.numel() < 2:
                        continue
                    _v = _v.float().flatten().cpu()
                    _q = torch.quantile(_v, torch.tensor([0.01, 0.5, 0.99]))
                    _unif = float(_q[2] - _q[0]) <= 0.01 * max(abs(float(_q[1])), 1e-12)
                    print(f"  champ {_f.name} : mediane {_q[1]:.4g}, centiles 1 et 99 {_q[0]:.4g} a {_q[2]:.4g}{'  UNIFORME' if _unif else ''}", flush=True)
            for _k in (1, 2, 3):
                _th = getattr(_d, f"theta{_k}", None)
                _por = _ths.get(f"thetas{_k}") if isinstance(_ths, dict) else None
                if _th is not None:
                    _por = _por if torch.is_tensor(_por) else getattr(_sp, f"porosity_{_k}")
                    _comp[f"sat{_k}"] = _moy(_th.to(dev) / _por.to(dev).reshape(1, -1).clamp(min=1e-6))
            # Part de la journee que la boucle de sous-pas n'a pas traitee : sa pluie est
            # versee au ruissellement par la fermeture de masse du clone.
            # Profondeur de gel (cm) et apport au sol (pluie et fonte, mm/j) : un sol gele
            # refuse l'infiltration, et l'apport ruisselle.
            # Eau disponible par couche, 1 a la capacite au champ et 0 au point de fletrissement,
            # avec les seuils que l'ETR emploie (ceux de la couche 1, comme le clone) : dit si le
            # profil se creuse l'ete, donc s'il a un deficit a combler a l'automne (R216).
            _pe = getattr(m.vertical_column, "_static", {}).get("etr", {})
            if isinstance(_pe, dict) and "thetacc" in _pe and "thetapf" in _pe:
                _cc = _pe["thetacc"].to(dev).reshape(1, -1)
                _pf = _pe["thetapf"].to(dev).reshape(1, -1)
                for _k in (1, 2, 3):
                    _th = getattr(_d, f"theta{_k}", None)
                    if _th is not None:
                        _comp[f"dispo{_k}"] = _moy((_th.to(dev) - _pf) / (_cc - _pf).clamp(min=1e-6))
            _comp["gel_cm"] = _moy(getattr(_d, "prof_gel_cm", None))
            _comp["recharge"] = _moy(getattr(_d, "recharge", None))
            for _k in (1, 2, 3):
                _comp[f"et{_k}"] = _moy(getattr(_d, f"etr{_k}", None))
            _comp["apport"] = _moy(getattr(_d, "snowmelt", None))
            # Equivalent en eau de la neige moyen du mois (mm) et pluie liquide (mm/j) : P - apport
            # se lit alors comme accumulation du manteau, et le biais d'hiver se localise (2026-10-01).
            _comp["swe"] = _moy(getattr(_d, "swe", None))
            _fn = f"{_p.DERIVED_ROOT}/auxiliaires/neisim-{reg}.npz"
            if os.path.exists(_fn):
                # NEISIM (modele du gouvernement, valide sur CanSWE) aux memes noeuds, moyenne ponderee.
                _z = np.load(_fn)
                _ti = pd.DatetimeIndex(_z["times"])
                _col = np.searchsorted(_z["node_idx"], np.asarray(s["idx"]))
                _vz = _z["valeurs"][:, _col]
                _wn = _w.detach().cpu().numpy().ravel()
                _ok = np.isfinite(_vz)
                _sw = np.where(_ok, _vz, 0.0) @ _wn / np.maximum(_ok.astype(float) @ _wn, 1e-9)
                _sw[~_ok.any(axis=1)] = np.nan
                _comp["swe_neisim"] = pd.Series(_sw, index=_ti).reindex(pd.DatetimeIndex(temps)).to_numpy()
            # CanSWE aux noeuds du sous-bassin qui portent un site (2026-10-01) : couples
            # journaliers mesure / simule / NEISIM, moyennes par mois, mesure OBSERVEE et non modele.
            try:
                from meandre.data.basin_cache import BasinCache as _BCs
                _mes, _sit = _BCs(s["base"]).load_canswe(str(temps[0].date()), str(temps[-1].date()))
            except Exception as _e:
                _mes = None
                print(f"  CanSWE : lecture impossible ({_e})", flush=True)
            if _mes is not None and not _mes.empty:
                _rang = {int(v): k for k, v in enumerate(s["idx"])}
                _dans = _mes["node_idx"].isin(_rang)
                if not _dans.any() and _sit is not None and {"lat", "lon"} <= set(_sit.columns):
                    # Aucun site sur un noeud du sous-bassin : on prend les sites a moins de
                    # 15 km d'un noeud du sous-bassin (meme rayon que R113), rattaches au
                    # noeud le plus proche, pour comparer le manteau sur le meme climat.
                    _nc = s["node_coords"].detach().cpu().numpy()
                    _la, _lo = np.radians(_nc[:, 1]), np.radians(_nc[:, 0])
                    _prox = {}
                    for _, _r in _sit.iterrows():
                        if not (np.isfinite(_r["lat"]) and np.isfinite(_r["lon"])):
                            continue
                        _dl = np.radians(_r["lat"]) - _la
                        _dn = np.radians(_r["lon"]) - _lo
                        _h = np.sin(_dl / 2) ** 2 + np.cos(_la) * np.cos(np.radians(_r["lat"])) * np.sin(_dn / 2) ** 2
                        _dk = 2 * 6371.0 * np.arcsin(np.sqrt(_h))
                        if _dk.min() <= 15.0:
                            _prox[_r["swe_station_id"]] = int(_dk.argmin())
                    _mes = _mes[_mes["swe_station_id"].isin(_prox)].copy()
                    _mes["local"] = _mes["swe_station_id"].map(_prox).astype(int)
                    print(f"  CanSWE : aucun site sur un noeud du sous-bassin ; {len(_prox)} sites a moins de 15 km rattaches au noeud le plus proche", flush=True)
                else:
                    _mes = _mes[_dans].copy()
                    _mes["local"] = _mes["node_idx"].map(_rang).astype(int)
                if _mes.empty:
                    print("  CanSWE : aucun site dans ou pres du sous-bassin", flush=True)
                else:
                    _pos = {d: k for k, d in enumerate(pd.DatetimeIndex(temps).normalize())}
                    _mes["t"] = _mes["date"].map(lambda x: _pos.get(pd.Timestamp(x).normalize()))
                    _mes = _mes[_mes["t"].notna()].copy()
                    _mes["t"] = _mes["t"].astype(int)
                    _swe_sim = _d.swe.detach().cpu().numpy()
                    _mes["sim"] = _swe_sim[_mes["t"].values, _mes["local"].values]
                    if "swe_neisim" in _comp and os.path.exists(_fn):
                        _ri = {d: k for k, d in enumerate(_ti)}
                        _mes["tn"] = _mes["date"].map(lambda x: _ri.get(pd.Timestamp(x).normalize()))
                        _ok_n = _mes["tn"].notna()
                        _mes["neisim"] = np.nan
                        # _vz est deja restreint aux noeuds du sous-bassin, dans l'ordre local.
                        _mes.loc[_ok_n, "neisim"] = _vz[_mes.loc[_ok_n, "tn"].astype(int).values, _mes.loc[_ok_n, "local"].values]
                    _mes["mois"] = pd.DatetimeIndex(_mes["date"]).month
                    _g = _mes.groupby("mois").agg(n=("swe_mm", "size"), obs=("swe_mm", "mean"), sim=("sim", "mean"), neisim=("neisim", "mean") if "neisim" in _mes else ("sim", "size"))
                    _g["sim/obs"] = _g["sim"] / _g["obs"]
                    if "neisim" in _mes:
                        _g["neisim/obs"] = _g["neisim"] / _g["obs"]
                    print(f"  CanSWE, {_mes.swe_station_id.nunique()} sites dans le sous-bassin, {len(_mes)} couples, mm d'equivalent en eau :", flush=True)
                    print(_g.round(2).to_string(), flush=True)
            _tnt = getattr(_d, "temps_non_traite", None)
            if _tnt is not None:
                _comp["non_traite"] = _moy(_tnt)
            df = pd.DataFrame({"mois": _mois, "P": P, "ET": ET, "MOD16": MOD, "Qobs": QO, "Qsim": QS, **_comp})[_ok_t]
            if os.environ.get("MEANDRE_BANC_SERIES"):
                # Series journalieres moyennees sur le bassin, en mm/j, pour l'analyse hors banc.
                pd.DataFrame({"P": P, "ET": ET, "MOD16": MOD, "Qobs": QO, "Qsim": QS, **_comp}, index=pd.DatetimeIndex(temps)).to_parquet(os.environ["MEANDRE_BANC_SERIES"])
                print(f"  series journalieres ecrites dans {os.environ['MEANDRE_BANC_SERIES']}", flush=True)
            t = df.groupby("mois").mean()
            t["P-Qobs"] = t.P - t.Qobs
            print(f"  bilan mensuel, {os.path.basename(_ck)}, mm/j, moyennes 2011-{int(pd.DatetimeIndex(temps).year.max())} :", flush=True)
            print(t.round(2).to_string(), flush=True)
            print(f"  annee : P {df.P.mean():.2f}, ET {df.ET.mean():.2f}, MOD16 {np.nanmean(df.MOD16):.2f}, Qobs {np.nanmean(df.Qobs):.2f}, Qsim {df.Qsim.mean():.2f}, P-Qobs {df.P.mean() - np.nanmean(df.Qobs):.2f}", flush=True)
            # STOCK PAR RESERVOIR (2026-10-01), mm, moyenne du mois moins moyenne annuelle, a cote
            # du stock OBSERVE du bassin : cumul de P - MOD16 - Qobs, debiaise de son residu annuel
            # moyen (MOD16 n'est pas ferme sur le bilan), puis centre. Dit quel reservoir porte
            # la mise en reserve d'automne et la restitution d'avril, et lequel manque.
            _stk = {}
            _zs = {k: _ths.get(f"z{k}") for k in (1, 2, 3)} if isinstance(_ths, dict) else {}
            for _k in (1, 2, 3):
                _th = getattr(_d, f"theta{_k}", None)
                _z = _zs.get(_k)
                if _th is not None and _z is not None:
                    _stk[f"sol{_k}"] = _moy(_th.to(dev) * (_z.to(dev) if torch.is_tensor(_z) else float(_z)) * 1000.0)
            for _nom, _att in (("neige", "swe"), ("nappe", "s_gw"), ("mh", "wet_vol"), ("canopee", "canopy")):
                _v = getattr(_d, _att, None)
                if _v is not None:
                    _stk[_nom] = _moy(_v)
            if _stk:
                _ds = pd.DataFrame({"mois": _mois, **_stk})[_ok_t]
                _ann = pd.DatetimeIndex(temps)[_ok_t].year
                _clim = _ds.groupby("mois").mean()
                _clim = _clim - _clim.mean()
                _clim["total"] = _clim.sum(axis=1)
                _r = (df.P - df.MOD16 - df.Qobs)
                _rs = (df.P - df.ET - df.Qsim)
                _jours = pd.Series(pd.DatetimeIndex(temps)[_ok_t].days_in_month, index=df.index)
                _mo = (_r.groupby(df.mois).mean() - _r.mean()) * _jours.groupby(df.mois).mean()
                _ms = (_rs.groupby(df.mois).mean() - _rs.mean()) * _jours.groupby(df.mois).mean()
                # Stock de fin de mois par cumul ; on le ramene au milieu du mois et on le centre.
                _co = _mo.cumsum() - _mo / 2.0
                _cs = _ms.cumsum() - _ms / 2.0
                _clim["obs P-MOD16-Q"] = _co - _co.mean()
                _clim["sim P-ET-Q"] = _cs - _cs.mean()
                print(f"  stock par reservoir, mm, ecart a la moyenne annuelle (obs = cumul de P - MOD16 - Qobs debiaise) :", flush=True)
                print(_clim.round(0).to_string(), flush=True)
            if mesure is not None:
                # Debit mensuel sur les SEULS jours mesures (2026-10-01) : l'hiver observe est aux
                # trois quarts reconstruit sous glace (R98) ; ce tableau separe le defaut du
                # modele de celui de la reference.
                _dm = df.assign(mes=mesure[_ok_t])
                _tm = _dm[_dm.mes].groupby("mois").agg(jours=("Qobs", "size"), Qobs=("Qobs", "mean"), Qsim=("Qsim", "mean"))
                _tt = _dm.groupby("mois").agg(Qobs_tous=("Qobs", "mean"), Qsim_tous=("Qsim", "mean"))
                _tm = _tm.join(_tt)
                _tm["sim/obs mesures"] = _tm.Qsim / _tm.Qobs
                _tm["sim/obs tous"] = _tm.Qsim_tous / _tm.Qobs_tous
                print("  debit par mois, jours mesures contre tous les jours, mm/j :", flush=True)
                print(_tm.round(2).to_string(), flush=True)
                # Par annee, sur jours mesures : avril, et juin a octobre (2026-10-01). Dit si les
                # deux ecarts sont systematiques ou portes par quelques evenements.
                _dm["annee"] = pd.DatetimeIndex(temps)[_ok_t].year
                _m = _dm[_dm.mes]
                _av = _m[_m.mois == 4].groupby("annee").agg(avril_obs=("Qobs", "mean"), avril_sim=("Qsim", "mean"))
                _et = _m[_m.mois.isin((6, 7, 8, 9, 10))].groupby("annee").agg(jun_oct_obs=("Qobs", "mean"), jun_oct_sim=("Qsim", "mean"))
                _pa = _av.join(_et, how="outer")
                _pa["avril sim/obs"] = _pa.avril_sim / _pa.avril_obs
                _pa["jun-oct sim/obs"] = _pa.jun_oct_sim / _pa.jun_oct_obs
                print("  par annee, jours mesures, mm/j :", flush=True)
                print(_pa.round(2).to_string(), flush=True)
            # VOLUME PAR ANNEE (2026-10-01) : debit simule et observe, precipitation et ET, sur les
            # jours ou l'observe existe. Dit si le biais de volume derive d'une periode a l'autre.
            _va = df.assign(annee=pd.DatetimeIndex(temps)[_ok_t].year).dropna(subset=["Qobs"])
            _vy = _va.groupby("annee").agg(jours=("Qobs", "size"), P=("P", "mean"), ET=("ET", "mean"), Qobs=("Qobs", "mean"), Qsim=("Qsim", "mean"))
            _vy["Qsim/Qobs"] = _vy.Qsim / _vy.Qobs
            _vy["(P-Qobs)"] = _vy.P - _vy.Qobs
            print("  volume par annee, jours ou le debit observe existe, mm/j :", flush=True)
            print(_vy.round(2).to_string(), flush=True)
            # Correlation journaliere des debits par saison, 2011 a la fin du chargement.
            _sais = {"hiver (dec-fev)": (12, 1, 2), "printemps (mar-mai)": (3, 4, 5), "ete (jun-aou)": (6, 7, 8), "automne (sep-nov)": (9, 10, 11)}
            _r = []
            for _nom, _ms in _sais.items():
                _k = df[df.mois.isin(_ms)][["Qobs", "Qsim"]].dropna()
                _r.append(f"{_nom} {np.corrcoef(_k.Qobs, _k.Qsim)[0, 1]:.2f}" if len(_k) > 30 else f"{_nom} -")
            print("  correlation journaliere par saison : " + ", ".join(_r), flush=True)
            # KGE de la derniere annee chargee, a poids fixes : separe ce qu'une recette coute
            # par elle-meme de ce que l'entrainement en fait.
            _an = int(pd.DatetimeIndex(temps).year.max())
            _k = df[pd.DatetimeIndex(temps)[_ok_t].year == _an][["Qobs", "Qsim"]].dropna()
            if len(_k) > 60:
                _rr = np.corrcoef(_k.Qobs, _k.Qsim)[0, 1]
                _b = _k.Qsim.mean() / _k.Qobs.mean()
                _g = (_k.Qsim.std() / _k.Qsim.mean()) / (_k.Qobs.std() / _k.Qobs.mean())
                _po = _k.Qsim.max() / _k.Qobs.max()
                _q99 = _k.Qsim.quantile(0.99) / _k.Qobs.quantile(0.99)
                print(f"  pointe annuelle {_an} sim/obs {_po:.2f} | q99 {_q99:.2f}", flush=True)
                # Les trois plus gros jours simules de l'annee, avec leurs composantes : dit
                # d'ou vient une pointe isolee (ruissellement, hypodermique, nappe, apport).
                _da = df[pd.DatetimeIndex(temps)[_ok_t].year == _an].copy()
                _da["date"] = pd.DatetimeIndex(temps)[_ok_t][pd.DatetimeIndex(temps)[_ok_t].year == _an]
                _cols = [c for c in ("date", "P", "apport", "Qobs", "Qsim", "surf", "hypo", "nappe", "sat1", "gel_cm") if c in _da.columns]
                print("  trois plus gros jours simules :", flush=True)
                _top = _da.nlargest(3, "Qsim")[_cols]
                _top["date"] = _top.date.dt.strftime("%Y-%m-%d")
                print(_top.round(2).to_string(index=False), flush=True)
                print(f"  KGE {_an} a poids fixes : {1 - np.sqrt((_rr - 1) ** 2 + (_b - 1) ** 2 + (_g - 1) ** 2):.3f} | r {_rr:.3f} | beta {_b:.3f} | gamma {_g:.3f}", flush=True)
            if getattr(m.vertical_column, "_nappe_apprise", False):
                _nv = m.vertical_column.nappe_valeurs()
                print(f"  nappe apprise : conductance {_nv['k_b']:.2e} m/j, extraction {_nv['e_frac']:.3f}, exposant {_nv['exposant']:.2f}", flush=True)
            # TEMPS DE TRANSFERT DE MUSKINGUM APPRIS (2026-09-30) : mediane, quartiles et part
            # des troncons colles aux bornes. Dit si le routage libre compense encore un
            # stockage absent (borne haute) ou suit le parcours physique (borne basse).
            try:
                _km = _sp.K_musk_hours.detach().float().cpu().numpy().ravel()
                from meandre.spatial.field_network import _KMUSK_MIN as _kmn, _KMUSK_MAX as _kmx
                print(f"  temps de transfert Muskingum : mediane {np.median(_km):.1f} h, quartiles {np.quantile(_km, 0.25):.1f} a {np.quantile(_km, 0.75):.1f} h | a la borne basse {100 * np.mean(_km < _kmn + 0.5):.0f} %, a la borne haute {100 * np.mean(_km > _kmx - 0.5):.0f} % ({_kmn:.0f} a {_kmx:.0f} h)", flush=True)
            except Exception as _ek:
                print(f"  temps de transfert Muskingum : lecture impossible ({type(_ek).__name__})", flush=True)
            # PUITS, TOUS, A POIDS FIXES (2026-09-30) : correlation des moyennes mensuelles
            # entre la profondeur simulee au troncon du puits et le niveau mesure. Seule
            # observation qui separe des reglages de nappe que le debit ne distingue pas.
            try:
                _zp = getattr(_d, "profondeur_nappe", None)
                if _zp is not None and len(_pu):
                    _zs = _zp.detach().cpu().numpy()[:, _sb_idx]
                    _ax = pd.DatetimeIndex(temps)
                    _rp = []
                    for _j in range(len(_pu)):
                        _o = pd.Series(_val[:, _j], index=_ax).resample("MS").mean()
                        _s = pd.Series(_zs[:, _j], index=_ax).resample("MS").mean()
                        _okp = _o.notna() & _s.notna()
                        _rp.append(float(np.corrcoef(-_s[_okp], -_o[_okp])[0, 1]) if int(_okp.sum()) >= 24 else float("nan"))
                    print(f"  puits a poids fixes, r mensuel : moyenne {np.nanmean(_rp):.2f} | " + ", ".join(f"{_pu[j]} {_rp[j]:.2f}" for j in range(len(_pu))), flush=True)
            except (NameError, TypeError, IndexError) as _ep:
                # Sans puits lus (variable ecrasee par un flottant), la section se tait au lieu
                # de faire tomber le bilan avant les grandeurs d'etiage (2026-10-02).
                print(f"  puits a poids fixes : lecture impossible ({type(_ep).__name__})", flush=True)
            # GRANDEURS D'ETIAGE (2026-09-30), sur les jours mesures de TOUTE la periode
            # d'evaluation (debut_eval a la fin du chargement) : une annee seule en mode rapide,
            # 2020 a 2024 sur la longue fenetre. Minimum glissant de 7 jours par annee (rapport
            # median des annees), jours sous le dixieme centile observe de la periode (compte
            # total), volume d'aout-septembre. Le KGE ne voit pas l'etiage.
            _ann_t = pd.DatetimeIndex(temps)[_ok_t].year
            _sel = _ann_t >= int(debut_eval)
            if mesure is not None:
                _sel = _sel & mesure[_ok_t]
            _ke = df.assign(annee=_ann_t)[_sel].dropna(subset=["Qobs", "Qsim"])
            if len(_ke) > 60:
                _r7 = []
                for _a, _ka in _ke.groupby("annee"):
                    if len(_ka) > 60:
                        _q7o = _ka.Qobs.rolling(7, min_periods=7).mean().min()
                        _q7s = _ka.Qsim.rolling(7, min_periods=7).mean().min()
                        if _q7o > 0:
                            _r7.append(_q7s / _q7o)
                _seuil = _ke.Qobs.quantile(0.10)
                _jo = int((_ke.Qobs < _seuil).sum()); _js = int((_ke.Qsim < _seuil).sum())
                _as = _ke[_ke.mois.isin((8, 9))]
                _vol = _as.Qsim.sum() / max(_as.Qobs.sum(), 1e-9)
                _pl = int(_ke.annee.min()), int(_ke.annee.max())
                print(f"  etiage {_pl[0]}-{_pl[1]}, jours mesures : Q7min sim/obs {np.median(_r7) if _r7 else float('nan'):.2f} (mediane de {len(_r7)} an(s)) | jours sous le Q90 observe sim {_js} contre obs {_jo} | volume aout-sept sim/obs {_vol:.2f}", flush=True)
            # REPONSE AUX PLUIES D'ETE (2026-09-30) : pour chaque jour de juin a septembre
            # d'au moins 10 mm, hausse du debit sur les trois jours suivants rapportee a la
            # pluie, observee et simulee, mediane des evenements. Dit si le modele repond aux
            # orages d'ete, ce que l'etiage trop plat suggere qu'il ne fait pas (R239).
            _qo_s = df.Qobs; _qs_s = df.Qsim; _p_s = df.P
            _ev = (_p_s >= 10.0) & df.mois.isin((6, 7, 8, 9))
            _dqo = (_qo_s.shift(-1).rolling(3).max().shift(-2) - _qo_s.shift(1)).clip(lower=0.0)
            _dqs = (_qs_s.shift(-1).rolling(3).max().shift(-2) - _qs_s.shift(1)).clip(lower=0.0)
            _ok_e = _ev & _dqo.notna() & _dqs.notna()
            if int(_ok_e.sum()) >= 8:
                _ro = (_dqo / _p_s)[_ok_e]; _rs = (_dqs / _p_s)[_ok_e]
                print(f"  reponse aux pluies d'ete (>= 10 mm, {int(_ok_e.sum())} evenements) : hausse sur 3 jours / pluie, obs {_ro.median():.3f}, sim {_rs.median():.3f} | rapport sim/obs {(_rs.median() / max(_ro.median(), 1e-6)):.2f}", flush=True)
            # COMPOSITE D'ORAGE D'ETE (2026-09-30) : moyenne sur les memes evenements, du jour
            # precedent au troisieme jour suivant, de la pluie, de l'apport au sol, des trois
            # productions et des debits. Dit ou va la pluie d'un orage dans le modele.
            if int(_ok_e.sum()) >= 8:
                _pos_e = np.flatnonzero(_ok_e.to_numpy())
                _cols_e = [c for c in ("P", "apport", "surf", "hypo", "nappe", "Qsim", "Qobs", "sat1", "ET") if c in df.columns]
                _lig = []
                for _dj in (-1, 0, 1, 2, 3):
                    _ix = [i + _dj for i in _pos_e if 0 <= i + _dj < len(df)]
                    _lig.append({"jour": _dj, **{c: float(np.nanmean(df[c].to_numpy()[_ix])) for c in _cols_e}})
                print("  composite d'orage d'ete, mm/j (moyenne des evenements) :", flush=True)
                print(pd.DataFrame(_lig).set_index("jour").round(3).to_string(), flush=True)
            if mesure is not None:
                # Meme correlation sur les seuls jours MESURES : l'hiver observe est surtout une
                # reconstruction, et ne peut pas juger la physique hivernale (registre, R98).
                df["mesure"] = mesure[_ok_t]
                _r = []
                for _nom, _ms in {**_sais, "nov-dec": (11, 12)}.items():
                    _k = df[df.mois.isin(_ms) & df.mesure][["Qobs", "Qsim"]].dropna()
                    _r.append(f"{_nom} {np.corrcoef(_k.Qobs, _k.Qsim)[0, 1]:.2f} ({len(_k)} j)" if len(_k) > 20 else f"{_nom} - ({len(_k)} j)")
                print("  correlation sur jours mesures : " + ", ".join(_r), flush=True)
                # Octobre a decembre, jours mesures, separes selon le gel du jour : dit si une
                # porte gagne par la physique du gel ou par un ecoulement rapide qui manque.
                _od = df[df.mois.isin((10, 11, 12)) & df.mesure].dropna(subset=["Qobs", "Qsim"])
                _r = []
                for _nom, _k in (("gel", _od[_od.gel_cm > 0]), ("sans gel", _od[_od.gel_cm <= 0])):
                    _r.append(f"{_nom} {np.corrcoef(_k.Qobs, _k.Qsim)[0, 1]:.2f} ({len(_k)} j, Qobs {_k.Qobs.mean():.2f}, Qsim {_k.Qsim.mean():.2f}, surf {_k.surf.mean():.2f})" if len(_k) > 20 else f"{_nom} - ({len(_k)} j)")
                print("  octobre-decembre mesures : " + ", ".join(_r), flush=True)
        return
    if os.environ.get("MEANDRE_BANC_MULT"):
        # PASSE AVANT A CHAMP MULTIPLIE (2026-09-28) : « K_c:1.5,C_f:0.8 » multiplie ces
        # champs sur tous les noeuds, evalue, et s'arrete. Repond sans entrainer a la
        # question : tel parametre peut-il corriger tel defaut.
        for _spec in os.environ["MEANDRE_BANC_MULT"].split(";"):
            m.spatial_encoder.multiplicateurs = {a.split(":")[0]: torch.tensor(float(a.split(":")[1]), device=dev) for a in _spec.split(",") if a} if _spec != "aucun" else None
            _evaluer(m, f"x {_spec}")
        return
    q0, ev = _evaluer(m, "zero epoque")

    if os.environ.get("MEANDRE_SONDE_GRADIENT") == "1":
        # SONDE DU GRADIENT (2026-09-26). Le gradient du KGE de l'annee d'entrainement, pris
        # a travers la simulation poursuivie depuis l'etat de mise en regime comme le fait le
        # trainer, vaut 1e23 a 1e24 sur le champ spatial : il EXPLOSE. On mesure ici (1) sa
        # croissance avec l'horizon de retropropagation, (2) les champs qui la portent, par
        # un multiplicateur unite par champ, (3) sa dependance au nombre de sous-pas.
        import dataclasses as _dc
        _o_all = torch.tensor(o, dtype=torch.float32, device=dev)

        def _kge_t(qs, ob):
            ok = torch.isfinite(ob)
            a, b = qs[ok], ob[ok]
            r = ((a - a.mean()) * (b - b.mean())).mean() / (a.std(unbiased=False) * b.std(unbiased=False))
            return 1 - torch.sqrt((r - 1) ** 2 + (a.mean() / b.mean() - 1) ** 2 + ((a.std() / a.mean()) / (b.std() / b.mean()) - 1) ** 2)

        if os.environ.get("MEANDRE_SONDE_JACOBIEN") == "1":
            # JACOBIEN D'UN JOUR de l'humidite du sol : d theta(t+1) / d theta(t), par noeud
            # (3 x 3, les noeuds sont independants dans la colonne). Un rayon spectral
            # superieur a un fait croitre le gradient d'un jour a l'autre.
            import dataclasses as _dc2
            os.environ["MEANDRE_NSUBSTEP"] = os.environ.get("MEANDRE_SONDE_NSUBSTEP", "16").split(",")[0]
            torch.manual_seed(int(os.environ.get("ETL_SEED", "1234")))
            mm = _construire()
            mm.eval()
            for jour in [int(x) for x in os.environ.get("MEANDRE_SONDE_JOURS", "30,60,90,120,150,200,250,300").split(",")]:
                t0 = tr_sl.start + jour
                with torch.no_grad():
                    _, st = mm.simulate(forcing=F[:t0], initial_state=HydroState.zeros(n, device=dev), graph=g, node_coords=coords, territorial=terr, withdrawals=WithdrawalData(net=w.net[:t0]), day_of_year=doy[:t0])
                th = [st.theta1.clone().requires_grad_(True), st.theta2.clone().requires_grad_(True), st.theta3.clone().requires_grad_(True)]
                st_g = _dc2.replace(st, theta1=th[0], theta2=th[1], theta3=th[2])
                _, st1 = mm.simulate(forcing=F[t0:t0 + 1], initial_state=st_g, graph=g, node_coords=coords, territorial=terr, withdrawals=WithdrawalData(net=w.net[t0:t0 + 1]), day_of_year=doy[t0:t0 + 1], poursuivre_etat=True)
                out = [st1.theta1, st1.theta2, st1.theta3]
                J = torch.zeros(n, 3, 3, device=dev)
                for i in range(3):
                    gi = torch.autograd.grad(out[i].sum(), th, retain_graph=True, allow_unused=True)
                    for k2 in range(3):
                        if gi[k2] is not None:
                            J[:, i, k2] = gi[k2]
                rho = torch.linalg.eigvals(J).abs().max(dim=1).values
                sp0 = mm.spatial_encoder(coords, terr.data)
                sat3 = (st.theta3 / sp0.porosity_3.detach()).cpu().numpy() if hasattr(sp0, "porosity_3") else None
                r = rho.detach().cpu().numpy()
                pire = int(np.nanargmax(r))
                print(f"  sonde : jour {jour:3d} ({str(temps[t0].date())}) | rayon spectral median {np.nanmedian(r):.3f}, 90e centile {np.nanpercentile(r, 90):.3f}, max {np.nanmax(r):.3e} ; noeuds au-dessus de 1 : {int((r > 1.0001).sum())} sur {n} | pire noeud {pire} : theta {float(st.theta1[pire]):.3f} {float(st.theta2[pire]):.3f} {float(st.theta3[pire]):.3f}" + (f", saturation couche 3 {sat3[pire]:.3f}" if sat3 is not None else "") + f" | diag {J[pire].diagonal().detach().cpu().numpy().round(3)}", flush=True)
            return
        for nsub in [int(x) for x in os.environ.get("MEANDRE_SONDE_NSUBSTEP", "16,64").split(",")]:
            os.environ["MEANDRE_NSUBSTEP"] = str(nsub)
            torch.manual_seed(int(os.environ.get("ETL_SEED", "1234")))
            mm = _construire()
            if os.environ.get("MEANDRE_SONDE_CHARGER"):
                # Sonde sur un modele entraine (2026-10-02) : le signe dit ou l'optimiseur
                # pousserait chaque champ depuis ce point, et non depuis l'initialisation.
                mm.load(os.environ["MEANDRE_SONDE_CHARGER"])
                print(f"  sonde : point de reprise {os.path.basename(os.environ['MEANDRE_SONDE_CHARGER'])}", flush=True)
            mm.eval()
            with torch.no_grad():
                sp0 = mm.spatial_encoder(coords, terr.data)
                noms = [f.name for f in _dc.fields(sp0) if torch.is_tensor(getattr(sp0, f.name)) and getattr(sp0, f.name).ndim >= 1 and getattr(sp0, f.name).shape[0] == n]
                _, st0 = mm.simulate(forcing=F[:tr_sl.start], initial_state=HydroState.zeros(n, device=dev), graph=g, node_coords=coords, territorial=terr, withdrawals=WithdrawalData(net=w.net[:tr_sl.start]), day_of_year=doy[:tr_sl.start])
            for H, _var in [(int(h), v) for v in os.environ.get("MEANDRE_SONDE_DETACHE", "aucun").split(",") for h in os.environ.get("MEANDRE_SONDE_HORIZONS", "30,90,180,365").split(",")]:
                os.environ["MEANDRE_CASCADE_DETACHE"] = "1" if _var == "cascade" else "0"
                os.environ["MEANDRE_DETACHE_JOUR"] = "" if _var in ("aucun", "cascade") else _var.replace("+", ",")
                sl = slice(tr_sl.start, tr_sl.start + H)
                mult = {nm: torch.ones((), device=dev, requires_grad=True) for nm in noms}
                mm.spatial_encoder.multiplicateurs = mult
                # Chaque horizon repart du meme etat interne de mise en regime.
                with torch.no_grad():
                    _, st0 = mm.simulate(forcing=F[:tr_sl.start], initial_state=HydroState.zeros(n, device=dev), graph=g, node_coords=coords, territorial=terr, withdrawals=WithdrawalData(net=w.net[:tr_sl.start]), day_of_year=doy[:tr_sl.start])
                Q, _ = mm.simulate(forcing=F[sl], initial_state=st0, graph=g, node_coords=coords, territorial=terr, withdrawals=WithdrawalData(net=w.net[sl]), day_of_year=doy[sl], poursuivre_etat=True)
                k = _kge_t(Q[:, s["exutoire"]], _o_all[sl])
                # CIBLE DE LA SONDE (2026-10-02) : « kge » (1 - KGE, defaut historique) ou
                # « etiage » (ecart absolu des logarithmes sous le 30e centile observe, le terme
                # de la perte). Le signe dit ou l'optimiseur poussera : negatif, augmenter le
                # champ fait baisser la cible.
                _cible = os.environ.get("MEANDRE_SONDE_CIBLE", "kge")
                if _cible == "etiage":
                    from meandre.training.loss import differentiable_etiage_loss
                    _ob = _o_all[sl]
                    _okk = torch.isfinite(_ob)
                    _L = differentiable_etiage_loss(_ob[_okk], Q[:, s["exutoire"]][_okk])
                elif _cible == "biais":
                    _ob = _o_all[sl]
                    _okk = torch.isfinite(_ob)
                    _L = torch.abs(Q[:, s["exutoire"]][_okk].mean() / _ob[_okk].mean() - 1.0)
                else:
                    _L = 1 - k
                _L.backward()
                gm = {nm: float(t.grad.double()) if t.grad is not None else 0.0 for nm, t in mult.items()}
                mm.spatial_encoder.multiplicateurs = None
                haut = sorted(gm.items(), key=lambda kv: -abs(kv[1]))[:8]
                print(f"  sonde : cible {_cible} {float(_L):.3f}, {nsub} sous-pas, horizon {H:3d} j, KGE {float(k):.3f} | plus forts : " + ", ".join(f"{a} {b:+.2e}" for a, b in haut) + f" | nuls {sum(1 for v in gm.values() if v == 0)} sur {len(gm)}", flush=True)
                _vus = [x for x in os.environ.get("MEANDRE_SONDE_CHAMPS", "K_sat_1,K_sat_2,K_sat_3,porosity_2,porosity_3,Z2,Z3,K_c,C_f").split(",") if x in gm]
                print("    champs suivis : " + ", ".join(f"{a} {gm[a]:+.2e}" for a in _vus), flush=True)
        return

    # NORMALISATION PAR STATION (2026-09-13). L'ecart quadratique se calcule en metres
    # cubes par seconde au carre : sur ce sous-bassin il vaut environ 147 quand tous les
    # autres termes valent moins de deux, si bien qu'il emportait 95 % de la perte quel
    # que soit son poids affiche. Le pilote regional divise chaque station par la variance
    # observee de ses debits d'entrainement, ce qui rend le terme sans dimension et
    # d'ordre un ; le banc ne le faisait pas, et minimisait donc autre chose que la region
    # qu'il est cense representer. Mesure sur un entrainement regional reel, la perte se
    # repartit alors en 44 % de Kling-Gupta, 26 % d'ecart quadratique logarithmique, 25 %
    # de pics, 15 % de biais de volume, 11 % d'evapotranspiration et 3 % d'ecart
    # quadratique brut.
    _qtr = q_obs[tr_sl]
    _svar = torch.ones(_qtr.shape[1], dtype=torch.float32, device=dev)
    for _i in range(_qtr.shape[1]):
        _mk = torch.isfinite(_qtr[:, _i])
        if int(_mk.sum()) > 30:
            _svar[_i] = _qtr[_mk, _i].var()
    _svar = torch.clamp(_svar, min=1e-6)
    # Seuil du terme de pics : troisieme quartile des debits observes d'entrainement par
    # station, comme dans le pilote regional.
    _pthr = torch.full((_qtr.shape[1],), float("inf"), dtype=torch.float32, device=dev)
    for _i in range(_qtr.shape[1]):
        _mk = torch.isfinite(_qtr[:, _i])
        if int(_mk.sum()) > 100:
            _pthr[_i] = torch.quantile(_qtr[_mk, _i], 0.75)
    print(f"  normalisation par station : variance observee de {float(_svar.min()):.1f} "
          f"a {float(_svar.max()):.1f} (m3/s)^2 | seuil de pics {float(_pthr.min()):.1f} m3/s",
          flush=True)

    if aux and et_obs is not None:
        # Poids de la recette du socle (gasp-v4 [loss] + ETL_WET=0.4), sans GRACE.
        # per_station=True est OBLIGATOIRE (trouve le 2026-09-04 a 14 h 35) : le defaut
        # de HydroLoss est la branche « pooled », qui calcule le KGE sur le seul bloc
        # courant SANS l'historique detache. Tous les essais du banc jusqu'ici ont donc
        # optimise un KGE de quinze ou quarante-cinq jours, la faute meme que R67
        # corrige ; slso.py passe per_station=True, le banc ne le faisait pas.
        loss_fn = HydroLoss(w_kge=float(w_kge), w_pbias=float(w_pbias), w_mse=float(w_mse),
                            w_nse=0.0, w_nrmse=0.0, w_dq=float(w_dq), w_fdc_bas=float(w_fdc), w_etiage=float(os.environ.get("MEANDRE_BANC_W_ETIAGE", "0")),
                            w_dq_log=float(w_dq_log), station_var=_svar,
                            w_log_nse=0.0, w_log_mse=float(w_log_mse),
                            w_peak=float(w_peak),
                            peak_threshold=_pthr if float(w_peak) > 0 else None,
                            w_et=float(w_et), w_nappe=float(w_nappe), per_station=True,
                            # TENDANCE, pas niveau (R24, socle.toml et_mode = "anomaly") :
                            # MOD16 donne la forme de l'ET, jamais son volume. Le banc
                            # laissait le defaut « level » jusqu'a 15 h 30 le 2026-09-04,
                            # et un essai a conclu a tort que MOD16 vidait la riviere.
                            # MEANDRE_BANC_ET_MODE (2026-09-28) : « level » ou « bassin » pour
                            # rejuger, sur le schema de sol corrige, si MOD16 peut porter le
                            # NIVEAU de l'ET et absorber le surplus de precipitation de CaSR.
                            et_mode=os.environ.get("MEANDRE_BANC_ET_MODE", "anomaly"))
        print(f"  perte : KGE {float(w_kge):.2f} + biais {float(w_pbias):.2f} + MSE {float(w_mse):.2f}"
              f" + MSE log {float(w_log_mse):.2f} + pics {float(w_peak):.2f}"
              f" + ET MOD16 {float(w_et):.2f} en mode {os.environ.get("MEANDRE_BANC_ET_MODE", "anomaly")} + dQ {float(w_dq):.2f}"
              f" + soutien d'etiage {float(w_fdc):.2f} + etiage direct {float(os.environ.get('MEANDRE_BANC_W_ETIAGE', '0')):.2f}", flush=True)
    else:
        # Sans cible MOD16 : meme perte, terme d'ET en moins. Le 2026-09-05 cette branche
        # ignorait w_pbias, w_mse et w_dq_log, si bien qu'un balayage de dosage a rendu
        # trois resultats identiques sans que rien ne le signale.
        loss_fn = HydroLoss(w_kge=float(w_kge), w_pbias=float(w_pbias), w_mse=float(w_mse),
                            w_nse=0.0, w_nrmse=0.0, station_var=_svar,
                            w_dq=float(w_dq), w_fdc_bas=float(w_fdc), w_etiage=float(os.environ.get("MEANDRE_BANC_W_ETIAGE", "0")), w_dq_log=float(w_dq_log),
                            w_log_nse=0.0, w_log_mse=float(w_log_mse),
                            w_peak=float(w_peak),
                            peak_threshold=_pthr if float(w_peak) > 0 else None,
                            per_station=True)
        print(f"  perte SANS cible MOD16 : KGE {float(w_kge):.2f} + biais {float(w_pbias):.2f}"
              f" + MSE {float(w_mse):.2f} + dQ {float(w_dq):.2f} + dQ log {float(w_dq_log):.2f}"
              f" + soutien d'etiage {float(w_fdc):.2f} + etiage direct {float(os.environ.get('MEANDRE_BANC_W_ETIAGE', '0')):.2f}", flush=True)
    # warmup_epochs=0 : le defaut de cinq epoques de rechauffement rendait un essai
    # court entierement nul (cinq pas d'Adam a taux presque nul).
    tconf = TrainingConfig(n_epochs=epoques, lr=lr, chunk_steps=int(chunk), tbptt_steps=365,
                           grad_clip=1.0, w_prior=0.005, w_latent_reg=0.0,
                           # Choix du point de reprise (2026-10-01) : par la PERTE de debit sur
                           # la validation des que l'etiage est dans la perte, sinon le KGE seul
                           # choisissait et l'etiage ne comptait pas dans le modele retenu.
                           # MEANDRE_BANC_CHOIX=kge_median ou val_loss force le choix.
                           best_metric=("nll" if quantile else os.environ.get("MEANDRE_BANC_CHOIX", "val_loss" if (float(os.environ.get("MEANDRE_BANC_W_ETIAGE", "0")) > 0 or float(w_fdc) > 0) else "kge_median")),
                           autopilot=False, warmup_epochs=0)
    print(f"  choix du point de reprise : {tconf.best_metric}", flush=True)
    ck = f"{_p.DATA_ROOT}/quebec/sousbassin/best-{reg}-{station}{tag}.pt"
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    if charger:
        # Depart a chaud : le socle vient d'un entrainement anterieur. Obligatoire pour la
        # phase probabiliste, qui n'apprend qu'une enveloppe autour d'un debit deja fixe.
        m.load(charger)
        print(f"  socle charge depuis {os.path.basename(charger)}", flush=True)
        if os.environ.get("MEANDRE_CHAMP_GELE"):
            # Apres chargement, le gel par poids nuls ne tient plus : on fige les CARTES
            # chargees, capturees au premier appel du champ.
            m.spatial_encoder.figer_au_prochain_appel = {x for x in os.environ["MEANDRE_CHAMP_GELE"].split(",") if x}
            print(f"  champ spatial : cartes chargees figees {os.environ['MEANDRE_CHAMP_GELE']}", flush=True)
    elif quantile:
        raise SystemExit("phase quantile sans --charger : une enveloppe autour d'un modele "
                         "non entraine ne veut rien dire")
    if quantile:
        # PHASE PROBABILISTE. Le socle est GELE : la mediane reste exactement le debit du
        # point de reprise charge, donc aucun score deterministe ne bouge, et seule
        # l'enveloppe s'apprend. Sans gel, un modele gonfle son incertitude pour masquer
        # un biais qu'il devrait corriger.
        _libres = 0
        for _nn, _pp in m.named_parameters():
            _pp.requires_grad = "quantile_head" in _nn
            if _pp.requires_grad:
                _libres += _pp.numel()
        loss_fn = HydroLoss(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0,
                            w_log_nse=0.0, w_log_mse=0.0, w_quantile=1.0, per_station=True)
        print(f"  phase quantile : socle gele, {_libres:,} parametres libres (tete K=6), "
              f"perte de pinball seule", flush=True)
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
    # RECETTE DU PILOTE (2026-10-01). Le banc relisait les reglages un par un, sous ses propres
    # noms, et a tourne une semaine sans la neige du socle (R254). ETL_CONFIG pose la MEME
    # recette que le pilote, par la meme fonction, sans ecraser une variable deja posee.
    if os.environ.get("ETL_CONFIG"):
        import tomllib
        from meandre.utils.recette import appliquer_recette
        with open(os.environ["ETL_CONFIG"], "rb") as _f:
            _rec = tomllib.load(_f).get("recette")
        _pos = appliquer_recette(_rec)
        print(f"  recette du pilote {os.path.basename(os.environ['ETL_CONFIG'])} : {len(_pos)} reglage(s) poses depuis le fichier", flush=True)
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
    ap.add_argument("--fonte-diurne", action="store_true",
                    help="degre-jour integrant le cycle diurne au lieu de la moyenne")
    ap.add_argument("--seuil", type=float, default=None)
    ap.add_argument("--debut", type=int, default=None)
    ap.add_argument("--sol", choices=["complet", "sauf_ks", "libre"], default=None)
    ap.add_argument("--sans-aquifere", action="store_true")
    ap.add_argument("--entrainer", type=int, default=0, metavar="EPOQUES")
    ap.add_argument("--device", default=None, help="cuda ou cpu (defaut : cuda si dispo)")
    ap.add_argument("--tag", default="", help="suffixe du point de reprise et du run")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--quantile", action="store_true",
                    help="phase probabiliste : socle GELE, seule la tete de quantiles apprend")
    ap.add_argument("--w-kge", type=float, default=1.0)
    ap.add_argument("--w-pbias", type=float, default=0.5)
    ap.add_argument("--w-mse", type=float, default=0.1)
    ap.add_argument("--w-dq", type=float, default=0.0,
                    help="rapport des ecarts-types des variations journalieres (punit plateau ET nervosite)")
    ap.add_argument("--w-dq-log", type=float, default=0.0,
                    help="rapport des variations journalieres en espace log : voit la platitude d etiage")
    ap.add_argument("--w-log-mse", type=float, default=0.0,
                    help="ecart quadratique sur le logarithme des debits ; la recette pose 0,3")
    ap.add_argument("--w-peak", type=float, default=0.0,
                    help="terme de pics au-dela du troisieme quartile ; la recette pose 0,5")
    ap.add_argument("--w-fdc", type=float, default=0.0,
                    help="soutien d'etiage Q20/Q50 (contraint le chemin de l'eau)")
    ap.add_argument("--w-et", type=float, default=0.4,
                    help="poids du terme MOD16 (0 = meme perte sans le terme d ET)")
    ap.add_argument("--w-nappe", type=float, default=0.0,
                    help="poids du terme des niveaux de nappe ; part plafonnee par ETL_NAPPE_PART_MAX")
    ap.add_argument("--nappe-valid", type=float, default=0.5,
                    help="part des puits tenue de cote pour la validation independante")
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
                  sol=(None if a.sol == "libre" else (a.sol or "sauf_ks")), aquifere=not a.sans_aquifere,
                  kge_continu=not a.ancienne_boucle, etat_continu=not a.ancienne_boucle,
                  device=a.device, tag=a.tag, pas_par_bloc=not a.pas_par_epoque, lr=a.lr,
                  amorce=a.amorce, aux=not a.kge_seul,
                  debut_train=2012, fin_train=2012, fin_val=2013, debut_eval=2013,
                  fin_charge=2013, substeps=int(os.environ.get("MEANDRE_BANC_NSUBSTEP", "16")), chunk=a.chunk, w_et=a.w_et,
                  w_kge=a.w_kge, w_pbias=a.w_pbias, w_mse=a.w_mse, w_dq=a.w_dq, w_fdc=a.w_fdc,
                  w_dq_log=a.w_dq_log, quantile=a.quantile, charger=a.charger,
                  w_log_mse=a.w_log_mse, w_peak=a.w_peak,
                  w_nappe=a.w_nappe, nappe_valid=a.nappe_valid)
        return
    if a.entrainer:
        entrainer(a.region, a.station, epoques=a.entrainer,
                  sol=(None if a.sol == "libre" else (a.sol or "sauf_ks")), aquifere=not a.sans_aquifere,
                  kge_continu=not a.ancienne_boucle, etat_continu=not a.ancienne_boucle,
                  device=a.device, tag=a.tag, pas_par_bloc=not a.pas_par_epoque, lr=a.lr,
                  amorce=a.amorce, aux=not a.kge_seul,
                  chunk=a.chunk, w_et=a.w_et, w_kge=a.w_kge, w_pbias=a.w_pbias,
                  w_mse=a.w_mse, w_dq=a.w_dq, w_fdc=a.w_fdc, w_dq_log=a.w_dq_log,
                  quantile=a.quantile, charger=a.charger,
                  w_log_mse=a.w_log_mse, w_peak=a.w_peak,
                  w_nappe=a.w_nappe, nappe_valid=a.nappe_valid)
        return
    if a.simuler:
        rapport(a.region, a.station, ancrer=not a.sans_ancrage,
                kc=a.kc, kmusk=a.kmusk, annees=a.annees, melt_saison=a.fonte,
                seuil_neige=a.seuil, debut=a.debut, sol=a.sol,
                aquifere=not a.sans_aquifere, charger=a.charger,
                melt_diurne=a.fonte_diurne)
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
