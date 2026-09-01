"""LE DOMAINE : les quatorze caches de plateforme fondus en UN seul territoire.

Motif (Essi, 2026-08-25). `joint.py` faisait tourner les plateformes en ROTATION, une
boucle de simulation par territoire et par epoch. Mesure du run provincial de la veille :
5 h 36 de temps ecoule, 19 403 s de CPU sur un seul coeur, GPU a 0-1 %, epoch 0 pas
termine. Le modele est limite par le LANCEMENT de noyaux, pas par le calcul, donc le
cout est gouverne par le NOMBRE de boucles et non par leur largeur : quatorze plateformes
coutent quatorze fois neuf mille pas de temps, alors que leurs 25 656 troncons tiendraient
dans un seul pas vectorise.

Les reseaux sont DISJOINTS -- aucune arete ne traverse une frontiere de plateforme --
donc leur union est un graphe bloc-diagonal, topologiquement trivial : on decale les
indices de noeuds et on concatene. Le tri topologique fusionne aussi ses NIVEAUX, si
bien que la profondeur du domaine vaut celle de la plateforme la plus profonde et non
leur somme.

C'est enfin la consigne de fond appliquee au calcul : « tout chargeur fusionne les caches
par region en UN domaine des la lecture, et rien en aval ne porte de dimension region ».
La plateforme ne survit ici que comme une TRANCHE d'indices, gardee pour deux usages
legitimes : rapporter la tenue de cote par territoire contre la flotte gen1, et agreger
GRACE, qui est un observable a l'echelle du bassin et non du troncon.

MEMOIRE. Le forcage provincial pese 9131 x 25 656 x 6 x 4 o = 5.6 Go, soit les deux tiers
d'une carte de 8 Go, et il ne laissait plus de place aux activations d'un chunk couvrant
tous les troncons a la fois. Il reste donc en RAM hote et ne monte au GPU que la tranche
du chunk courant (~27 Mo), ce qui rend au calcul les 5.6 Go. Les prelevements, eux,
passent en representation creuse : ils ne touchent qu'une petite fraction des troncons.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from joint_data import load_region
from meandre.routing.graph import RiverGraph
from meandre.spatial.territorial import TerritorialFeatures
from meandre.training.loss import HydroLoss
from meandre.training.trainer import TrainingData


class HostSeries:
    """Serie (T, ...) hors GPU, tranchee vers le GPU a la demande.

    Se substitue a un tenseur partout ou le code ne fait que TRANCHER (`data.forcing[sl]`,
    `data.forcing[sl, :, 0]`), ce qui est le seul usage dans le trainer et le modele.
    Le gain n'est pas cosmetique : c'est ce qui permet au chunk de couvrir les 25 656
    troncons a la fois au lieu de 3 900.

    SUR DISQUE quand `host` est un memmap numpy (remarque d'Essi, 2026-08-25 : la RAM
    de la machine saturait). Le forcage provincial pese 5.6 Go ; le garder en RAM le
    deplacait simplement du GPU vers la memoire vive, ou il concurrencait Hydrotel et
    le reste. Sur memmap, seules les pages du chunk courant (~55 Mo) sont residentes,
    et le cache de pages du systeme garde les plus chaudes sans qu'on ait a le gerer.
    """

    def __init__(self, host, device: str) -> None:
        self._host = host
        self._device = device
        self._np = not isinstance(host, torch.Tensor)

    def __getitem__(self, key):
        v = self._host[key]
        if self._np:
            v = torch.from_numpy(np.ascontiguousarray(v))
        return v.to(self._device, non_blocking=True)

    def __len__(self) -> int:
        return int(self._host.shape[0])

    @property
    def shape(self):
        return self._host.shape

    @property
    def ndim(self) -> int:
        return int(self._host.ndim)

    @property
    def dtype(self):
        return torch.float32 if self._np else self._host.dtype

    @property
    def device(self):
        return torch.device(self._device)


def _memmap_forcing(parts, names, device):
    """Concatene le forcage des plateformes dans un fichier sur DISQUE, puis le relit
    en memmap. Ecrit plateforme par plateforme pour ne jamais tenir la province entiere
    en RAM, et libere chaque tenseur source au passage. Le fichier est mis en cache :
    tant que la liste des plateformes et la forme ne changent pas, il est reutilise."""
    T = parts[0]["train_data"].forcing.shape[0]
    C = parts[0]["train_data"].forcing.shape[2]
    N = sum(p["n_nodes"] for p in parts)
    rep = f"{_mpaths_root()}/quebec/cache"
    os.makedirs(rep, exist_ok=True)
    # LA VARIANTE DE FORCAGE DOIT ENTRER DANS LA CLE. Sans elle, changer
    # JOINT_FX_SUFFIX relit silencieusement le cache de l'autre variante : meme
    # liste de regions, memes dimensions, meme nom de fichier. Mesure le 2026-08-31,
    # c'est ce qui faisait rendre 0.68 et 0.36 au MEME modele sur les MEMES stations,
    # le forcage differant de 70 % de son echelle entre les deux chargements. Trois
    # jours de diagnostic sont partis la-dedans.
    _sfx = os.environ.get("JOINT_FX_SUFFIX", "-defaut") or "-defaut"
    cle = "-".join(names) + _sfx + f"_{T}x{N}x{C}"
    if len(cle) > 120:                      # noms de fichiers Windows
        import hashlib
        cle = f"dom{len(names)}_{T}x{N}x{C}_{hashlib.md5(cle.encode()).hexdigest()[:8]}"
    f = f"{rep}/forcage-{cle}.npy"
    if not os.path.exists(f):
        mm = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(T, N, C))
        o = 0
        for p in parts:
            v = p["train_data"].forcing
            mm[:, o:o + v.shape[1], :] = v.numpy()
            o += v.shape[1]
            mm.flush()
        del mm
        print(f"[domaine] forcage ecrit sur disque : {f} ({T * N * C * 4 / 1e9:.1f} Go)")
    else:
        print(f"[domaine] forcage repris du cache disque : {os.path.basename(f)}")
    return HostSeries(np.load(f, mmap_mode="r"), device)


class SparseWithdrawals:
    """Prelevements et rejets nets, stockes sur les seuls troncons concernes.

    Meme interface que WithdrawalData vue par le modele (`net_withdrawal`,
    `gw_withdrawal`). En dense, la province demanderait 0.94 Go par champ pour un
    tenseur presque entierement nul.
    """

    def __init__(self, cols, vals, cols_gw, vals_gw, n_reaches, device):
        self._cols = cols.to(device)
        self._vals = vals.to(device)
        self._cols_gw = cols_gw.to(device)
        self._vals_gw = vals_gw.to(device)
        self.n_reaches = int(n_reaches)
        self._device = device

    def _scatter(self, cols, vals, t):
        out = torch.zeros(self.n_reaches, dtype=vals.dtype, device=self._device)
        if cols.numel():
            out[cols] = vals[t]
        return out

    def net_withdrawal(self, t: int):
        return self._scatter(self._cols, self._vals, t)

    def gw_withdrawal(self, t: int):
        return self._scatter(self._cols_gw, self._vals_gw, t)


def _compress(dense: torch.Tensor):
    """(T, N) dense -> (colonnes non nulles, valeurs sur ces colonnes)."""
    if dense is None:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, 0)
    nz = (dense != 0).any(dim=0).nonzero(as_tuple=True)[0]
    return nz.cpu(), dense[:, nz].cpu()


def load_domain(names: list[str], lcfg: dict, device: str = "cuda"):
    """Charge les plateformes et les fond en un domaine unique.

    Retourne le meme contrat qu'une region de `joint_data.load_region`, plus la table
    des tranches par plateforme (`slices`) dont l'evaluation a besoin.
    """
    # Normalisation PROVINCIALE des attributs, sans quoi le noeud median de CHAQUE
    # plateforme arrive au NeRF avec des attributs ~0 et le champ ne peut pas les
    # differencier (diagnostic 2026-07-31, cause des cinq echecs conjoints). Dans un
    # domaine fusionne ce n'est plus une option : deux troncons voisins de part et
    # d'autre d'une frontiere recevraient des attributs incomparables.
    os.environ["JOINT_GLOBAL_NORM"] = "1"

    parts = [load_region(n, dict(lcfg), device="cpu") for n in names]
    sizes = [p["n_nodes"] for p in parts]
    offs = np.cumsum([0] + sizes)
    n_total = int(offs[-1])
    T = len(parts[0]["times"])

    # ── graphe bloc-diagonal ────────────────────────────────────────────────
    ei, ea, tt, lake = [], [], [], []
    for p, o in zip(parts, offs[:-1]):
        g = p["train_data"].graph
        if g.edge_index.numel():
            ei.append(g.edge_index + int(o))
            ea.append(g.edge_attr)
            tt.append(g.travel_time_days)
        lake.append(g.is_lake)
    edge_index = torch.cat(ei, dim=1) if ei else torch.zeros(2, 0, dtype=torch.long)
    graph = RiverGraph(
        edge_index=edge_index.to(device),
        edge_attr=(torch.cat(ea, dim=0) if ea else torch.zeros(0, 3)).to(device),
        # topo_order omis serait plus propre, mais le champ n'a pas de defaut : on
        # laisse Kahn le recalculer sur l'union et on le passe tel quel. Le controle
        # de coherence de RiverGraph le revalide de toute facon.
        topo_order=_kahn(edge_index, n_total).to(device),
        is_lake=torch.cat(lake).to(device),
        travel_time_days=(torch.cat(tt) if tt else torch.zeros(0, dtype=torch.long)).to(device),
    )

    # ── attributs, coordonnees ──────────────────────────────────────────────
    cols = parts[0]["territorial"].columns
    for p in parts[1:]:
        if p["territorial"].columns != cols:
            raise ValueError(f"{p['name']} : colonnes territoriales differentes du premier "
                             "cache -- la fusion melangerait des attributs sans rapport")
    # SCHEMA PHYSIQUE INEGAL entre plateformes, decouvert par la fusion (2026-08-25) :
    # douze caches portent quatre champs, ABIT et LABI en portent treize (milieux
    # humides, c_prod, ksat_bs). L'intersection est le seul choix sur : un champ absent
    # sur douze plateformes ne peut pas etre invente. Mais elle RETIRE a ABIT et LABI
    # des ancrages qu'ils avaient dans leur run regional, donc leurs scores fusionnes ne
    # sont plus directement comparables a gen1. C'est un artefact de PRODUCTION des
    # caches, de la meme famille que la region : a corriger a la source en rebatissant
    # les quatorze caches sur le meme schema, pas a masquer ici.
    phys_keys = set(parts[0]["territorial"].physical)
    union = set(parts[0]["territorial"].physical)
    for p in parts[1:]:
        phys_keys &= set(p["territorial"].physical)
        union |= set(p["territorial"].physical)
    perdus = sorted(union - phys_keys)
    if perdus:
        _qui = [p["name"] for p in parts if set(p["territorial"].physical) - phys_keys]
        print(f"[domaine] ATTENTION {len(perdus)} champs physiques presents seulement sur "
              f"{_qui} sont ECARTES : {perdus}")
    territorial = TerritorialFeatures(
        data=torch.cat([p["territorial"].data for p in parts]).to(device),
        columns=cols,
        physical={k: torch.cat([p["territorial"].physical[k] for p in parts]).to(device)
                  for k in sorted(phys_keys)},
    )
    node_coords = torch.cat([p["node_coords"] for p in parts]).to(device)

    # ── stations : indices decales, colonnes concatenees ────────────────────
    st_idx, q_cols, st_ids, st_var, st_peak = [], [], [], [], []
    for p, o in zip(parts, offs[:-1]):
        td = p["train_data"]
        st_idx.append(td.station_idx + int(o))
        q_cols.append(p["train_data"].q_obs)
        st_ids += [f"{p['name']}:{s}" for s in p["station_ids"]]
        st_var.append(p["loss_fn"].station_var)
        pk = getattr(p["loss_fn"], "peak_threshold", None)
        st_peak.append(pk if pk is not None
                       else torch.full((p["n_gauges"],), float("inf")))
    station_idx = torch.cat(st_idx).to(device)
    q_obs = torch.cat(q_cols, dim=1).to(device)
    station_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    station_mask[station_idx] = True

    # ── forcage, prelevements ───────────────────────────────────────────────
    forcing = _memmap_forcing(parts, names, device)
    # Les copies par plateforme ont fait leur office : les garder doublait l'empreinte
    # (5.6 Go de plus) pour rien, puisque tout passe desormais par le memmap.
    for p in parts:
        p["train_data"].forcing = None
        p["val_data"].forcing = None
    wc, wv = _compress(torch.cat([_dense_w(p, "net") for p in parts], dim=1))
    gc, gv = _compress(torch.cat([_dense_w(p, "net_gw") for p in parts], dim=1))
    withdrawals = SparseWithdrawals(wc, wv, gc, gv, n_total, device)

    # ── cibles auxiliaires ──────────────────────────────────────────────────
    # ET et SWE restent des tenseurs DENSES sur le GPU : le trainer ne fait pas que les
    # trancher, il les centre (`_center_et` balaye toute la serie), donc l'astuce de la
    # serie hote ne s'y applique pas. Le budget le permet une fois le forcage sorti.
    et_obs = _cat_nodes([p["train_data"].et_obs for p in parts], sizes, T)
    if et_obs is not None:
        et_obs = et_obs.to(device)
    swe_obs = _cat_nodes([p["train_data"].swe_obs for p in parts], sizes, T)
    if swe_obs is not None:
        swe_obs = swe_obs.to(device)

    # GRACE : un observable de BASSIN. La colonne g de `tws_obs` est la serie de la
    # plateforme g, et `tws_group` dit a quel bassin appartient chaque troncon. Sans
    # ce groupement, la moyenne de stockage melangerait la Gaspesie et l'Abitibi pour
    # les confronter a une seule serie satellitaire.
    tws_cols, tws_group = [], []
    for gi, (p, sz) in enumerate(zip(parts, sizes)):
        t = p["train_data"].tws_obs
        tws_cols.append(t if t is not None else torch.full((T,), float("nan")))
        tws_group.append(torch.full((sz,), gi, dtype=torch.long))
    tws_obs = torch.stack(tws_cols, dim=1).to(device)          # (T, G)
    tws_group = torch.cat(tws_group).to(device)

    # CanSWE : cibles par SITE, le noeud vise se decale comme les autres
    sm_obs, sm_node = [], []
    for p, o in zip(parts, offs[:-1]):
        td = p["train_data"]
        if td.swe_mass_obs is not None and td.swe_mass_node is not None:
            sm_obs.append(td.swe_mass_obs)
            sm_node.append(td.swe_mass_node + int(o))
    swe_mass_obs = torch.cat(sm_obs, dim=1).to(device) if sm_obs else None
    swe_mass_node = torch.cat(sm_node).to(device) if sm_node else None

    # ── perte unique ────────────────────────────────────────────────────────
    ref = parts[0]["loss_fn"]
    loss_fn = HydroLoss(
        w_nse=ref.w_nse, w_kge=ref.w_kge, w_pbias=ref.w_pbias, w_mse=ref.w_mse,
        w_nrmse=ref.w_nrmse, w_log_nse=ref.w_log_nse, w_log_mse=ref.w_log_mse,
        w_et=ref.w_et, et_mode=lcfg.get("et_mode", "level"),
        w_tws=ref.w_tws, w_tws_clim=ref.w_tws_clim, w_snow=ref.w_snow,
        w_swe_mass=ref.w_swe_mass if swe_mass_obs is not None else 0.0,
        w_peak=ref.w_peak, w_physics=ref.w_physics, w_residual=ref.w_residual,
        per_station=True, station_weights=None,
        station_var=torch.cat(st_var).to(device),
        peak_threshold=torch.cat(st_peak).to(device) if ref.w_peak > 0 else None,
    )

    train_sl = parts[0]["train_data"].train_slice
    val_sl = parts[0]["val_data"].val_slice
    day_of_year = parts[0]["train_data"].day_of_year.to(device)

    def mk(sl_):
        return TrainingData(
            forcing=forcing, q_obs=q_obs[sl_.start:], station_mask=station_mask,
            station_idx=station_idx, graph=graph, node_coords=node_coords,
            territorial=territorial, withdrawals=withdrawals, day_of_year=day_of_year,
            train_slice=sl_, val_slice=sl_,
            et_obs=_shift(et_obs, sl_), swe_obs=_shift(swe_obs, sl_),
            tws_obs=tws_obs[sl_.start:], tws_group=tws_group,
            swe_mass_obs=(swe_mass_obs[sl_.start:] if swe_mass_obs is not None else None),
            swe_mass_node=swe_mass_node,
        )

    # ── ANCRAGES DE PLATEFORME : occupation, milieux humides, fonte ─────────
    # PIEGE MAJEUR, decouvert par la fusion (2026-08-25). Ces champs ne sont PAS dans
    # les caches DuckDB : ils viennent des paquets Hydrotel et n'etaient charges que par
    # `etl_run.py`. `joint.py` ne les chargeait pas, donc tous les entrainements
    # conjoints tournaient sur un territoire sans foret, sans eau libre et sans milieu
    # humide -- tout en classe DECOUVERT, la plus fondante. Mesure sur OUTV a intrants
    # identiques et parametres figes : KGE aux jauges 0.482 sans, 0.749 avec.
    # Ce ne sont pas des boutons regionaux mais des DONNEES physiographiques par troncon,
    # simplement livrees par plateforme : elles se concatenent comme le reste.
    land_cover, melt_params, phenology = {}, {}, None
    try:
        from meandre.data.hydrotel_calib import (load_occupation_sol, load_milieux_humides,
                                                 load_melt_nodes, load_phenologie)
        from meandre.utils import paths as _mp
        lc_parts, mp_parts = [], []
        for p in parts:
            pl = f"{_mp.PLATFORMS_ROOT}/LN24HA/{p['name'].upper()}_LN24HA_2020"
            d = load_occupation_sol(pl, p["node_ids"], device="cpu")
            d.update(load_milieux_humides(pl, p["node_ids"], device="cpu"))
            lc_parts.append(d)
            mp_parts.append(load_melt_nodes(pl, p["node_ids"], device="cpu"))
            if phenology is None:
                phenology = load_phenologie(pl)
        # MILIEUX HUMIDES : presents sur DOUZE plateformes sur quatorze (seules MONT et
        # SLNO n'en declarent pas). L'intersection les supprimait donc pour TOUT LE MONDE
        # a cause de deux territoires -- exactement l'inverse de ce qu'il faut faire.
        # Un territoire qui n'en declare pas n'en a pas : zero est ici la valeur
        # PHYSIQUEMENT juste, pas un bouche-trou.
        WETLAND = ("wet_a_raw", "wet_dra_fr_raw", "wetdnor_raw", "wetdmax_raw",
                   "wet_vol_init_raw", "frac_raw")
        for k in sorted(set().union(*[set(d) for d in lc_parts])):
            presents = [d for d in lc_parts if k in d]
            if len(presents) == len(lc_parts):
                land_cover[k] = torch.cat([d[k] for d in lc_parts]).to(device)
            elif k in WETLAND:
                v = [(d[k] if k in d else torch.zeros(sz))
                     for d, sz in zip(lc_parts, sizes)]
                land_cover[k] = torch.cat(v).to(device)
                _abs = [p["name"] for p, d in zip(parts, lc_parts) if k not in d]
                print(f"[domaine] milieu humide {k} : absent de {_abs}, pose a ZERO "
                      "(pas de milieu humide declare)")
            else:
                # Coefficients de production et de sol : zero serait FAUX. On pose la
                # mediane provinciale des plateformes qui l'ont, et on le dit.
                med = float(torch.cat([d[k] for d in presents]).median())
                v = [(d[k] if k in d else torch.full((sz,), med))
                     for d, sz in zip(lc_parts, sizes)]
                land_cover[k] = torch.cat(v).to(device)
                _abs = [p["name"] for p, d in zip(parts, lc_parts) if k not in d]
                print(f"[domaine] {k} : absent de {_abs}, pose a la mediane "
                      f"provinciale {med:.4g} (HYPOTHESE, pas une mesure)")
        for k in sorted(set().union(*[set(d) for d in mp_parts])):
            if all(k in d for d in mp_parts):
                v = [d[k] for d in mp_parts]
                melt_params[k] = (torch.cat(v).to(device) if v[0].ndim else v[0].to(device))
        _f = land_cover.get("pct_forest", land_cover.get("f_forest"))
        print(f"[domaine] ancrages de plateforme : occupation {len(land_cover)} champs"
              + (f" (foret moyenne {float(_f.mean()):.2f})" if _f is not None else "")
              + f", fonte {len(melt_params)} champs")
    except Exception as exc:
        print(f"[domaine] ATTENTION ancrages de plateforme INDISPONIBLES ({exc}) : "
              "le territoire sera traite en classe DECOUVERT, ce qui coute ~0.27 de KGE")

    # ── ETP LINACRE et CHAMP k_gw : les deux autres ancrages de la flotte gen1 ──
    # Meme famille que l'occupation : des DONNEES par troncon livrees par plateforme
    # (Linacre) ou par un champ provincial deja estime (k_gw, GP sur les recessions de
    # 127 stations). Les omettre ne casse rien de visible, ca change juste la recette --
    # et comparer la province aux champions gen1 sous une AUTRE recette ne mesurerait
    # pas l'architecture. C'est l'erreur que l'occupation du sol a failli faire commettre.
    linacre, kgw = None, None
    try:
        from meandre.data.hydrotel_calib import load_linacre_nodes
        from meandre.utils import paths as _mp2
        # six champs par noeud : lat, altitude, T froide, T chaude, albedo, coefficient
        # d'optimisation regional (~0.4-0.5). Ils se concatenent comme les autres.
        cols_lin = None
        for p in parts:
            pl = f"{_mp2.PLATFORMS_ROOT}/LN24HA/{p['name'].upper()}_LN24HA_2020"
            v = load_linacre_nodes(pl, p["node_ids"], device="cpu")
            cols_lin = [[x] for x in v] if cols_lin is None else                 [c + [x] for c, x in zip(cols_lin, v)]
        linacre = tuple(torch.cat(c).to(device) for c in cols_lin)
        print(f"[domaine] ETP Linacre du projet : {len(linacre)} champs, coefficient "
              f"median {float(linacre[5].median()):.3f}")
    except Exception as exc:
        print(f"[domaine] Linacre indisponible ({exc}) : ETP McGuinness par defaut")
    try:
        import pandas as _pd
        _cf = _pd.read_parquet(f"{_mpaths_root()}/quebec/champ_kgw_QC.parquet")
        vals = []
        for p, sz in zip(parts, sizes):
            sub = _cf[_cf.region == p["name"]].sort_values("node_idx")
            if len(sub) != sz:
                raise ValueError(f"{p['name']} : {len(sub)} noeuds dans le champ k_gw "
                                 f"pour {sz} attendus")
            vals.append(torch.tensor(sub.k_gw.values, dtype=torch.float32))
        kgw = torch.cat(vals).to(device)
        print(f"[domaine] champ k_gw provincial : med {float(kgw.median()):.4f} | "
              f"q10-q90 {float(kgw.quantile(0.1)):.4f}-{float(kgw.quantile(0.9)):.4f}")
    except Exception as exc:
        print(f"[domaine] champ k_gw indisponible ({exc}) : k_gw laisse au NeRF")

    # ── SOL CALIBRE HYDROTEL, par noeud : le depart de la flotte gen1 ──────
    # Ce n'est PAS un ancrage qui fige le sol (la loi des ancrages l'interdit), c'est un
    # POINT DE DEPART : le champ est ajuste par regression sur les valeurs calibrees,
    # puis rendu entierement libre. Mesure du 2026-08-13 : le reseau reproduit le champ
    # a 2 % pres et sa dispersion passe de 0.0017 a 0.741 -- la capacite n'etait pas le
    # verrou, le point de depart l'etait. Decisif pour un run court : sans lui, quatre
    # epochs partent d'un champ plat au K_sat huit fois trop bas.
    soil = None
    try:
        from meandre.data.hydrotel_calib import load_calibrated_soil
        from meandre.utils import paths as _mp3
        cs_parts = []
        for p in parts:
            pl = f"{_mp3.PLATFORMS_ROOT}/LN24HA/{p['name'].upper()}_LN24HA_2020"
            cs_parts.append(load_calibrated_soil(pl, p["node_ids"], 0.15, device="cpu"))
        communs = set(cs_parts[0])
        for d in cs_parts[1:]:
            communs &= set(d)
        soil = {}
        for k in sorted(communs):
            v = [d[k] for d in cs_parts]
            if all(hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == sz
                   for x, sz in zip(v, sizes)):
                soil[k] = torch.cat(v).to(device)
        print(f"[domaine] sol calibre Hydrotel : {len(soil)} champs par noeud")
    except Exception as exc:
        print(f"[domaine] sol calibre indisponible ({exc}) : depart litterature")

    # FRACTION AGRICOLE BRUTE. Les attributs du champ ne la portent qu'en z-score, ce
    # qui ne peut pas servir de fraction de surface. Le parquet provincial la garde en
    # brut ; on la joint a l'occupation pour que le drainage agricole sache ou draîner.
    try:
        import pandas as _pd2
        _raw = _pd2.read_parquet(f"{_mpaths_root()}/quebec/territorial-raw-QC.parquet")
        _ag = []
        for p, sz in zip(parts, sizes):
            sub = _raw[_raw.region == p["name"]]
            if len(sub) != sz:
                raise ValueError(f"{p['name']} : {len(sub)} lignes pour {sz} noeuds")
            _ag.append(torch.tensor(sub["f_agriculture"].to_numpy(), dtype=torch.float32))
        land_cover["pct_agricole"] = torch.cat(_ag).to(device)
        print(f"[domaine] fraction agricole : moyenne {float(land_cover['pct_agricole'].mean()):.3f}"
              f" | q90 {float(land_cover['pct_agricole'].quantile(0.9)):.3f}")
    except Exception as exc:
        print(f"[domaine] fraction agricole indisponible ({exc}) : drainage agricole inerte")

    slices = {p["name"]: (int(a), int(b)) for p, a, b in zip(parts, offs[:-1], offs[1:])}
    print(f"[domaine] {len(parts)} plateformes fondues | {n_total} troncons | "
          f"{q_obs.shape[1]} jauges | {graph.n_edges} aretes | "
          f"forcage {forcing.shape[0]}x{forcing.shape[1]} en RAM hote")
    return dict(n_nodes=n_total, train_data=mk(train_sl), val_data=mk(val_sl),
                loss_fn=loss_fn, node_coords=node_coords, territorial=territorial,
                times=parts[0]["times"], station_ids=st_ids, slices=slices,
                n_gauges=q_obs.shape[1], land_cover=land_cover,
                melt_params=melt_params, phenology=phenology,
                linacre=linacre, kgw=kgw, soil=soil)


def _mpaths_root():
    from meandre.utils import paths as _p
    return _p.DATA_ROOT


def _kahn(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Tri topologique de l'union. Reimplemente ici car la fonction du module est
    privee et le graphe bloc-diagonal en est le cas le plus simple."""
    from meandre.routing.graph import _topological_sort
    return _topological_sort(edge_index, n)


def _dense_w(part, champ):
    w = part["train_data"].withdrawals
    v = getattr(w, champ, None)
    if v is None:
        v = torch.zeros_like(w.net)
    return v.cpu()


def _cat_nodes(tensors, sizes, T):
    """Concatene des cibles (T, n_noeuds) en tolerant les plateformes sans donnee."""
    if all(t is None for t in tensors):
        return None
    out = []
    for t, sz in zip(tensors, sizes):
        out.append(t.cpu() if t is not None else torch.full((T, sz), float("nan")))
    return torch.cat(out, dim=1)


def _shift(series, sl_):
    """Aligne une serie hote sur le debut de la tranche, comme le fait joint_data."""
    if series is None:
        return None
    if isinstance(series, HostSeries):
        return HostSeries(series._host[sl_.start:], series._device)
    return series[sl_.start:]
