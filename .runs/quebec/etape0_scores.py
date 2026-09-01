"""ETAPE 0, seconde moitie : les stations regularisees sont-elles simplement MAL SIMULEES ?

La premiere moitie (etape0_barrages.py) a mesure les PARAMETRES et n'a rien trouve : sur
les cinq champs reellement libres, l'ecart apparie est sous 1 % et quatre signes sur cinq
sont contraires a ce que la contamination predit.

Il reste une facon dont la regularisation pourrait couter quelque chose sans deformer le
champ : elle pourrait simplement faire RATER ces stations. Le modele ne se tordrait pas
pour les imiter, il echouerait dessus, et elles pollueraient la selection en tirant la
mediane vers le bas. Dans ce cas l'astuce rapide existe quand meme, sous une autre forme :
les ecarter de la selection.

Meme appariement que la premiere moitie : chaque station regularisee contre ses trois
temoins les plus proches en superficie sur la MEME plateforme, et on ne lit que l'ecart
intra-paire. Sans ca on mesurerait la difference de population, les bassins regularises
etant plus grands et concentres dans SLNO et OUTV.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")

BARRAGES = os.environ.get("BARRAGES_DATA", f"{_DATA_ROOT}/barrages/data")
CKPT = os.environ.get("ETAPE0_CKPT", f"{_DATA_ROOT}/quebec/runpod/best-province.pt")
SEUIL_JOURS = float(os.environ.get("ETAPE0_SEUIL", "15"))


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 365:
        return np.nan
    s, o = sim[m], obs[m]
    if s.std() < 1e-9 or o.std() < 1e-9:
        return np.nan
    r = float(np.corrcoef(s, o)[0, 1])
    b = float(s.mean() / o.mean())
    g = float((s.std() / s.mean()) / (o.std() / o.mean()))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))


def main():
    from domain_data import load_domain
    from meandre.model import HydroModel
    from meandre.utils.state import HydroState

    plats = (os.environ.get("ETAPE0_PLATEFORMES")
             or "slno,outv,mont,slso,gasp,sagu,cnda")
    noms = [p.strip() for p in plats.split(",") if p.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dom = load_domain(noms, {}, device=dev)
    td = dom["train_data"]
    terr = dom["territorial"]

    # POSER LES ANCRAGES DE PLATEFORME AVANT DE SIMULER. Sans eux la colonne recoit
    # une occupation du sol TOUTE NULLE et traite la province comme du sol nu : c'est
    # la dette #3 du registre, qui avait deja fait tourner tout un entrainement conjoint
    # sur un territoire sans foret. Premiere version de ce script : je construisais le
    # modele sans les poser, et l'avertissement de la colonne le disait dans le journal.
    modele = HydroModel(
        n_territorial=int(terr.n_features), n_nodes=int(dom["n_nodes"]),
        n_forcing=6, use_temporal=False, use_residual=False,
        param_mode="nerf", column_mode="hydrotel", et_mode="linacre",
        use_latent_codes=False, spatial_melt=True, use_aquifer=True,
        use_temperature=False,   # sinon la thermie desactive le routage par operateur
        predict_lake_params=True, routing_mode="operator-lagged").to(dev)
    if dom.get("land_cover"):
        modele.vertical_column.set_land_cover(dom["land_cover"])
    if dom.get("melt_params"):
        modele.vertical_column.set_melt_params(dom["melt_params"])
    if dom.get("phenology"):
        modele.vertical_column.set_phenology(dom["phenology"])
    if dom.get("linacre"):
        modele.vertical_column.set_linacre_params(*dom["linacre"])
        modele.vertical_column.etp_channel = None
    if dom.get("kgw") is not None:
        _o = modele.spatial_encoder.forward
        def _kgw(*a, _o=_o, _k=dom["kgw"], **kw):
            sp = _o(*a, **kw)
            sp.k_gw = _k
            return sp
        modele.spatial_encoder.forward = _kgw
    # Partage pluie-neige de la recette 1.0. Sans lui le seuil reste a 0 au lieu de
    # -0.8 en bulbe humide, ce qui change 35 % de la neige (R35/R37) : la fiche
    # d'execution du point de reprise le signale et annonce des scores FAUX.
    modele.vertical_column.split_mode = "wet_bulb"
    modele.vertical_column.t_neige_seuil = -0.8
    if dom.get("soil"):
        modele.vertical_column.set_calibrated_soil(dom["soil"])
    modele.load(CKPT)
    modele.eval()
    with torch.no_grad():
        Q, _ = modele.simulate(
            forcing=td.forcing[:],
            initial_state=HydroState.zeros(int(dom["n_nodes"]), device=dev),
            graph=td.graph, node_coords=td.node_coords, territorial=terr,
            withdrawals=td.withdrawals, day_of_year=td.day_of_year)

    times = pd.DatetimeIndex(dom["times"])
    sl = (times >= "2022-01-01") & (times <= "2024-12-31")
    q_obs = dom["val_data"].q_obs
    n = int(sl.sum())

    sr = pd.read_csv(f"{BARRAGES}/stations-regulees.csv")
    sr = sr[np.isfinite(sr.station_id)].copy()
    sr["sid"] = sr.station_id.astype(float).astype(int).astype(str)
    reg = dict(zip(sr.sid, sr.jours_de_debit))
    aire = dict(zip(sr.sid, sr.aire_km2))

    lignes = []
    for j, brut in enumerate(dom["station_ids"]):
        plat, _, s = str(brut).partition(":")
        s = str(int(float(s))) if s.replace(".", "").isdigit() else s
        if s not in reg:
            continue
        node = int(td.station_idx[j])
        o = q_obs[-n:, j].detach().cpu().numpy()
        sim = Q[sl][:, node].detach().cpu().numpy()
        k = kge(sim, o)
        if not np.isfinite(k):
            continue
        lignes.append({"plat": plat, "sid": s, "kge": k, "jours": float(reg[s]),
                       "aire": float(aire[s])})
    st = pd.DataFrame(lignes)
    st["regule"] = st.jours > SEUIL_JOURS
    print(f"\n[scores] {len(st)} stations notees | {int(st.regule.sum())} regularisees")
    print(f"  KGE median, regularisees : {st[st.regule].kge.median():.4f}")
    print(f"  KGE median, les autres   : {st[~st.regule].kge.median():.4f}")

    paires = []
    for _, r in st[st.regule].iterrows():
        c = st[(~st.regule) & (st.plat == r.plat)]
        if len(c) < 3:
            continue
        c = c.iloc[(np.log(c.aire) - np.log(r.aire)).abs().argsort()[:3]]
        paires.append((r, c, float(np.exp(np.abs(np.log(c.aire) - np.log(r.aire)).max()))))

    e = np.array([float(r.kge) - float(c.kge.median()) for r, c, _ in paires])
    fact = np.array([f for _, _, f in paires])
    from scipy import stats as _st
    print(f"\n[scores] {len(e)} paires. Ecart APPARIE de KGE, station regularisee moins "
          f"la mediane de ses temoins :")
    print(f"  median {np.median(e):+.4f} | moyenne {e.mean():+.4f} | "
          f"q25 {np.quantile(e, .25):+.4f} | q75 {np.quantile(e, .75):+.4f}")
    print(f"  Wilcoxon p = {_st.wilcoxon(e).pvalue:.3f}")
    ok = fact <= 2.0
    if ok.sum() >= 6:
        print(f"  en ne gardant que les {int(ok.sum())} paires a moins d'un facteur 2 "
              f"en superficie : median {np.median(e[ok]):+.4f}, "
              f"p = {_st.wilcoxon(e[ok]).pvalue:.3f}")

    print("\n  les 8 ecarts les plus negatifs")
    idx = np.argsort(e)[:8]
    for i in idx:
        r, c, f = paires[i]
        print(f"    {r.sid:>6} {r.plat} {r.aire:>7.0f} km2 | KGE {r.kge:.3f} "
              f"contre {c.kge.median():.3f} | ecart {e[i]:+.3f} | "
              f"{r.jours:.0f} j de stockage amont")

    print("\nLECTURE. Un ecart median franchement negatif et significatif dirait que la")
    print("regularisation COUTE, sans deformer le champ, et qu'ecarter ces stations de la")
    print("selection est une astuce rapide legitime. Un ecart nul ferme la porte pour de")
    print("bon : les stations regularisees se simulent comme les autres.")


if __name__ == "__main__":
    main()
