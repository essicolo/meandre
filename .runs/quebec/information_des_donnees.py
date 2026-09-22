"""Combien de directions indépendantes TOUTES les observations portent-elles ensemble.

Le champ spatial produit 43 paramètres par tronçon, soit plusieurs milliers d'inconnues par
territoire. La question n'est pas combien d'observations on possède, mais combien de
DIRECTIONS INDÉPENDANTES elles contraignent, et si ces directions sont distinctes les unes
des autres. Une contrainte qui duplique le débit ne lève aucune équifinalité du débit.

Même mesure que pour la neige : moyennes mensuelles, corrélation sur les mois communs à
chaque paire, composantes à 95 % de la variance, rapport de participation, et recouvrement
des quatre premières directions entre les deux moitiés de la période, qui sépare une
direction indépendante d'un bruit indépendant.

Le débit entre en LOGARITHME : c'est l'échelle sur laquelle ses variations sont comparables
d'une station à l'autre, un bassin de mille kilomètres carrés et un de dix ne se comparant
pas en mètres cubes par seconde.

    .venv/bin/python .runs/quebec/information_des_donnees.py outv
"""
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_ici = os.path.dirname(os.path.abspath(__file__))
_inf = SourceFileLoader("inf", os.path.join(_ici, "information_de_la_neige.py")).load_module()

from meandre.utils import paths as _paths

DERIVES = f"{os.environ.get('MEANDRE_DERIVES', _paths.DERIVED_ROOT)}/auxiliaires"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"


def _serie_large(df, col_id, col_date, col_val, axe):
    """Passe une table longue en matrice jours x entites, alignée sur l'axe demandé."""
    df = df.dropna(subset=[col_val])
    if df.empty:
        return None
    t = pd.to_datetime(df[col_date]).dt.normalize()
    large = pd.DataFrame({"t": t, "id": df[col_id].astype(str), "v": df[col_val].astype(float)})
    large = large.groupby(["t", "id"], as_index=False)["v"].mean()
    m = large.pivot(index="t", columns="id", values="v").reindex(axe)
    return m.to_numpy()


def _apport_marginal(sources, axe, _rec=24):
    """Ce qu'une source ajoute AUX AUTRES, et non ce qu'elle porte seule.

    Une source qui duplique le debit ne leve aucune equifinalite du debit. On mesure donc le
    rang effectif de l'ENSEMBLE, puis celui de l'ensemble prive de chaque source : la
    difference est ce que la source apporte en propre. Une source redondante donne zero,
    meme si elle porte beaucoup de directions a elle seule.

    Toutes les series sont mises au meme pas mensuel et empilees en colonnes ; le rang
    effectif de l'ensemble tient compte des correlations entre sources.
    """
    blocs = {}
    plein = np.ones(len(axe), dtype=bool)
    for nom, mat, masque in sources:
        # Toutes les sources sur le MEME axe mensuel : une source saisonniere est mise a
        # l'absence hors de sa saison plutot que raccourcie, sinon les blocs ne s'empilent pas.
        mm = _inf._mensualise(np.where(masque[:, None], mat, np.nan), axe, plein)
        # On ne garde qu'un nombre borne de colonnes par source : au-dela, une source tres
        # redondante gonflerait la matrice sans rien ajouter, et le calcul deviendrait lourd.
        n95, _part, _c = _inf._rang_effectif(mm, recouvrement_min=_rec)
        k = int(n95) * 4 if np.isfinite(n95) else 8
        v = np.isfinite(mm).sum(axis=0)
        ordre = np.argsort(v)[::-1][:max(k, 8)]
        bloc = mm[:, ordre]
        # Une source dont aucune colonne n'a de recouvrement utilisable ne peut pas entrer
        # dans l'ensemble sans le trouer : on la nomme et on l'ecarte.
        if np.isfinite(bloc).sum(axis=0).max() < max(4, _rec // 2):
            print(f"[apport] {nom} ecartee : recouvrement mensuel insuffisant")
            continue
        blocs[nom] = bloc
    noms = list(blocs)
    ensemble = np.concatenate([blocs[n] for n in noms], axis=1)
    n_tout, part_tout, _ = _inf._rang_effectif(ensemble, recouvrement_min=_rec)
    print("")
    print(f"ensemble des sources : {n_tout} directions a 95 %, participation {part_tout:.1f}")
    print("")
    print(f"{'source retiree':<26s} {'directions restantes':>21s} {'apport propre':>15s} "
          f"{'bruit de meme forme':>20s}")
    rng = np.random.default_rng(1234)
    for n in noms:
        reste = np.concatenate([blocs[m] for m in noms if m != n], axis=1)
        n_sans, _p, _c = _inf._rang_effectif(reste, recouvrement_min=_rec)
        if not (np.isfinite(n_tout) and np.isfinite(n_sans)):
            continue
        # TEMOIN : la meme source remplacee par du bruit de meme forme et de memes trous.
        # Si le bruit apporte autant, l'apport mesure ne vient pas de l'information mais du
        # nombre de colonnes.
        faux = np.where(np.isfinite(blocs[n]), rng.standard_normal(blocs[n].shape), np.nan)
        avec_bruit = np.concatenate([faux] + [blocs[m] for m in noms if m != n], axis=1)
        n_bruit, _pb, _cb = _inf._rang_effectif(avec_bruit, recouvrement_min=_rec)
        t = f"{n_bruit - n_sans:>20d}" if np.isfinite(n_bruit) else f"{'—':>20s}"
        print(f"{n:<26s} {n_sans:>21d} {n_tout - n_sans:>15d} {t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--debut", default=DATE_START)
    p.add_argument("--fin", default=DATE_END)
    p.add_argument("--recouvrement", type=int, default=0,
                   help="mois communs exiges par paire ; 0 = la moitie de la fenetre")
    a = p.parse_args()
    reg = a.region.lower()
    import duckdb

    axe = pd.date_range(a.debut, a.fin, freq="D")
    tout = pd.Series(True, index=axe).to_numpy()
    hiver = np.isin(axe.month, (11, 12, 1, 2, 3, 4, 5))
    con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)

    sources = []
    q = con.execute("SELECT station_id, date, discharge FROM observations").df()
    m = _serie_large(q, "station_id", "date", "discharge", axe)
    if m is not None:
        sources.append(("debit (log)", np.log(np.clip(m, 1e-3, None)), tout))

    g = con.execute("SELECT date, tws_mm FROM grace_tws WHERE quality_ok").df()
    mg = _serie_large(g.assign(id="bassin"), "id", "date", "tws_mm", axe)
    if mg is not None:
        sources.append(("GRACE, moyenne du bassin", mg, tout))

    try:
        e = con.execute("SELECT node_idx, date, etr_mm_day FROM modis_et WHERE quality_ok").df()
        me = _serie_large(e, "node_idx", "date", "etr_mm_day", axe)
        if me is not None:
            sources.append(("evapotranspiration MODIS", me, tout))
    except Exception as exc:
        print(f"[modis_et] non lu : {exc}")
    con.close()

    from meandre.data.basin_cache import BasinCache
    from meandre.data.canswe_loader import build_swe_targets

    cache = BasinCache(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb")
    if cache.has_canswe():
        _mes, _sit = cache.load_canswe(a.debut, a.fin)
        _obs, _ni, _g = build_swe_targets(_mes, _sit, axe)
        if _obs is not None:
            sources.append(("neige, reseau au sol", _obs.cpu().numpy(), hiver))

    f = f"{DERIVES}/neisim-{reg}.npz"
    if os.path.exists(f):
        _z = np.load(f)
        # La cible NEISIM porte son propre axe : on la recale sur la fenetre demandee plutot
        # que de supposer qu'elles coincident.
        _t = pd.DatetimeIndex(_z["times"])
        _pos = pd.Series(np.arange(len(_t)), index=_t).reindex(axe).to_numpy()
        _v = np.full((len(axe), _z["valeurs"].shape[1]), np.nan, dtype="float32")
        _vu = np.isfinite(_pos)
        _v[_vu] = _z["valeurs"][_pos[_vu].astype(int)]
        sources.append(("neige NEISIM", _v, hiver))
    pu = f"{DERIVES}/rsesq-niveaux-journaliers.parquet"
    if os.path.exists(pu):
        d = pd.read_parquet(pu)
        d = d[d["region"].astype(str).str.lower() == reg]
        # Memes ecarts que le chargeur de la perte : un puits captif mesure une charge et
        # non un stock, un puits sous pompage mesure l'exploitation.
        d = d[(d["confinement"].astype(str) != "Captive") & (d["influence"].astype(str) == "Non")]
        mn = _serie_large(d, "puits", "date", "niveau_m", axe)
        if mn is not None:
            sources.append(("nappes RSESQ", mn, tout))

    # Le recouvrement minimal exige doit suivre la LONGUEUR de la fenetre : a vingt-quatre
    # mois sur une fenetre de dix-sept, toute colonne est ecartee et le tableau sort vide.
    _n_mois = max(1, len(pd.PeriodIndex(axe, freq="M").unique()))
    _rec = int(a.recouvrement) if a.recouvrement else max(4, min(24, _n_mois // 2))
    print(f"{reg} : {_n_mois} mois, recouvrement minimal {_rec} mois\n")
    print(f"{'source':<26s} {'series':>8s} {'95 %':>7s} {'participation':>14s} "
          f"{'reproductible':>14s} {'hasard':>8s}")
    for nom, mat, masque in sources:
        mm = _inf._mensualise(mat, axe, masque)
        n95, part, cols = _inf._rang_effectif(mm, recouvrement_min=_rec)
        rep, pp = _inf._reproductibilite(mm, recouvrement_min=max(3, _rec // 2))
        # TEMOIN NUL, sans lequel le chiffre n'a pas d'echelle. Du bruit independant de meme
        # forme et de memes trous gonfle le rang effectif a lui seul : mesure sur l'altimetrie
        # satellitaire le 2026-09-22, il rend 28 directions quand la donnee en rend 19. Le
        # rang seul ne discrimine donc rien, et c'est la reproductibilite qui tranche.
        _rng = np.random.default_rng(1234)
        _faux = np.where(np.isfinite(mm), _rng.standard_normal(mm.shape), np.nan)
        _nb, _pb, _ = _inf._rang_effectif(_faux, recouvrement_min=_rec)
        _rb, _ = _inf._reproductibilite(_faux, recouvrement_min=max(3, _rec // 2))
        n95_t = f"{n95:>7d}" if np.isfinite(n95) else f"{'—':>7s}"
        part_t = f"{part:>14.1f}" if np.isfinite(part) else f"{'—':>14s}"
        rep_t = f"{rep:>14.2f}" if np.isfinite(rep) else f"{'—':>14s}"
        haz = f"{4.0 / pp:>8.2f}" if pp and np.isfinite(rep) else f"{'—':>8s}"
        print(f"{nom:<26s} {cols:>8d} {n95_t} {part_t} {rep_t} {haz}")
        _nbt = f"{_nb:>7d}" if np.isfinite(_nb) else f"{'—':>7s}"
        _pbt = f"{_pb:>14.1f}" if np.isfinite(_pb) else f"{'—':>14s}"
        _rbt = f"{_rb:>14.2f}" if np.isfinite(_rb) else f"{'—':>14s}"
        print(f"{'  temoin de bruit':<26s} {'':>8s} {_nbt} {_pbt} {_rbt} {haz}")
    _apport_marginal(sources, axe, _rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
