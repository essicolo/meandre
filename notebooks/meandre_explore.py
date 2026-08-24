"""Notebook marimo d'ANALYSE du modele 1.0 -- le Quebec entier, sans regions.

Reecrit le 2026-08-24 apres rejet explicite (et repete) du concept de region par Essi :
la premiere version portait un selecteur de region, artefact de production remonte
jusqu'a l'interface. Ici le decoupage par base de donnees est FONDU au chargement et
n'apparait nulle part : un seul domaine, le Quebec meridional.

Partage des roles :
  - la CARTE interactive (MapLibre, troncons cliquables, zones rouges, hydrogrammes au
    clic) est une instance feuillage : docs/carte/, couches par
    .runs/quebec/build_feuillage.py ;
  - ce NOTEBOOK fait l'analyse : les KGE dans l'espace, les distributions, les
    agregations statistiques, les diagnostics de calibration.

Aucune simulation ici : les caches viennent du pilote (ETL_DUMP_Q, ETL_DUMP_REACH),
donc chaque chiffre porte le runtime exact du run qui l'a produit (dette #6).

Lancer :  uv run marimo edit notebooks/meandre_explore.py
"""
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import os
    import sys

    import marimo as mo
    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lets_plot import (LetsPlot, aes, coord_fixed, facet_wrap, geom_boxplot,
                           geom_histogram, geom_line, geom_point, ggplot, ggsize,
                           ggtitle, labs)
    from meandre.utils import paths as mpaths

    LetsPlot.setup_html()
    RESULTS = f"{mpaths.DATA_ROOT}/quebec/results"

    def lp(p):
        return mo.Html(p._repr_html_())
    return (RESULTS, aes, coord_fixed, facet_wrap, geom_boxplot, geom_histogram,
            geom_line, geom_point, ggplot, ggsize, ggtitle, glob, labs, lp, mo,
            mpaths, np, os, pd)


@app.cell
def _(RESULTS, glob, mo, np, os, pd):
    # ── CHARGEMENT FUSIONNE : tout ce qui existe, presente comme UN domaine ──
    def kge_comp(o, s):
        r = np.corrcoef(o, s)[0, 1]
        beta = s.mean() / o.mean()
        gamma = (s.std() / s.mean()) / (o.std() / o.mean())
        return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2), r, beta, gamma

    lignes_st, lignes_mois = [], []
    troncons = []
    for fq in sorted(glob.glob(f"{RESULTS}/nb-*-q.npz")):
        src = os.path.basename(fq).split("-")[1]
        qd = np.load(fq, allow_pickle=True)
        dates = pd.to_datetime(qd["dates"])
        for j, sid in enumerate(qd["station_ids"]):
            qo, qs = qd["q_obs"][:, j], qd["q_sim"][:, j]
            v = np.isfinite(qo) & np.isfinite(qs)
            if v.sum() < 60:
                continue
            kge, r, beta, gamma = kge_comp(qo[v], qs[v])
            lignes_st.append(dict(station=str(sid), kge=kge, r=r, beta=beta,
                                  gamma=gamma, n=int(v.sum()), source=src))
            for m in range(1, 13):
                k = v & (dates.month == m)
                if k.sum() > 10:
                    lignes_mois.append(dict(station=str(sid), mois=m,
                                            rapport=float(qs[k].sum() / max(qo[k].sum(), 1e-9))))
    for fa in sorted(glob.glob(f"{RESULTS}/nb-*-avec.npz")):
        src = os.path.basename(fa).split("-")[1]
        avec = np.load(fa, allow_pickle=True)
        fs = fa.replace("-avec", "-sans")
        sans = np.load(fs, allow_pickle=True) if os.path.exists(fs) else None
        d = pd.DataFrame({"lon": avec["coords"][:, 0], "lat": avec["coords"][:, 1],
                          "q_annuel": avec["q_annuel"],
                          "prelev_abs": avec["prelev_net_abs"]})
        for k in avec.files:
            if k.startswith("param_"):
                d[k[6:]] = avec[k]
        if sans is not None:
            d["effet_prelev_pct"] = 100.0 * (avec["q_annuel"] - sans["q_annuel"]) / \
                np.clip(sans["q_annuel"], 1e-6, None)
        troncons.append(d)
    stations = pd.DataFrame(lignes_st)
    mensuel = pd.DataFrame(lignes_mois)
    quebec = pd.concat(troncons, ignore_index=True) if troncons else pd.DataFrame()
    mo.md(f"# meandre 1.0 — analyse provinciale\n\n"
          f"{len(stations)} stations et {len(quebec):,} tronçons chargés, un seul domaine. "
          f"La carte interactive vit dans docs/carte/ (feuillage).")
    return mensuel, quebec, stations


@app.cell
def _(mo):
    mo.md("## 1. Les KGE dans l'espace et en distribution\n"
          "Tenue de côté 2022-2024, toutes stations confondues. La médiane provinciale "
          "et la queue basse comptent davantage que n'importe quelle moyenne : c'est la "
          "queue qui dit où le modèle ne vaut pas encore livraison.")
    return


@app.cell
def _(aes, geom_histogram, ggplot, ggsize, ggtitle, lp, mo, stations):
    if stations.empty:
        outh = mo.md("(aucun cache de stations)")
    else:
        outh = mo.vstack([
            mo.md(f"médiane {stations.kge.median():.3f} | q10 {stations.kge.quantile(.1):.3f} "
                  f"| q90 {stations.kge.quantile(.9):.3f} | n = {len(stations)}"),
            lp(ggplot(stations, aes("kge")) + geom_histogram(bins=25)
               + ggtitle("Distribution provinciale des KGE") + ggsize(800, 300)),
        ])
    outh
    return


@app.cell
def _(aes, geom_boxplot, ggplot, ggsize, ggtitle, lp, mo, pd, stations):
    if stations.empty:
        outc = mo.md("")
    else:
        _l = stations.melt(id_vars=["station"], value_vars=["r", "beta", "gamma"],
                           var_name="composante", value_name="valeur")
        outc = lp(ggplot(_l, aes("composante", "valeur")) + geom_boxplot()
                  + ggtitle("Composantes du KGE (1 = parfait)") + ggsize(800, 300))
    outc
    return


@app.cell
def _(aes, geom_line, ggplot, ggsize, ggtitle, lp, mensuel, mo):
    if mensuel.empty:
        outm = mo.md("")
    else:
        _agg = mensuel.groupby("mois").rapport.median().reset_index()
        outm = mo.vstack([
            mo.md("## 2. Le cycle du biais\nRapport simulé/observé par mois, médiane des "
                  "stations : la signature saisonnière résiduelle du modèle, celle que "
                  "toute la campagne d'août a travaillée."),
            lp(ggplot(_agg, aes("mois", "rapport")) + geom_line()
               + ggtitle("Rapport simulé/observé, médiane provinciale") + ggsize(800, 300)),
        ])
    outm
    return


@app.cell
def _(mo, quebec):
    _params = sorted(c for c in (quebec.columns if not quebec.empty else [])
                     if c not in ("lon", "lat", "q_annuel", "prelev_abs", "effet_prelev_pct"))
    param = mo.ui.dropdown(_params or ["-"],
                           value=("krec" if "krec" in _params else (_params[0] if _params else "-")),
                           label="Paramètre")
    mo.md("## 3. Le champ appris, statistiquement\n"
          "Distribution provinciale d'un paramètre et sa structure spatiale en quantiles. "
          "La carte fine est dans feuillage ; ici on juge la DISPERSION (un champ plat = "
          "un NeRF collapsé, le mode de défaillance historique). " + str(param))
    return (param,)


@app.cell
def _(aes, coord_fixed, geom_histogram, geom_point, ggplot, ggsize, ggtitle, lp,
      mo, np, param, pd, quebec):
    if quebec.empty or param.value == "-":
        outp = mo.md("(aucun cache de tronçons : lancer les dumps ETL_DUMP_REACH)")
    else:
        _v = quebec[param.value].to_numpy()
        _log = np.nanmin(_v) > 0 and np.nanmax(_v) / max(np.nanmin(_v), 1e-30) > 100
        _d = pd.DataFrame({"lon": quebec.lon, "lat": quebec.lat,
                           "valeur": np.log10(_v) if _log else _v})
        _cv = float(np.nanstd(_v) / max(abs(np.nanmean(_v)), 1e-12))
        outp = mo.vstack([
            mo.md(f"CV provincial = {_cv:.3f} (un CV proche de zéro = champ collapsé)"),
            lp(ggplot(_d, aes("valeur")) + geom_histogram(bins=40)
               + ggtitle(f"{param.value}" + (" (log10)" if _log else "")) + ggsize(800, 250)),
            lp(ggplot(_d.sample(min(len(_d), 20000)), aes("lon", "lat", color="valeur"))
               + geom_point(size=1.2) + coord_fixed(ratio=1.4)
               + ggtitle("Structure spatiale") + ggsize(800, 600)),
        ])
    outp
    return


@app.cell
def _(mo, np, quebec):
    if quebec.empty or "effet_prelev_pct" not in quebec.columns:
        outw = mo.md("## 4. Prélèvements et rejets\n(paire avec/sans absente : lancer les "
                     "dumps renaturalisation)")
    else:
        _e = quebec.effet_prelev_pct.to_numpy()
        _touche = np.abs(_e) > 1.0
        outw = mo.vstack([
            mo.md("## 4. Prélèvements et rejets, agrégats provinciaux\n"
                  "Le détail cliquable (zones rouges, hydrogrammes avec/renaturalisé) est "
                  "dans la carte feuillage. Ici, les ordres de grandeur du mandat :"),
            mo.md(f"- tronçons où l'effet dépasse 1 % du débit renaturalisé : "
                  f"**{int(_touche.sum()):,}** sur {len(_e):,} "
                  f"({100 * _touche.mean():.1f} %)\n"
                  f"- effet médian sur les tronçons touchés : "
                  f"{np.median(_e[_touche]) if _touche.any() else 0:.1f} %\n"
                  f"- pire assèchement : {np.nanmin(_e):.1f} % | pire rehaussement : "
                  f"{np.nanmax(_e):.1f} %"),
        ])
    outw
    return


if __name__ == "__main__":
    app.run()
