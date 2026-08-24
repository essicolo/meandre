"""Notebook marimo d'exploration du modele 1.0 (demande d'Essi, 2026-08-24).

Quatre vues : hydrogrammes par station, cartes des parametres du champ NeRF, cartes
de l'effet des prelevements et rejets par troncon, et hydrogrammes prelevements
contre debit. Le notebook ne fait AUCUNE simulation : il lit les caches produits par
le pilote lui-meme (ETL_DUMP_Q et ETL_DUMP_REACH), pour que chaque figure porte
exactement le runtime du run qui l'a produite (dette #6 du registre).

Produire les caches pour une region (exemple OUTV, paire renaturalisation) :
  ETL_DUMP_REACH=D:/meandre-data/quebec/results/nb-outv-avec.npz  [recette]  etl_run.py
  ETL_SANS_PRELEV=1 ETL_DUMP_REACH=...nb-outv-sans.npz            [recette]  etl_run.py

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

    import duckdb
    import marimo as mo
    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lets_plot import (LetsPlot, aes, coord_fixed, geom_line, geom_point,
                           ggplot, ggsize, ggtitle, labs, scale_color_gradient)
    from meandre.utils import paths as mpaths

    LetsPlot.setup_html()
    RESULTS = f"{mpaths.DATA_ROOT}/quebec/results"

    def lp(p):
        """Rend une figure lets_plot dans marimo."""
        return mo.Html(p._repr_html_())
    return duckdb, glob, lp, mo, mpaths, np, os, pd, RESULTS, aes, coord_fixed, \
        geom_line, geom_point, ggplot, ggsize, ggtitle, labs


@app.cell
def _(glob, mo, os, RESULTS):
    _regions = sorted({os.path.basename(f).split("-")[1]
                       for f in glob.glob(f"{RESULTS}/nb-*-avec.npz")})
    region = mo.ui.dropdown(_regions or ["outv"], value=(_regions[0] if _regions else "outv"),
                            label="Region")
    mo.md(f"# Exploration meandre 1.0\n{region}")
    return (region,)


@app.cell
def _(mo, np, os, region, RESULTS):
    _fa = f"{RESULTS}/nb-{region.value}-avec.npz"
    _fs = f"{RESULTS}/nb-{region.value}-sans.npz"
    _fq = f"{RESULTS}/nb-{region.value}-q.npz"
    manque = [f for f in (_fa, _fs, _fq) if not os.path.exists(f)]
    avec = np.load(_fa, allow_pickle=True) if os.path.exists(_fa) else None
    sans = np.load(_fs, allow_pickle=True) if os.path.exists(_fs) else None
    qd = np.load(_fq, allow_pickle=True) if os.path.exists(_fq) else None
    mo.md("ATTENTION, caches manquants : " + ", ".join(map(os.path.basename, manque))
          if manque else "Caches charges (avec / sans prelevements / series stations).")
    return avec, qd, sans


@app.cell
def _(mo):
    mo.md("## 1. Hydrogrammes de tenue de cote (2022-2024)\n"
          "Simule contre observe aux stations. Le simule est la MEDIANE du modele ; "
          "l'enveloppe quantile et l'ensemble de forcage s'y superposeront quand la "
          "chaine probabiliste sera scellee.")
    return


@app.cell
def _(mo, qd):
    station = mo.ui.dropdown(list(qd["station_ids"]) if qd is not None else ["-"],
                             value=(list(qd["station_ids"])[0] if qd is not None else "-"),
                             label="Station")
    station
    return (station,)


@app.cell
def _(aes, geom_line, ggplot, ggsize, labs, lp, mo, pd, qd, station):
    if qd is None:
        out1 = mo.md("(cache stations absent)")
    else:
        _i = list(qd["station_ids"]).index(station.value)
        _d = pd.DataFrame({"date": pd.to_datetime(qd["dates"]),
                           "observe": qd["q_obs"][:, _i], "simule": qd["q_sim"][:, _i]})
        _d = _d.melt("date", var_name="serie", value_name="debit")
        out1 = lp(ggplot(_d, aes("date", "debit", color="serie")) + geom_line()
                  + labs(y="debit (m3/s)", title=f"Station {station.value}")
                  + ggsize(900, 350))
    out1
    return


@app.cell
def _(avec, mo):
    _params = sorted(k[6:] for k in (avec.files if avec is not None else [])
                     if k.startswith("param_"))
    param = mo.ui.dropdown(_params or ["-"], value=("krec" if "krec" in _params
                                                    else (_params[0] if _params else "-")),
                           label="Parametre du champ")
    mo.md("## 2. Cartes des parametres NeRF\nLe champ appris, par troncon. " + str(param))
    return (param,)


@app.cell
def _(aes, avec, coord_fixed, geom_point, ggplot, ggsize, ggtitle, lp, mo, np, param, pd):
    if avec is None or param.value == "-":
        out2 = mo.md("(cache absent)")
    else:
        _v = avec[f"param_{param.value}"]
        _c = avec["coords"]
        _log = (_v.min() > 0) and (_v.max() / max(_v.min(), 1e-30) > 100)
        _d2 = pd.DataFrame({"lon": _c[:, 0], "lat": _c[:, 1],
                            "valeur": np.log10(_v) if _log else _v})
        out2 = lp(ggplot(_d2, aes("lon", "lat", color="valeur")) + geom_point(size=2)
                  + coord_fixed(ratio=1.4)
                  + ggtitle(f"{param.value}" + (" (log10)" if _log else ""))
                  + ggsize(750, 600))
    out2
    return


@app.cell
def _(mo):
    mo.md("## 3. Effet des prelevements et rejets par troncon\n"
          "Difference relative de debit annuel entre le run AVEC et le run renaturalise "
          "(SANS), sur le meme point de reprise : c'est le protocole du mandat. Les "
          "jauges ne voient que 12-33 % du flux anthropique (R17) : cette carte est "
          "precisement ce que le modele apporte au-dela d'elles.")
    return


@app.cell
def _(aes, avec, coord_fixed, geom_point, ggplot, ggsize, ggtitle, lp, mo, np, pd, sans):
    if avec is None or sans is None:
        out3 = mo.md("(paire avec/sans absente)")
    else:
        _qa, _qs = avec["q_annuel"], sans["q_annuel"]
        _eff = 100.0 * (_qa - _qs) / np.clip(_qs, 1e-6, None)
        _c3 = avec["coords"]
        _d3 = pd.DataFrame({"lon": _c3[:, 0], "lat": _c3[:, 1], "effet_pct": _eff})
        _seuil = float(np.percentile(np.abs(_eff), 99.5))
        _d3["effet_pct"] = _d3.effet_pct.clip(-_seuil, _seuil)
        out3 = lp(ggplot(_d3, aes("lon", "lat", color="effet_pct")) + geom_point(size=2)
                  + coord_fixed(ratio=1.4)
                  + ggtitle("Effet des prelevements et rejets sur le debit annuel (%)")
                  + ggsize(750, 600))
    out3
    return


@app.cell
def _(duckdb, mo, mpaths, pd, region):
    _db = mpaths.data_path("quebec", f"{region.value}.duckdb")
    _con = duckdb.connect(_db, read_only=True)
    try:
        prelev = _con.execute(
            "SELECT * FROM withdrawals ORDER BY 1 LIMIT 1").df()
        cols_w = [c[0] for c in _con.execute("DESCRIBE withdrawals").fetchall()]
        stations_db = _con.execute(
            "SELECT station_id, node_idx FROM stations").df()
    finally:
        _con.close()
    mo.md("## 4. Prelevements et rejets contre le debit\n"
          "Serie du prelevement net rapportee au debit simule du troncon : l'ordre de "
          "grandeur LOCAL du signal anthropique, la ou il vit.")
    return cols_w, stations_db


@app.cell
def _(avec, mo, np, pd, aes, geom_line, ggplot, ggsize, labs, lp):
    if avec is None:
        out4 = mo.md("(cache absent)")
    else:
        _w = avec["prelev_net_abs"]
        _qa4 = avec["q_annuel"] * 86400.0 * 365.25 / 1e6   # hm3/an approx pour ratio
        _top = np.argsort(_w)[::-1][:12]
        _d4 = pd.DataFrame({"troncon": _top.astype(str),
                            "prelevement_abs": _w[_top],
                            "part_du_debit_pct": 100.0 * _w[_top] / np.clip(_qa4[_top] * 1e6 / 31_557_600, 1e-6, None)})
        out4 = mo.vstack([
            mo.md("Les 12 troncons au plus fort prelevement net (valeur absolue annuelle) :"),
            mo.ui.table(_d4, selection=None),
        ])
    out4
    return


if __name__ == "__main__":
    app.run()
