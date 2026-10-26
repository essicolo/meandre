"""Candidate 5.2, épreuve 3.3 : existe-t-il une forme commune de la loi de vidange souterraine,
au-delà de la puissance, qui transfère aux stations jamais vues ?

Matériau : débits observés 2000-2024 des bases DuckDB (jours mesurés seulement, les
reconstitutions du CEHQ exclues), segments de décrue franche de plus de cinq jours hors les deux
jours suivant une pointe (meandre.data.recession_anchor), enveloppe inférieure du nuage
(Brutsaert-Nieber), hors de la saison de fonte (mars à mai exclus) et hors gel (décembre à
février exclus).

Cible : y = log(−dQ/dt), en débit spécifique (mm/j), en fonction de x = log(Q) spécifique.

Modèles, du plus contraint au plus libre :
  P1  puissance par station : y = a_s + b_s x (le Brutsaert-Nieber classique) ;
  P0  puissance à exposant commun : y = a_s + b x ;
  G   forme commune non linéaire : y = a_s + g(x − x_s), g monotone croissante (différences
      cumulées de termes positifs sur une grille), x_s = log du débit médian de la station ;
  GA  même g, conditionnée par les attributs du tronçon exutoire (petit réseau, 16 unités).

Validation : (A) années retirées, une sur cinq, toutes stations vues ; (B) stations retirées,
cinq plis : l'échelle a_s de la station retirée est estimée sur ses propres décrues de 2000 à
2012 (ce que fait le champ k_gw ancré sur les récessions), et la FORME vient des autres
stations ; évaluation 2013-2024. Erreur : écart absolu médian de y, et erreur relative
médiane sur −dQ/dt.

    .venv/Scripts/python.exe experiments/plan_ml/nappe_recession.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.data.recession_anchor import segments_de_decrue, _enveloppe_inferieure
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
MOIS_EXCLUS = {12, 1, 2, 3, 4, 5}


def points_station(reg, code, aire):
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    cols = {c[0] for c in con.execute("describe observations").fetchall()}
    sel = "date, discharge" + (", reconstructed" if "reconstructed" in cols else "")
    d = con.execute(f"select {sel} from observations where station_id = ? order by date", [code]).fetchdf()
    con.close()
    d = d.set_index(pd.DatetimeIndex(d["date"]))
    q = d["discharge"].astype(float)
    if "reconstructed" in d:
        q[d["reconstructed"].fillna(False).astype(bool)] = np.nan
    q = q.asfreq("D")
    v = q.to_numpy()
    fini = np.isfinite(v) & (v > 0)
    bons = np.zeros(len(v), bool)
    # Segments calculés sur la série pleine, un trou coupe un segment.
    vv = np.where(fini, v, np.nan)
    idx = np.flatnonzero(fini)
    if len(idx) < 700:
        return None
    seg = segments_de_decrue(vv[fini])
    bons[idx] = seg
    dqdt = np.concatenate([[np.nan], -np.diff(vv)])
    qm = np.concatenate([[np.nan], 0.5 * (vv[1:] + vv[:-1])])
    garde = bons & np.isfinite(dqdt) & (dqdt > 0) & np.isfinite(qm) & (qm > 0) & ~q.index.month.isin(MOIS_EXCLUS)
    if garde.sum() < 100:
        return None
    # mm/j : m3/s * 86.4 / km2
    f = 86.4 / aire
    x = np.log(qm[garde] * f)
    y = np.log(dqdt[garde] * f)
    annee = q.index.year.to_numpy()[garde]
    ex, ey = _enveloppe_inferieure(x, y)
    if ex is None:
        return None
    # Retrouver les années des points d'enveloppe par appariement exact.
    cle = {(round(a, 9), round(b, 9)): an for a, b, an in zip(x, y, annee)}
    ea = np.array([cle.get((round(a, 9), round(b, 9)), -1) for a, b in zip(ex, ey)])
    out = pd.DataFrame({"x": ex, "y": ey, "annee": ea})
    out["x_s"] = float(np.log(np.nanmedian(v[fini]) * f))
    return out


class FormeCommune(nn.Module):
    """g monotone croissante par différences cumulées positives sur une grille de x centré ;
    option : conditionnée par des attributs."""

    def __init__(self, n_att=0, grille=np.linspace(-4, 3, 36)):
        super().__init__()
        self.grille = torch.tensor(grille, dtype=torch.float32)
        self.n_att = n_att
        if n_att:
            self.att = nn.Sequential(nn.Linear(n_att, 16), nn.Tanh(), nn.Linear(16, len(grille) - 1))
            nn.init.zeros_(self.att[-1].weight)
            nn.init.zeros_(self.att[-1].bias)
        self.inc = nn.Parameter(torch.zeros(len(grille) - 1))
        self.pente0 = nn.Parameter(torch.tensor(1.5))

    def forward(self, xc, att=None):
        inc = self.inc if att is None else self.inc + self.att(att)
        pas = torch.nn.functional.softplus(inc) * (self.grille[1:] - self.grille[:-1])
        cum = torch.cat([torch.zeros_like(pas[..., :1]), torch.cumsum(pas, -1)], -1)
        # interpolation linéaire sur la grille, extrapolation linéaire par la pente de départ
        xi = torch.clamp(xc, self.grille[0], self.grille[-1])
        k = torch.clamp(torch.searchsorted(self.grille, xi, right=True) - 1, 0, len(self.grille) - 2)
        g0, g1 = self.grille[k], self.grille[k + 1]
        if cum.dim() == 1:
            c0, c1 = cum[k], cum[k + 1]
        else:
            c0, c1 = cum.gather(-1, k.unsqueeze(-1)).squeeze(-1), cum.gather(-1, (k + 1).unsqueeze(-1)).squeeze(-1)
        w = (xi - g0) / (g1 - g0)
        g = c0 + w * (c1 - c0) + self.pente0 * (xc - xi)
        # écart à la puissance de référence : g vaut 0 à x centré nul
        k0 = torch.clamp(torch.searchsorted(self.grille, torch.zeros(1), right=True) - 1, 0, len(self.grille) - 2)
        c00 = cum[..., k0] if cum.dim() == 1 else cum[..., k0].squeeze(-1)
        return g - c00


def ajuster_forme(D, stations_app, annees_app, att=None, epoques=600):
    """g commune et échelles a_s par station, par moindres carrés alternés sur Adam."""
    d = D[D.station.isin(stations_app) & D.annee.isin(annees_app)]
    st = sorted(d.station.unique())
    ids = {s: i for i, s in enumerate(st)}
    xc = torch.tensor((d.x - d.x_s).to_numpy(np.float32))
    y = torch.tensor(d.y.to_numpy(np.float32))
    si = torch.tensor(d.station.map(ids).to_numpy())
    A = None if att is None else torch.tensor(att.loc[d.station].to_numpy(np.float32))
    m = FormeCommune(0 if att is None else att.shape[1])
    a = nn.Parameter(torch.zeros(len(st)))
    opt = torch.optim.Adam(list(m.parameters()) + [a], lr=0.03)
    for _ in range(epoques):
        opt.zero_grad()
        pred = a[si] + m(xc, A)
        l = ((pred - y) ** 2).mean()
        l.backward()
        opt.step()
    return m


def echelle(m, D, station, annees, att=None):
    d = D[(D.station == station) & D.annee.isin(annees)]
    if len(d) < 10:
        return None
    with torch.no_grad():
        g = m(torch.tensor((d.x - d.x_s).to_numpy(np.float32)), None if att is None else torch.tensor(att.loc[[station] * len(d)].to_numpy(np.float32)))
    return float((torch.tensor(d.y.to_numpy(np.float32)) - g).median())


def predire(m, D, station, annees, a, att=None):
    d = D[(D.station == station) & D.annee.isin(annees)]
    with torch.no_grad():
        g = m(torch.tensor((d.x - d.x_s).to_numpy(np.float32)), None if att is None else torch.tensor(att.loc[[station] * len(d)].to_numpy(np.float32)))
    return d.y.to_numpy(), a + g.numpy()


def puissance(D, station, annees_app, b=None):
    d = D[(D.station == station) & D.annee.isin(annees_app)]
    if len(d) < 10:
        return None
    if b is None:
        b, a = np.polyfit(d.x - d.x_s, d.y, 1)
    else:
        a = float(np.median(d.y - b * (d.x - d.x_s)))
    return a, b


def main():
    deb = pd.read_parquet(f"{REF}/debits.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    blocs = []
    for st in deb["q_obs"].columns:
        reg, code = st.split(":")
        p = points_station(reg, code, float(at.loc[st, "drainage_area_km2"]))
        if p is None:
            continue
        p["station"] = st
        blocs.append(p)
    D = pd.concat(blocs)
    D = D[(D.annee > 0) & np.isfinite(D.x) & np.isfinite(D.y) & np.isfinite(D.x_s)]
    stations = np.array(sorted(D.station.unique()))
    print(f"{len(stations)} stations, {len(D)} points d'enveloppe de décrue, juin à novembre, 2000-2024")
    bs = {s: puissance(D, s, range(2000, 2025))[1] for s in stations}
    print(f"exposant b par station : médiane {np.median(list(bs.values())):.2f}, quartiles {np.quantile(list(bs.values()), 0.25):.2f} à {np.quantile(list(bs.values()), 0.75):.2f}")
    cols = ["mean_slope_pct", "mean_elevation_m", "f_forest", "f_agriculture", "f_wetland", "f_sand", "f_clay", "lake_fraction", "drainage_area_km2"]
    att = at.loc[stations, cols].copy()
    att["drainage_area_km2"] = np.log(att["drainage_area_km2"])
    att = (att - att.mean()) / att.std().replace(0, 1)

    def evaluer(nom, pairs):
        y = np.concatenate([p[0] for p in pairs])
        h = np.concatenate([p[1] for p in pairs])
        e = np.abs(y - h)
        print(f"  {nom:40s} écart absolu médian de log(−dQ/dt) {np.median(e):.3f} ; erreur relative médiane {100 * np.median(np.abs(np.exp(h - y) - 1)):.1f} % ; n {len(y)}")
        return np.median(e)

    print("\nA. années retirées (une sur cinq), toutes stations vues")
    res = {k: [] for k in ("P1", "P0", "G", "GA")}
    annees = np.arange(2000, 2025)
    for pli in range(5):
        a_te = annees[pli::5]
        a_app = np.setdiff1d(annees, a_te)
        b_com = np.median([puissance(D, s, a_app)[1] for s in stations if puissance(D, s, a_app)])
        mG = ajuster_forme(D, stations, a_app)
        mGA = ajuster_forme(D, stations, a_app, att)
        for s in stations:
            p1 = puissance(D, s, a_app)
            if p1 is None or len(D[(D.station == s) & D.annee.isin(a_te)]) == 0:
                continue
            d_te = D[(D.station == s) & D.annee.isin(a_te)]
            res["P1"].append((d_te.y.to_numpy(), p1[0] + p1[1] * (d_te.x - d_te.x_s).to_numpy()))
            p0 = puissance(D, s, a_app, b_com)
            res["P0"].append((d_te.y.to_numpy(), p0[0] + b_com * (d_te.x - d_te.x_s).to_numpy()))
            res["G"].append(predire(mG, D, s, a_te, echelle(mG, D, s, a_app)))
            res["GA"].append(predire(mGA, D, s, a_te, echelle(mGA, D, s, a_app, att), att))
    for k, nom in (("P1", "puissance par station"), ("P0", "puissance, exposant commun"), ("G", "forme commune monotone"), ("GA", "forme commune conditionnée par attributs")):
        evaluer(nom, res[k])

    print("\nB. stations retirées (cinq plis) ; échelle de la station estimée sur ses décrues 2000-2012, forme venue des autres ; évaluation 2013-2024")
    res = {k: [] for k in ("P1", "P0", "G", "GA")}
    a_ech, a_te = np.arange(2000, 2013), np.arange(2013, 2025)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(stations))
    for pli in range(5):
        tenues = stations[perm[pli::5]]
        app = np.setdiff1d(stations, tenues)
        b_com = np.median([puissance(D, s, annees)[1] for s in app if puissance(D, s, annees)])
        mG = ajuster_forme(D, app, annees)
        mGA = ajuster_forme(D, app, annees, att)
        for s in tenues:
            p1 = puissance(D, s, a_ech)
            if p1 is None or len(D[(D.station == s) & D.annee.isin(a_te)]) == 0:
                continue
            d_te = D[(D.station == s) & D.annee.isin(a_te)]
            res["P1"].append((d_te.y.to_numpy(), p1[0] + p1[1] * (d_te.x - d_te.x_s).to_numpy()))
            p0 = puissance(D, s, a_ech, b_com)
            res["P0"].append((d_te.y.to_numpy(), p0[0] + b_com * (d_te.x - d_te.x_s).to_numpy()))
            res["G"].append(predire(mG, D, s, a_te, echelle(mG, D, s, a_ech)))
            res["GA"].append(predire(mGA, D, s, a_te, echelle(mGA, D, s, a_ech, att), att))
    for k, nom in (("P1", "puissance par station (b propre, 2000-2012)"), ("P0", "puissance, exposant commun des autres"), ("G", "forme commune monotone des autres"), ("GA", "forme commune conditionnée par attributs")):
        evaluer(nom, res[k])

    # Forme apprise sur tout, pour la lire
    mG = ajuster_forme(D, stations, annees)
    with torch.no_grad():
        xs = torch.linspace(-3, 2, 11)
        g = mG(xs).numpy()
    pente = np.gradient(g, xs.numpy())
    print("\nforme commune g (x = log Q centré sur la médiane de la station) et pente locale (exposant b local) :")
    print("  x     : " + " ".join(f"{v:6.1f}" for v in xs.numpy()))
    print("  pente : " + " ".join(f"{v:6.2f}" for v in pente))
    D.to_parquet(f"{DOSSIER}/recessions_points.parquet")


if __name__ == "__main__":
    main()
