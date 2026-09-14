"""Quelle part des parametres transferes vient de la POSITION plutot que des attributs ?

Au depart a chaud, un point de reprise entraine sur une region donneuse est applique aux
troncons d'une region d'accueil. La voie des attributs interpole : les seize attributs du
territoire d'accueil ressemblent a ceux vus a l'entrainement. La voie de la position
extrapole : le reseau n'a jamais vu ces coordonnees, et l'encodage de Fourier est
periodique, si bien qu'un point lointain retombe sur des valeurs deja vues pour un point
sans rapport. Les periodes des six bandes valent 2400, 1200, 600, 300, 150 et 75 km sur un
rayon de reference de 1200 km ; les trois plus hautes sont donc plus courtes qu'une region.

Rien a l'entrainement ne force le reseau a s'appuyer sur les attributs. Le score d'une
region transferee melange donc la transferabilite de la relation attributs vers parametres
et le bruit d'une voie positionnelle hors domaine. Ce script les separe, sans entrainement.

Trois mesures :
  1. parametres de l'accueil avec ses vraies positions, puis avec les positions des
     troncons du donneur les plus proches en attributs ;
  2. la meme, en annulant les bandes de Fourier dont la periode est plus courte qu'une
     region, qui sont celles qui ne peuvent pas extrapoler ;
  3. gradient des parametres par rapport a l'encodage de position contre gradient par
     rapport aux attributs.

    .venv/Scripts/python.exe .runs/quebec/audit_position.py sagu gasp
"""
import dataclasses
import sys

import duckdb
import numpy as np
import pandas as pd
import torch

from meandre.utils import paths as _p

TAILLE_REGION_KM = 300.0


# Les SEIZE attributs reellement consommes par le champ, normalises sur les statistiques
# PROVINCIALES et non par region. La table `territorial` de chaque base en porte vingt, dont
# trois en unites brutes ; s'en servir telle quelle donne au reseau une entree de dimension
# 20 alors qu'il en attend 16, et une mesure sans valeur. Le brut provincial et ses
# statistiques sont dans le depot.
BRUT = f"{_p.DATA_ROOT}/quebec/territorial-raw-QC.parquet"
STATS = "reports/territorial_stats_QC.csv"


def charge(reg, n_attendu=None):
    c = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    nd = c.execute("select node_idx,lon,lat from nodes order by node_idx").fetchdf()
    c.close()
    raw = pd.read_parquet(BRUT)
    raw = raw[raw["region"] == reg]
    st = pd.read_csv(STATS, index_col=0)
    cols = list(st.index)
    if n_attendu is not None and len(cols) != n_attendu:
        cols = cols[:n_attendu]
    if len(raw) != len(nd):
        raise SystemExit(f"{reg} : {len(raw)} lignes brutes pour {len(nd)} noeuds")
    arr = np.stack([((raw[c].values - st.loc[c, "mean"]) / (st.loc[c, "std"] + 1e-9))
                    for c in cols], axis=1).astype(np.float32)
    return (torch.tensor(nd[["lon", "lat"]].values, dtype=torch.float32),
            torch.tensor(arr), cols)


def params_dict(sp):
    return {f.name: getattr(sp, f.name) for f in dataclasses.fields(sp)
            if torch.is_tensor(getattr(sp, f.name))}


def ecart(a, b):
    """Ecart relatif par parametre, robuste au signe et aux valeurs proches de zero."""
    out = {}
    for k in a:
        if k not in b or a[k].shape != b[k].shape:
            continue
        x, y = a[k].detach().float().flatten(), b[k].detach().float().flatten()
        ech = torch.maximum(x.abs(), torch.tensor(1e-8))
        e = ((y - x).abs() / ech).numpy()
        out[k] = (float(np.median(e)), float(np.percentile(e, 90)))
    return out


def main(donneur, accueil, ckpt=None):
    from meandre.model import HydroModel

    ckpt = ckpt or f".runs/quebec/checkpoints/best-{donneur}-etl-socle30.pt"
    import torch as _t
    _sd = _t.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    n_terr = int(_sd["spatial_encoder.fc1.weight"].shape[1]) - 26
    cd, td, _ = charge(donneur, n_terr)
    ca, ta, noms = charge(accueil, n_terr)
    print(f"attributs consommés par le champ : {n_terr} ({', '.join(noms)})")
    print(f"donneur {donneur.upper()} {len(cd)} tronçons | accueil {accueil.upper()} {len(ca)} tronçons")
    print(f"point de reprise {ckpt}")

    m = HydroModel(n_nodes=len(ca), n_territorial=ta.shape[1], n_forcing=6,
                   use_temporal=False, use_residual=False, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_latent_codes=False,
                   spatial_melt=True, compile_soil=False)
    m.load(ckpt)
    m.eval()
    enc = m.spatial_encoder

    # Voisin le plus proche en ATTRIBUTS, dans l'espace standardise du donneur.
    mu, sd = td.mean(0), td.std(0).clamp(min=1e-6)
    zd, za = (td - mu) / sd, (ta - mu) / sd
    d = torch.cdist(za, zd)
    proche = d.argmin(dim=1)
    print(f"distance de Gower au voisin retenu : médiane {float(d.min(dim=1).values.median()):.2f} "
          f"écart-type standardisé")
    coords_sub = cd[proche]
    dkm = torch.sqrt(((enc._project_coords(ca) - enc._project_coords(coords_sub)) ** 2).sum(-1)) * enc._RAYON_KM
    print(f"déplacement imposé par la substitution : médiane {float(dkm.median()):.0f} km, "
          f"90e centile {float(dkm.quantile(0.9)):.0f} km\n")

    with torch.no_grad():
        p_vrai = params_dict(enc(ca, ta))
        p_sub = params_dict(enc(coords_sub, ta))
    e1 = ecart(p_vrai, p_sub)

    # 2. bandes de Fourier plus courtes qu'une region, annulees
    per = np.array([2 * enc._RAYON_KM / (2.0 ** k) for k in range(enc.coord_enc.n_freqs)])
    garde = per >= TAILLE_REGION_KM
    anciennes = enc.coord_enc.freqs.clone()
    enc.coord_enc.freqs = anciennes * torch.tensor(garde.astype(np.float32))
    with torch.no_grad():
        p_vrai_b = params_dict(enc(ca, ta))
        p_sub_b = params_dict(enc(coords_sub, ta))
    e2 = ecart(p_vrai_b, p_sub_b)
    enc.coord_enc.freqs = anciennes
    print(f"bandes conservées à l'étape 2 : périodes {list(per[garde].astype(int))} km ; "
          f"annulées : {list(per[~garde].astype(int))} km\n")

    # 3. gradients : position contre attributs
    ca_g = ca.clone().requires_grad_(True)
    ta_g = ta.clone().requires_grad_(True)
    sp = enc(ca_g, ta_g)
    gr = {}
    for f in dataclasses.fields(sp):
        v = getattr(sp, f.name)
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        gc, gt = torch.autograd.grad(v.sum(), [ca_g, ta_g], retain_graph=True, allow_unused=True)
        if gc is None or gt is None:
            continue
        # normalisation : gradient par ECART-TYPE de l'entree, pour comparer des unites
        s_pos = float((gc * ca.std(0)).abs().sum(-1).mean())
        s_att = float((gt * ta.std(0)).abs().sum(-1).mean())
        gr[f.name] = (s_pos, s_att, s_pos / max(s_pos + s_att, 1e-12))

    lignes = []
    for k in sorted(e1, key=lambda x: -gr.get(x, (0, 0, 0))[2]):
        g = gr.get(k, (float("nan"),) * 3)
        lignes.append({"paramètre": k,
                       "écart médian (%)": round(100 * e1[k][0], 1),
                       "écart 90e centile (%)": round(100 * e1[k][1], 1),
                       "écart médian, hautes fréquences annulées (%)": round(100 * e2.get(k, (np.nan,))[0], 1),
                       "part du gradient due à la position (%)": round(100 * g[2], 1)})
    t = pd.DataFrame(lignes)
    t.to_csv(f".reports/quebec/caches/audit-position-{donneur}-{accueil}.csv", index=False)
    print(t.to_string(index=False))
    print(f"\nrésumé sur les {len(t)} paramètres")
    print(f"  écart médian dû à la position : {t['écart médian (%)'].median():.1f} % "
          f"| 90e centile des écarts : {t['écart 90e centile (%)'].median():.1f} %")
    print(f"  après annulation des bandes plus courtes qu'une région : "
          f"{t['écart médian, hautes fréquences annulées (%)'].median():.1f} %")
    print(f"  part médiane du gradient portée par la position : "
          f"{t['part du gradient due à la position (%)'].median():.1f} %")
    n = int((t["part du gradient due à la position (%)"] > 50).sum())
    print(f"  paramètres dont la position porte plus de la moitié du gradient : {n} sur {len(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu",
                  sys.argv[2] if len(sys.argv) > 2 else "gasp",
                  sys.argv[3] if len(sys.argv) > 3 else None))
