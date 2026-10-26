"""ETP LINACRE — clone fidèle de hydrotel/source/linacre.cpp (v4.3.6).
C'est le modèle d'ETP des 15 plateformes LN24HA 2020, avec le COEFFICIENT
MULTIPLICATIF OPTIMISATION par UHRH de linacre.csv = LE calage régional d'ETP.

Formule (Calcule(), l.65-117) :
  dtmxn  = T_mois_chaud - T_mois_froid            (linacre.csv, déf 20-(-10)=30)
  corn   = 24.41 / (100 - lat)
  tmoy   = (tmin+tmax)/2
  Delta  = 4098·e_s(tmoy) / (tmoy+237.3)²          e_s = 0.6108·exp(17.27T/(T+237.3))
  P      = 101.3·((293-0.0065·alti)/293)^5.26 ;  Gamma = 0.000665·P
  tmer   = corn·(tmoy + 0.006·alti)
  ea     = 0.3807·(0.0023·alti + 0.37·tmoy + 0.53·(tmax-tmin) + 0.35·dtmxn - 10.9)
  albedo = albedo_neige si couvert nival > 0, sinon albedo (linacre.csv, déf 0.23)
  ETP    = poids·((0.75-albedo)·tmer + ea)·Delta/(Delta+Gamma)   [poids=1 en 24h]
  etp_uhrh = max(ETP, 0) · coeff_multiplicatif    (somme des classes, pcts→1)
"""
from __future__ import annotations
import torch


def linacre_etp(tmin, tmax, lat_dd, alti_m, couvert_nival, albedo_neige,
                t_froid=-10.0, t_chaud=20.0, albedo=0.23, coeff=1.0):
    """ETP Linacre-Hydrotel (mm/j). Tous tensors (n,) ou scalaires broadcastés."""
    tmoy = (tmin + tmax) / 2.0
    dtmxn = (t_chaud - t_froid) if not torch.is_tensor(t_chaud) else (t_chaud - t_froid)
    corn = 24.41 / (100.0 - lat_dd)
    e_tmoy = 0.6108 * torch.exp(17.27 * tmoy / (tmoy + 237.3))
    delta = 4098.0 * e_tmoy / (tmoy + 237.3) ** 2
    p_atm = 101.3 * ((293.0 - 0.0065 * alti_m) / 293.0) ** 5.26
    gamma = 0.000665 * p_atm
    tmer = corn * (tmoy + 0.006 * alti_m)
    ea = 0.3807 * (0.0023 * alti_m + 0.37 * tmoy + 0.53 * (tmax - tmin) + 0.35 * dtmxn - 10.9)
    alb = torch.where(couvert_nival > 0.0, albedo_neige, albedo * torch.ones_like(tmoy))
    etp = ((0.75 - alb) * tmer + ea) * delta / (delta + gamma)
    return torch.clamp(etp, min=0.0) * coeff


def load_linacre_params(sim_dir, ids):
    """linacre.csv → tensors (t_froid, t_chaud, albedo, coeff) alignés sur ids."""
    d = {}
    for ln in open(f"{sim_dir}/linacre.csv", encoding="latin-1").read().splitlines():
        c = ln.split(";")
        if len(c) >= 5 and c[0].strip().isdigit():
            d[int(c[0])] = [float(x) for x in c[1:5]]
    def col(j, default):
        return torch.tensor([d[u][j] if u in d else default for u in ids], dtype=torch.get_default_dtype())
    return col(0, -10.0), col(1, 20.0), col(2, 0.23), col(3, 1.0)
