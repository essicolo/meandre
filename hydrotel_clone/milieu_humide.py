"""Clone FIDÈLE du milieu humide isolé d'Hydrotel (réservoir type SWAT), porté
ligne-à-ligne du C++ (bv3c2.cpp:1460 CalculMilieuHumideIsole, init
milieu_humide_isole.cpp:34), vectorisé, différentiable. Sous-projet PROPRE.

Le milieu humide isolé intercepte la production du sol sur sa fraction de
drainage, la stocke dans un réservoir dont la surface dépend du volume
(wetsa = B·vol^A), perd de l'eau par évaporation et infiltration (seepage), et
relâche le surplus au-dessus du volume normal (wetnvol) avec une constante de
temps c_prod, ou déborde au-dessus du volume max (wetmxvol). La sortie
(débordement + seepage) retourne à la production de l'UHRH.

A et B sont dérivés de la géométrie (init) :
  wetmxsa = wet_a·1e6 ; wetmxvol = wetdmax·wetmxsa
  wet_nsa = frac·wetmxsa ; wetnvol = wetdnor·wet_nsa
  A = (log10(wetmxsa)-log10(wet_nsa)) / (log10(wetmxvol)-log10(wetnvol))
  B = wetmxsa / wetmxvol^A
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


def init_wetland_geom(wet_a_km2, wetdmax, frac, wetdnor):
    """Dérive (A, B, wetnvol, wetmxvol) de la géométrie (milieu_humide_isole.cpp:56-63).
    wet_a_km2 = superficie du milieu humide [km2]."""
    wetmxsa = wet_a_km2 * 1.0e6              # m2
    wetmxvol = wetdmax * wetmxsa            # m3
    wet_nsa = frac * wetmxsa                # m2
    wetnvol = wetdnor * wet_nsa            # m3
    A = (math.log10(wetmxsa) - math.log10(wet_nsa)) / (math.log10(wetmxvol) - math.log10(wetnvol))
    B = wetmxsa / wetmxvol ** A
    return A, B, wetnvol, wetmxvol


def wetland_geom_vec(wet_a_km2, wetdmax, frac, wetdnor):
    """Version VECTORISÉE (tenseurs par nœud) de init_wetland_geom, différentiable.
    Suppose wet_a_km2 > 0 partout (les nœuds sans milieu humide doivent passer une
    valeur factice positive et être masqués en aval — sinon log10(0)=−inf → NaN)."""
    wetmxsa = wet_a_km2 * 1.0e6
    wetmxvol = wetdmax * wetmxsa
    wet_nsa = frac * wetmxsa
    wetnvol = wetdnor * wet_nsa
    A = (torch.log10(wetmxsa) - torch.log10(wet_nsa)) / (torch.log10(wetmxvol) - torch.log10(wetnvol))
    B = wetmxsa / wetmxvol.pow(A)
    return A, B, wetnvol, wetmxvol


def calcul_milieu_humide_isole(wet_vol, apport_mm, evp_mm, prod_mm, hru_ha, wet_fr,
                               A, B, wetnvol, wetmxvol, wet_k, c_ev, c_prod, pdt=24):
    """Un pas de temps (bv3c2.cpp:1460). wet_vol [m3] = volume au DÉBUT du pas.
    apport_mm = apport (neige) DÉJÀ ×wetfr ; evp_mm = ETP totale ; prod_mm =
    production sol (surf+hypo+base) ; hru_ha = aire UHRH [ha] ; wet_fr = fraction
    drainée (wetdrafr) ; wet_k = ksat_bs [mm/h] ; pdt = pas [h].
    Retourne (wet_vol_new, wetsep, wetflwi, wetflwo, wetprod_mm)."""
    wetsa = B * wet_vol.pow(A) / 10000.0                  # surface [ha]
    wetev = 10.0 * c_ev * evp_mm * wetsa
    wetsep = wet_k * wetsa * (pdt * 10.0)
    wetpcp = apport_mm * wetsa * 10.0
    wetflwi = prod_mm * 10.0 * (hru_ha * wet_fr - wetsa)

    wet_vol = wet_vol - wetsep - wetev + wetflwi + wetpcp

    # ajustement si volume négligeable (l.1487-1497)
    low = wet_vol < 0.001
    wetsep = torch.where(low, wetsep + wet_vol, wetsep)
    wet_vol = torch.where(low, torch.zeros_like(wet_vol), wet_vol)
    negsep = low & (wetsep < 0.0)
    wetev = torch.where(negsep, wetev + wetsep, wetev)
    wetsep = torch.where(negsep, torch.zeros_like(wetsep), wetsep)

    # débordement / relâche (l.1502-1514)
    above = wet_vol > wetnvol
    under_max = wet_vol <= wetmxvol
    flwo_norm = (wet_vol - wetnvol) / c_prod
    flwo_over = wet_vol - wetmxvol
    wetflwo = torch.where(above, torch.where(under_max, flwo_norm, flwo_over),
                          torch.zeros_like(wet_vol))
    wet_vol = torch.where(above, torch.where(under_max, wet_vol - flwo_norm,
                                             torch.full_like(wet_vol, float('nan'))), wet_vol)
    # cas débordement : wet_vol = wetmxvol
    wet_vol = torch.where(above & ~under_max, torch.as_tensor(wetmxvol, dtype=wet_vol.dtype), wet_vol)

    wetprod_mm = wetflwo / (hru_ha * 10.0) + wetsep / (hru_ha * 10.0)
    return wet_vol, wetsep, wetflwi, wetflwo, wetprod_mm
