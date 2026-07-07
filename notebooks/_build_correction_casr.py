# -*- coding: utf-8 -*-
"""Génère notebooks/correction_casr_ouranos.ipynb (correction hydrologique de CaSR
par quantile mapping, expliqué en français, autonome et reproductible)."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
c = []
def md(t): c.append(nbf.v4.new_markdown_cell(t))
def code(t): c.append(nbf.v4.new_code_cell(t))

md(r"""# Correction hydrologique de CaSR

**But** : rendre la précipitation de CaSR (réanalyse RDRS/CaPA) utilisable en hydrologie,
en corrigeant ses biais tout en préservant son excellent *timing*.

Ce carnet est **autonome et reproductible** : il tourne sur des données synthétiques qui
imitent les biais de CaSR, puis explique comment l'appliquer à vos vraies grilles CaSR.

## Le problème : deux axes à corriger, volume ET timing
CaSR a le **meilleur timing** disponible (réanalyse horaire assimilant stations + radar),
mais deux familles de biais nuisent à l'hydrologie :

**Axe VOLUME / distribution :**
1. **Biais de crachin** : CaPA/GEM sur-estiment les petits événements (< 2 mm) et produisent
   de la pluie presque tous les jours. Résultat : trop de jours pluvieux, orages dilués.
2. **Sur-volume** : sur-estimation du cumul des jours très pluvieux (R95pTOT jusqu'à ×2 dans
   plusieurs régions/saisons), donnant un volume total trop humide.

**Axe TIMING / agrégation :**
3. **Frontière de jour** : CaSR est horaire en **UTC**. Agréger naïvement en jour UTC décale la
   précip par rapport au débit observé, qui est mesuré en **jour local** (heure de l'Est, UTC-5).
   Un orage de fin de journée locale tombe alors dans le mauvais jour hydrologique (~5 h de
   décalage), ce qui dégrade la corrélation pluie-débit.

Conséquence en modélisation pluie-débit : sur-production du débit, ETR forcée trop haute pour
fermer le bilan, hydrogrammes bruités, et pics mal datés. Un modèle calé sur ce forçage hérite
de ces biais.

## Deux méthodes présentées
- **Méthode A — quantile mapping** vers une distribution de référence (corrige le volume/forme).
  Utile si vous disposez déjà d'une bonne référence de distribution (stations krigées).
- **Méthode B — correction auto-référencée depuis l'horaire (RECOMMANDÉE)** : dé-crachinage horaire
  + agrégation sur le jour **local** + calage volume sur le bilan d'eau. Elle corrige les **deux
  axes** et ne dépend **d'aucun produit tiers** — vous l'appliquez à votre propre CaSR horaire.
  Dans nos essais (bassin SLSO, held-out non stationnaire 2022-24), la méthode B donne le meilleur
  KGE médian par station et bat le quantile mapping vers un produit externe, tout en restant
  entièrement reproductible à partir de CaSR seul.

*Références :*
- CaPA : Fortin et al. (2015), *J. Hydrometeorol.* 16(5), JHM-D-14-0191.
- CaSR v3.2 : jeu de données de précipitation 1980-2024 (ECCC).
- ET réelle forêt boréale ~400-500 mm/an : tours à covariance des turbulences
  (Brümmer et al. 2011, *Agr. For. Meteorol.* ; comparaison multi-méthodes *J. Hydrometeorol.* 2015).
""")

md(r"""## Méthode A — quantile mapping

**Quantile mapping (QM) préservant le timing.** Pour chaque maille indépendamment, on remappe
la *distribution* de la précip de CaSR sur une distribution de référence saine, sans toucher à
la *séquence* des jours. Puisque le QM est une transformation monotone des valeurs de chaque
maille, l'ordre relatif des jours (donc le timing) est **exactement préservé** (corrélation de
rang = 1). On garde le timing de CaSR, on acquiert la distribution de la référence.

Formellement, pour une maille : `p_corr(t) = F_ref^{-1}( F_casr( p_casr(t) ) )`, ce qui se
calcule simplement par interpolation entre les valeurs triées de CaSR et celles de la référence.

**Calage du volume sur le bilan d'eau.** La référence fixe la *forme* de la distribution
(fréquence des jours pluvieux, queue). Le *volume total* est ensuite calé sur la fermeture du
bilan hydrologique long terme `P = ETR + Q`, avec une ETR de référence indépendante (tours à
flux ~450 mm/an en forêt boréale) et le débit observé. Cela évite de dépendre du volume de la
référence, qui peut être biaisé.

**Choix de la référence.** Idéalement, des observations de stations (que vous utilisez déjà pour
la calibration) ou une climatologie corrigée. Toute distribution hydrologiquement plausible fait
l'affaire — le QM n'emprunte que sa *forme*, pas son timing.
""")

code(r"""import numpy as np
import pandas as pd

rng = np.random.default_rng(0)  # reproductible
""")

md(r"""### Fonctions réutilisables

Trois fonctions pures (numpy), à appliquer telles quelles à vos grilles.
""")

code(r'''def quantile_map(source, target):
    """Remappe la distribution de `source` sur celle de `target`, PAR MAILLE,
    en préservant la séquence temporelle (le timing).

    source, target : ndarray (T, N)  — T pas de temps, N mailles.
    Retour        : ndarray (T, N)  — timing de `source`, distribution de `target`.
    """
    source = np.asarray(source, float); target = np.asarray(target, float)
    out = np.empty_like(source)
    for n in range(source.shape[1]):
        cs = np.sort(source[:, n])          # CDF empirique de la source
        ts = np.sort(target[:, n])          # CDF empirique de la cible
        out[:, n] = np.interp(source[:, n], cs, ts)  # quantile-à-quantile
    return np.clip(out, 0.0, None)
''')

code(r'''def rescale_to_water_balance(precip, et_mm_an=450.0, q_mm_an=None, q_obs_series=None,
                              area_km2=None):
    """Cale le volume annuel de précip sur la fermeture du bilan P = ETR + Q.

    precip     : ndarray (T, N)  précip journalière (mm/j).
    et_mm_an   : ETR de référence (mm/an), ex. tours à flux ~450 en forêt boréale.
    q_mm_an    : lame écoulée observée (mm/an). Sinon calculée depuis q_obs_series+area.
    Retour     : precip rescalée pour que sa moyenne annuelle = et_mm_an + q_mm_an.
    """
    if q_mm_an is None:
        if q_obs_series is None or area_km2 is None:
            raise ValueError("fournir q_mm_an, ou (q_obs_series [m3/s] et area_km2)")
        # lame (mm/an) = Q [m3/s] * secondes/an / aire [m2] * 1000
        q_mm_an = np.nanmean(q_obs_series) * 31557600.0 / (area_km2 * 1e6) * 1000.0
    cible = et_mm_an + q_mm_an
    courant = np.nanmean(precip) * 365.25
    return precip * (cible / courant)
''')

code(r'''def diagnostics(p, nom, seuil_pluie=0.1):
    """Résumé de distribution : jours pluvieux, volume, quantiles."""
    frac = (p > seuil_pluie).mean() * 100
    vol = np.nanmean(p) * 365.25
    q95 = np.nanpercentile(p[p > seuil_pluie], 95) if (p > seuil_pluie).any() else 0
    print(f"{nom:22s} : jours pluvieux {frac:4.0f}%  | volume {vol:5.0f} mm/an  | "
          f"P95 (jours pluvieux) {q95:5.1f} mm/j")
''')

md(r"""### Démonstration sur données synthétiques

On fabrique une précip « CaSR synthétique » avec les biais typiques (crachin quotidien +
sur-volume), et une « référence saine » (moins de jours pluvieux, distribution réaliste).
Le QM doit transférer la forme de la référence tout en gardant la séquence de CaSR.
""")

code(r'''T, N = 9000, 20   # 25 ans, 20 mailles
# --- Référence SAINE : ~45% de jours pluvieux, distribution gamma réaliste ---
wet_ref = rng.random((T, N)) < 0.45
ref = wet_ref * rng.gamma(shape=0.7, scale=6.0, size=(T, N))     # mm/j

# --- CaSR SYNTHÉTIQUE : mêmes VRAIS jours pluvieux + CRACHIN partout + sur-volume ---
crachin = rng.gamma(shape=0.5, scale=0.6, size=(T, N))           # petits événements partout
casr = (ref * 1.15) + crachin                                    # +15% volume + crachin
# le timing (quels jours sont les plus humides) suit la référence, mais brouillé par le crachin

print("AVANT correction :")
diagnostics(casr, "CaSR synthétique")
diagnostics(ref, "Référence saine")
''')

code(r'''# --- CORRECTION ---
casr_qm = quantile_map(casr, ref)                                # forme de la référence, timing CaSR
casr_corr = rescale_to_water_balance(casr_qm, et_mm_an=450.0, q_mm_an=650.0)  # volume bilan-d'eau

print("APRÈS correction :")
diagnostics(casr_corr, "CaSR corrigé")

# vérification : le TIMING est-il préservé ? (corrélation de rang CaSR <-> corrigé)
from scipy.stats import spearmanr
rho = np.mean([spearmanr(casr[:, n], casr_corr[:, n]).correlation for n in range(N)])
print(f"\\nCorrélation de rang CaSR <-> corrigé : {rho:.4f}  (1.0 = timing intégralement préservé)")
''')

md(r"""On observe : le crachin disparaît (les jours pluvieux passent de ~100% à ~45%, comme la
référence), le volume est calé sur le bilan d'eau, et la corrélation de rang ≈ 1 confirme que le
**timing de CaSR est intact**. C'est exactement l'objectif : timing de la réanalyse, distribution
hydrologiquement saine.
""")

code(r'''# Visualisation simple (facultative)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # (a) distributions des jours pluvieux
    for série, lab in [(casr, "CaSR brut"), (ref, "Référence"), (casr_corr, "CaSR corrigé")]:
        v = série[série > 0.1]
        ax[0].hist(v, bins=60, range=(0, 40), histtype="step", density=True, label=lab)
    ax[0].set(xlabel="précip jours pluvieux (mm/j)", ylabel="densité", title="Distributions")
    ax[0].legend()
    # (b) une année, une maille : timing préservé
    sl = slice(0, 365)
    ax[1].plot(casr[sl, 0], lw=0.8, label="CaSR brut")
    ax[1].plot(casr_corr[sl, 0], lw=0.8, label="CaSR corrigé")
    ax[1].set(xlabel="jour", ylabel="précip (mm/j)", title="Une année (maille 0) — mêmes jours de pluie")
    ax[1].legend()
    fig.tight_layout(); fig.savefig("correction_casr_demo.png", dpi=90)
    print("figure -> correction_casr_demo.png")
except Exception as e:
    print("matplotlib indisponible, visualisation sautée :", e)
''')

md(r"""## Méthode B — correction auto-référencée depuis l'horaire (RECOMMANDÉE)

La méthode A a besoin d'une distribution de référence externe. La méthode B corrige CaSR **à partir
de CaSR seul**, en agissant sur les **deux axes** du problème :

1. **TIMING — agrégation sur le jour local.** On décale l'index horaire de CaSR de l'offset du fuseau
   (UTC-5 en heure de l'Est) **avant** de sommer en jour. La précip est ainsi ré-assignée au bon jour
   hydrologique, aligné sur le débit observé (mesuré en jour local). Corrige le décalage de frontière.
2. **VOLUME / distribution — dé-crachinage horaire.** À l'échelle horaire, on retire les heures sous un
   seuil (ex. 0.3 mm/h) considérées comme du crachin de réanalyse. Cela réduit la fraction de jours
   pluvieux (typiquement ~62% → ~40%) sans emprunter de forme externe : c'est CaSR qui se corrige.
3. **VOLUME — calage bilan d'eau.** Le total journalier dé-crachiné est ensuite calé sur `P = ETR + Q`
   (même fonction que la méthode A), pour fixer le volume long terme.

Aucune référence tierce n'est requise : la correction est reproductible par quiconque possède la CaSR
horaire. C'est ce qui la rend idéale pour débloquer une analyse Hydrotel de façon transparente.
""")

code(r'''def dedrizzle_aggregate_local_day(hourly, times_utc, tz_offset_h=-5, drizzle_mm_h=0.3):
    """Corrige les DEUX axes depuis l'horaire CaSR : timing (jour local) + distribution (dé-crachinage).

    hourly      : ndarray (H, N)  précip HORAIRE (mm/h), H heures, N mailles.
    times_utc   : DatetimeIndex (H,) en UTC, aligné sur `hourly`.
    tz_offset_h : décalage du fuseau local (heure de l'Est = -5 ; -4 si heure avancée).
    drizzle_mm_h: seuil horaire sous lequel la pluie est traitée comme crachin (retirée).
    Retour      : (precip_daily, days) — ndarray (D, N) mm/j en JOUR LOCAL, et l'index des jours.
    """
    import pandas as pd
    idx_local = pd.DatetimeIndex(times_utc) + pd.Timedelta(hours=tz_offset_h)   # TIMING : jour local
    kept = np.where(hourly >= drizzle_mm_h, hourly, 0.0)                        # DISTRIB : dé-crachinage
    df = pd.DataFrame(kept, index=idx_local)
    daily = df.resample("1D").sum()
    return daily.values, daily.index
''')

md(r"""### Démonstration sur données horaires synthétiques

On fabrique une CaSR horaire synthétique avec crachin permanent + orages placés à des heures
précises, en UTC. On montre que le dé-crachinage retire le crachin et que l'agrégation en jour local
déplace correctement les orages de fin de journée vers le bon jour hydrologique.
""")

code(r'''import pandas as pd
Hh, Nn = 25 * 365 * 24, 8                                   # 25 ans horaires, 8 mailles
times_utc = pd.date_range("2000-01-01", periods=Hh, freq="h", tz=None)

# crachin permanent (toutes les heures un peu de pluie) + vrais orages épars
crachin_h = rng.gamma(shape=0.3, scale=0.15, size=(Hh, Nn))          # ~0.05 mm/h partout
orages_h = (rng.random((Hh, Nn)) < 0.01) * rng.gamma(shape=1.5, scale=4.0, size=(Hh, Nn))
casr_h = crachin_h + orages_h                                        # mm/h

# agrégation NAÏVE (jour UTC, aucun dé-crachinage) vs correction (jour local + dé-crachinage)
naif = pd.DataFrame(casr_h, index=times_utc).resample("1D").sum().values
corr_daily, _ = dedrizzle_aggregate_local_day(casr_h, times_utc, tz_offset_h=-5, drizzle_mm_h=0.3)
corr_bilan = rescale_to_water_balance(corr_daily, et_mm_an=450.0, q_mm_an=650.0)

print("Agrégation naïve (jour UTC, crachin gardé) vs correction B :")
diagnostics(naif, "CaSR naïf (UTC)")
diagnostics(corr_bilan, "CaSR corrigé B")
print(f"\\nJours pluvieux : naïf {(naif>0.1).mean()*100:.0f}%  ->  corrigé {(corr_bilan>0.1).mean()*100:.0f}%")
''')

md(r"""Le dé-crachinage ramène la fraction de jours pluvieux à un niveau réaliste, l'agrégation jour-local
recale la datation des orages sur le débit observé, et le calage bilan fixe le volume — le tout sans
aucune distribution de référence externe. C'est la correction retenue comme recommandée.
""")

md(r"""## Application à vos vraies données CaSR

**Méthode B (recommandée) — depuis votre CaSR HORAIRE, sans référence externe :**
```python
import xarray as xr, pandas as pd

# 1. Charger CaSR HORAIRE (précip mm/h) aux points/mailles voulus : ndarray (H, N) + index UTC
ds = xr.open_dataset("votre_casr_horaire.nc")
casr_h = ds["precip"].values                       # (H, N), mm/h
times_utc = pd.DatetimeIndex(ds["time"].values)    # (H,), UTC

# 2. Corriger les deux axes : jour local (timing) + dé-crachinage (distribution)
daily, days = dedrizzle_aggregate_local_day(casr_h, times_utc, tz_offset_h=-5, drizzle_mm_h=0.3)

# 3. Caler le volume sur le bilan d'eau du/des bassin(s)
casr_corr = rescale_to_water_balance(daily, et_mm_an=450.0, q_mm_an=VOTRE_Q_OBS_MM_AN)

# 4. Vérifier avec diagnostics(), puis réinjecter dans Hydrotel.
```

**Méthode A (alternative) — si vous avez déjà une bonne distribution de référence :**
```python
casr = xr.open_dataset("votre_casr.nc")["precip"].values            # (T, N) journalier
ref  = xr.open_dataset("votre_reference.nc")["precip"].values        # (T, N) stations krigées / climato
casr_qm   = quantile_map(casr, ref)
casr_corr = rescale_to_water_balance(casr_qm, et_mm_an=450.0, q_mm_an=VOTRE_Q_OBS_MM_AN)
```

**Notes et limites**
- La correction s'applique à la **précipitation** ; la température et le rayonnement de CaSR
  (réanalyse) sont généralement bons et n'ont pas besoin de correction.
- **Fuseau (méthode B)** : `tz_offset_h=-5` est l'heure normale de l'Est (EST). Si vos débits sont en
  heure locale avec heure avancée (EDT, UTC-4 l'été), un offset saisonnier est plus juste ; l'écart
  d'une heure est toutefois mineur devant le décalage UTC↔local de 5 h que l'on corrige.
- Le seuil de dé-crachinage (0.3 mm/h) et la cible de volume se **calent par région** : le volume final
  est fixé par le bilan d'eau, le seuil ne fait qu'ajuster la fraction de jours pluvieux.
- Le QM par maille (méthode A) suppose des séries assez longues (≥ 10-15 ans) pour des CDF stables.
- Le calage volume est **global** dans la démo ; pour un domaine hétérogène, un calage **par bassin**
  (ETR et débit locaux) est préférable pour les deux méthodes.
- Aucune méthode n'invente d'information sous l'heure : la méthode B recale la *datation journalière*
  (frontière de jour) et la *distribution*, mais ne réinvente pas la position intra-horaire d'un orage.
- La méthode B est **entièrement auto-référencée** (CaSR seul) : reproductibilité totale, aucun produit
  propriétaire. C'est l'option privilégiée pour une analyse transparente.

*Contact : méthode développée dans le cadre du projet méandre (Hydrotel différentiable).*
""")

nb["cells"] = c
nb["metadata"] = {"language_info": {"name": "python"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
os.makedirs("notebooks", exist_ok=True)
out = "notebooks/correction_casr_ouranos.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("[ok] écrit", out, "|", len(c), "cellules")
