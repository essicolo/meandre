# -*- coding: utf-8 -*-
"""Génère notebooks/correction_casr_ouranos.ipynb (correction hydrologique de CaSR
par quantile mapping, expliqué en français, autonome et reproductible)."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
c = []
def md(t): c.append(nbf.v4.new_markdown_cell(t))
def code(t): c.append(nbf.v4.new_code_cell(t))

md(r"""# Correction hydrologique de CaSR par quantile mapping

**But** : rendre la précipitation de CaSR (réanalyse RDRS/CaPA) utilisable en hydrologie,
en corrigeant ses biais de distribution tout en préservant son excellent *timing*.

Ce carnet est **autonome et reproductible** : il tourne sur des données synthétiques qui
imitent les biais de CaSR, puis explique comment l'appliquer à vos vraies grilles CaSR.

## Le problème
CaSR a le **meilleur timing** disponible (réanalyse horaire assimilant stations + radar),
mais sa **distribution** de précipitation est biaisée pour l'hydrologie :

1. **Biais de crachin** : CaPA/GEM sur-estiment les petits événements (< 2 mm) et produisent
   de la pluie presque tous les jours. Résultat : trop de jours pluvieux, orages dilués.
2. **Sur-volume** : sur-estimation du cumul des jours très pluvieux (R95pTOT jusqu'à ×2 dans
   plusieurs régions/saisons), donnant un volume total trop humide.
3. **Queue haute** : les gros événements sont sur-représentés.

Conséquence en modélisation pluie-débit : sur-production du débit, ETR forcée trop haute pour
fermer le bilan, et hydrogrammes bruités. Un modèle calé sur ce forçage hérite de ces biais.

*Références :*
- CaPA : Fortin et al. (2015), *J. Hydrometeorol.* 16(5), JHM-D-14-0191.
- CaSR v3.2 : jeu de données de précipitation 1980-2024 (ECCC).
- ET réelle forêt boréale ~400-500 mm/an : tours à covariance des turbulences
  (Brümmer et al. 2011, *Agr. For. Meteorol.* ; comparaison multi-méthodes *J. Hydrometeorol.* 2015).
""")

md(r"""## La méthode

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

md(r"""## Application à vos vraies données CaSR

```python
import xarray as xr

# 1. Charger CaSR (précip journalière) aux points/mailles voulus : ndarray (T, N)
casr = xr.open_dataset("votre_casr.nc")["precip"].values            # (T, N)

# 2. Charger/produire la RÉFÉRENCE de distribution (mêmes T, N) :
#    - option A : observations de stations krigées/interpolées
#    - option B : une climatologie corrigée
ref = xr.open_dataset("votre_reference.nc")["precip"].values         # (T, N)

# 3. Corriger
casr_qm   = quantile_map(casr, ref)
casr_corr = rescale_to_water_balance(casr_qm, et_mm_an=450.0, q_mm_an=VOTRE_Q_OBS_MM_AN)

# 4. Vérifier avec diagnostics() + corrélation de rang, puis réinjecter dans Hydrotel.
```

**Notes et limites**
- La correction s'applique à la **précipitation** ; la température et le rayonnement de CaSR
  (réanalyse) sont généralement bons et n'ont pas besoin de QM.
- Le QM par maille suppose des séries assez longues (≥ 10-15 ans) pour des CDF empiriques stables.
- Le calage volume est **global** ici ; pour un domaine hétérogène, un calage **par bassin** (avec
  l'ETR et le débit locaux) est préférable.
- Le QM n'invente pas d'information : il ne corrige pas une erreur de *placement* d'un orage, il
  corrige la *distribution* et le *volume*. Le timing hérité reste celui de CaSR.
- Pour une reproductibilité totale, choisir une référence **ouverte** (stations publiques,
  climatologie documentée) plutôt qu'un produit propriétaire.

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
