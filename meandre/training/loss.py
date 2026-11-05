"""Multi-objective differentiable loss for hydrological calibration.

All terms are smooth functions of Q_sim so gradients flow back through the
full model. Use meandre.utils.metrics for non-differentiable evaluation.

L = w1*(1-NSE) + w2*|PBIAS|/100 + w3*(1-KGE)
  + w4*L_snow  + w5*L_ET
  + w6*L_physics
"""

import logging

import os
import torch
import torch.nn as nn

from meandre.utils import settings
from torch import Tensor

logger = logging.getLogger(__name__)


def differentiable_nse_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """1 - NSE, formulated as a minimization target. Perfect = 0."""
    num = ((q_obs - q_sim) ** 2).sum()
    denom = ((q_obs - q_obs.mean()) ** 2).sum()
    return num / (denom + 1e-8)


def _kge_components(q_obs: Tensor, q_sim: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute KGE and its three components (r, beta, gamma).

    Returns:
        r:     Pearson correlation
        beta:  bias ratio (mu_sim / mu_obs)
        gamma: variability ratio (cv_sim / cv_obs)
        kge:   Kling-Gupta efficiency
    """
    x = q_obs - q_obs.mean()
    y = q_sim - q_sim.mean()
    var_obs = (x ** 2).mean()
    var_sim = (y ** 2).mean()
    # PLANCHER RELATIF sur les ecarts-types (2026-09-04). Sur une fenetre de debit
    # quasi constant (janvier sous glace, 34 jours, aucun historique au premier bloc),
    # var -> 0 et le plancher absolu de 1e-8 donnait std = 1e-4 : r = cov/1e-8 et
    # gamma = std_sim/std_obs explosaient, leur gradient devenait inf, puis NaN. Or le
    # trainer accumule les gradients de TOUS les blocs et ne fait qu'UN pas par epoque :
    # un seul bloc NaN mettait a zero le gradient accumule de tout le champ spatial, qui
    # n'apprenait donc RIEN de l'epoque (validation identique a la quatrieme decimale
    # d'une epoque a l'autre, perte qui ne bougeait que par le prior). Un plancher de
    # 0.1 % du debit moyen rend le gradient fini partout, sans effet mesurable ailleurs.
    # Fraction declaree : loss_terms.kge_std_floor.
    _floor = float(settings.get("loss_terms.kge_std_floor"))
    floor_obs = (_floor * q_obs.mean().abs()).clamp(min=1e-6)
    floor_sim = (_floor * q_sim.mean().abs()).clamp(min=1e-6)
    std_obs = torch.sqrt(var_obs + floor_obs ** 2)
    std_sim = torch.sqrt(var_sim + floor_sim ** 2)
    r = (x * y).mean() / (std_obs * std_sim)
    r = r.clamp(-1.0, 1.0)

    mu_obs = q_obs.mean().clamp(min=1e-8)
    beta = q_sim.mean() / mu_obs
    gamma = (std_sim / std_obs) / (beta.abs().clamp(min=1e-8))

    sq = (r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
    kge = 1.0 - torch.sqrt(sq + 1e-8)
    return r, beta, gamma, kge


def differentiable_kge_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """1 - KGE, formulated as a minimization target. Perfect = 0."""
    _, _, _, kge = _kge_components(q_obs, q_sim)
    return 1.0 - kge


def differentiable_peak_ratio_loss(q_obs: Tensor, q_sim: Tensor, seuil: Tensor) -> Tensor:
    """Rapport des pointes simulées aux pointes observées, en carré du logarithme.

    REMPLACE le terme de pics en écart quadratique, dont il a été mesuré le 2026-09-20 qu'il
    RÉCOMPENSE l'aplatissement sur 78 % des stations. Celui-là est un écart jour par jour
    restreint aux hauts débits, c'est-à-dire à l'endroit exact où un décalage d'une journée
    coûte le plus cher ; lisser y réduit l'erreur plus qu'ailleurs, et il paie pour cela.

    Cette forme compare des STATISTIQUES de magnitude et ne peut pas être dupée par un
    décalage : lisser abaisse la pointe simulée, et le terme le voit. Mesuré sur 77 stations,
    trois seuils et trois durées de lissage, il ne préfère JAMAIS le lissage au retard.

    Le seuil vient de l'OBSERVATION et ne bouge pas avec la simulation, sans quoi un modèle
    plat déplacerait son propre seuil. Le logarithme rend la pénalité symétrique entre
    sur-estimation et sous-estimation, une pointe deux fois trop forte coûtant autant qu'une
    pointe deux fois trop faible.
    """
    high = q_obs >= seuil
    if high.sum() < 10:
        return torch.zeros((), device=q_sim.device, dtype=q_sim.dtype)
    mean_obs = q_obs[high].mean().clamp(min=1e-8)
    mean_sim = q_sim[high].mean().clamp(min=1e-8)
    # VALEUR ABSOLUE DU LOGARITHME, non son carre : le carre repondrait au second ordre pour
    # une petite erreur de pointe, defaut mesure le meme jour sur les facteurs du KGE et sur
    # le terme d'etiage. La symetrie entre sur-estimation et sous-estimation est conservee,
    # |log(r)| valant |log(1/r)|.
    return torch.abs(torch.log(mean_sim / mean_obs))


def differentiable_recession_loss(q_obs: Tensor, q_sim: Tensor, delay: int = 2,
                                  min_duration: int = 5) -> Tensor:
    """Écart sur la VITESSE DE VIDANGE, mesurée sur les jours de décrue franche.

    POURQUOI CE TERME EXISTE. La colonne a deux chemins lents, l'écoulement hypodermique et
    la nappe, et aucune observation ne les séparait. Mesuré le 2026-09-21 : à volume lent
    constant, basculer tout ce volume d'un chemin à l'autre déplace l'indice d'écoulement de
    base d'Eckhardt de six millièmes seulement, parce que ce filtre ajuste sa constante de
    récession sur la série et absorbe l'information. La même bascule déplace la constante de
    récession de 0,966 à 0,735. C'est donc la FORME de la vidange qui identifie la partition,
    et aucun terme de la perte ne la regardait.

    LA GRANDEUR. Sur les jours où l'OBSERVATION décroît franchement, on prend la moyenne du
    logarithme du rapport d'un jour au précédent : c'est l'opposé de l'inverse du temps de
    vidange. Le masque vient de l'observation et ne bouge donc pas avec la simulation, sans
    quoi un modèle plat déplacerait ses propres jours de décrue. Le délai après la pointe
    écarte le ressuyage rapide, dont la constante est celle du versant et non de la nappe.

    DEUX BANDES. La vidange est plus rapide aux hauts débits qu'aux bas, d'un facteur cinq à
    six sur les hydrogrammes québécois, et c'est cet étalement qui porte l'information. Une
    moyenne unique serait dominée par la bande la mieux représentée : on sépare donc les jours
    de décrue par la médiane du débit observé et on somme les deux écarts.

    ÉCART ABSOLU, premier ordre, conformément à l'audit de tous les termes du 2026-09-21.
    """
    if q_obs.numel() < 20:
        return torch.zeros((), device=q_sim.device, dtype=q_sim.dtype)
    eps = (1e-3 * q_obs.median().abs()).clamp(min=1e-8)
    decreasing = q_obs[1:] < q_obs[:-1]
    # Un jour compte s'il decroit ET si les `delai` jours precedents decroissent aussi : on
    # ecarte ainsi mecaniquement le sommet de la pointe et le ressuyage qui le suit.
    guard = decreasing.clone()
    for k in range(1, delay + 1):
        guard[k:] = guard[k:] & decreasing[:-k]
    if int(guard.sum()) < min_duration * 4:
        return torch.zeros((), device=q_sim.device, dtype=q_sim.dtype)
    ratio_obs = torch.log((q_obs[1:] + eps) / (q_obs[:-1] + eps))
    ratio_sim = torch.log((q_sim[1:] + eps) / (q_sim[:-1] + eps))
    level_ = q_obs[:-1]
    seuil = level_[guard].median()
    loss_value = torch.zeros((), device=q_sim.device, dtype=q_sim.dtype)
    for band in (guard & (level_ >= seuil), guard & (level_ < seuil)):
        if int(band.sum()) < min_duration:
            continue
        loss_value = loss_value + torch.abs(ratio_sim[band].mean() - ratio_obs[band].mean())
    return loss_value


def differentiable_r_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Premier facteur du KGE, isolé : 1 − r, l'erreur de CALENDRIER. Parfait = 0."""
    r, _, _, _ = _kge_components(q_obs, q_sim)
    return 1.0 - r


# ECART ABSOLU ET NON CARRE (correction du 2026-09-20). La premiere version portait ces deux
# facteurs au CARRE, ce qui les rend du second ordre : pour une petite erreur sur un seul
# facteur, un carre repond en e^2 quand le KGE composite, qui passe par une racine de somme de
# carres, repond en |e|. Mesure sur 32 stations, prelevement estival de cinq pour cent du debit
# moyen : la forme au carre bouge de 0,0002 et la forme absolue de 0,0126, soit soixante-trois
# fois plus. Prendre des carres aurait donc rendu la perte AVEUGLE a la grandeur meme que le
# projet doit predire, tout en pretendant l'exposer. L'ecart absolu garde la sensibilite au
# premier ordre du composite et lui ajoute la separation des trois causes.
def differentiable_beta_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Deuxième facteur du KGE, isolé : |beta − 1|, l'erreur de VOLUME. Parfait = 0."""
    _, beta, _, _ = _kge_components(q_obs, q_sim)
    return torch.abs(beta - 1.0)


def differentiable_gamma_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Troisième facteur du KGE, isolé : |gamma − 1|, l'erreur d'AMPLITUDE. Parfait = 0."""
    _, _, gamma, _ = _kge_components(q_obs, q_sim)
    return torch.abs(gamma - 1.0)


def differentiable_composite_kge_loss(
    q_obs: Tensor, q_sim: Tensor, alpha: float = 0.5, eps: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Composite KGE loss: alpha*(1-KGE) + (1-alpha)*(1-KGE_log).

    KGE on normal flows captures peaks; KGE on log flows captures baseflow.
    Reference: Pool et al. (2018), Kratzert et al. (2019).

    Returns:
        loss: scalar
        info: dict with r, beta, gamma, kge (normal), r_log, beta_log, gamma_log, kge_log
    """
    r, beta, gamma, kge = _kge_components(q_obs, q_sim)

    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    r_log, beta_log, gamma_log, kge_log = _kge_components(log_obs, log_sim)

    loss = alpha * (1.0 - kge) + (1.0 - alpha) * (1.0 - kge_log)

    info = {
        "r": r, "beta": beta, "gamma": gamma, "kge": kge,
        "r_log": r_log, "beta_log": beta_log, "gamma_log": gamma_log, "kge_log": kge_log,
    }
    return loss, info


def differentiable_pbias_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """|PBIAS| / 100, in [0, ~1] range for loss weighting."""
    pbias = (q_sim - q_obs).sum() / (q_obs.sum() + 1e-8)
    return pbias.abs()


def differentiable_nested_loss(q_obs: Tensor, q_sim: Tensor, pairs: Tensor, areas: Tensor, min_days: int = 30) -> Tensor:
    """Atténuation entre stations emboîtées, simulée contre observée (2026-11-02, R467).

    Pour chaque paire (station amont, station aval) : rapport de l'écart-type des hausses
    journalières de débit spécifique, aval sur amont. Il vaut près de 1 quand le réseau transmet
    la crue intacte, moins quand il l'amortit. Mesuré sur 58 paires : il suit l'atténuation des
    vingt plus fortes crues (rang 0,81 observé, 0,85 simulé). Le terme est l'écart absolu des
    logarithmes du rapport simulé et du rapport observé, moyen sur les paires ; il contraint le
    stockage du réseau là où l'observation existe, avec une seule règle pour tout le domaine.
    `pairs` : (P, 2) colonnes de station (amont, aval) ; `areas` : (P, 2) aires drainées en km².
    """
    if pairs is None or pairs.numel() == 0 or q_obs.shape[0] < 3:
        return torch.zeros((), device=q_sim.device)
    up, down = pairs[:, 0], pairs[:, 1]
    ok = torch.isfinite(q_obs[:, up]) & torch.isfinite(q_obs[:, down])
    ok = ok[1:] & ok[:-1]
    keep = ok.sum(dim=0) >= min_days
    if not bool(keep.any()):
        return torch.zeros((), device=q_sim.device)
    m = ok.to(q_sim.dtype)
    cnt = m.sum(dim=0).clamp(min=1.0)

    def spread(x):
        r = torch.relu(x[1:] - x[:-1]) * m
        mean = r.sum(0) / cnt
        return torch.sqrt((((r - mean) ** 2) * m).sum(0) / cnt + 1e-12)

    def ratio(q):
        return spread(q[:, down] / areas[:, 1]) / spread(q[:, up] / areas[:, 0]).clamp(min=1e-12)

    r_obs = ratio(torch.nan_to_num(q_obs))
    r_sim = ratio(q_sim)
    gap = torch.abs(torch.log(r_sim.clamp(min=1e-6)) - torch.log(r_obs.clamp(min=1e-6)))
    return gap[keep].mean()


def differentiable_dq_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Rapport des ecarts-types des VARIATIONS journalieres, simule sur observe.

    POURQUOI (Essi, 2026-09-05, apres l'audit des plateaux). Aucun terme de la perte ne
    voyait un hydrogramme fige : gamma compare des ecarts-types de NIVEAU, qu'un plateau
    a la bonne moyenne et un signal nerveux peuvent egaler. Le rapport des ecarts-types
    des differences d'un jour a l'autre, lui, vaut 0.1 pour un plateau et 3 pour un
    signal nerveux : il punit les deux fautes SYMETRIQUEMENT, ce qui est exactement ce
    qui manquait entre les deux mondes mesures le 2026-09-05 (le monde plat gagnait le
    KGE). Formule : (sigma(dQ_sim)/sigma(dQ_obs) - 1)^2, parfait = 0.
    """
    if q_obs.numel() < 3:
        return torch.zeros((), device=q_sim.device)
    d_o = q_obs[1:] - q_obs[:-1]
    d_s = q_sim[1:] - q_sim[:-1]
    s_o = torch.sqrt((d_o ** 2).mean() + 1e-12)
    s_s = torch.sqrt((d_s ** 2).mean() + 1e-12)
    # ECART ABSOLU (2026-09-21). Un rapport de statistiques penalise au CARRE repond au
    # second ordre : l'audit de tous les termes ce jour-la le mesure a 2,00. Quatrieme terme
    # de la meme famille, apres les facteurs du KGE, le soutien d'etiage et les pics par
    # rapport. La symetrie entre plateau et signal trop nerveux, qui est la raison d'etre du
    # terme, est conservee.
    return torch.abs(s_s / s_o.clamp(min=1e-8) - 1.0)


def differentiable_dq_log_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Le meme rapport de variations, mais en espace LOGARITHMIQUE.

    DEFAUT DU TERME PRECEDENT, vu sur les hydrogrammes du rapport le 2026-09-05 : l'ecart-
    type des variations ABSOLUES est domine par les crues. Un etiage fige pendant deux
    mois n'y pese presque rien, et c'est pourtant la que les plateaux se voient. En
    passant au logarithme, une variation de 5 % compte autant a 2 m3/s qu'a 200 m3/s :
    le terme voit alors la platitude d'etiage, qui est celle qui reste.
    """
    if q_obs.numel() < 3:
        return torch.zeros((), device=q_sim.device)
    eps = (1e-2 * q_obs.median().abs()).clamp(min=1e-6)
    lo = torch.log(q_obs.clamp(min=0.0) + eps)
    ls = torch.log(q_sim.clamp(min=0.0) + eps)
    d_o, d_s = lo[1:] - lo[:-1], ls[1:] - ls[:-1]
    s_o = torch.sqrt((d_o ** 2).mean() + 1e-12)
    s_s = torch.sqrt((d_s ** 2).mean() + 1e-12)
    # Ecart absolu, meme raison que pour la version lineaire ci-dessus.
    return torch.abs(s_s / s_o.clamp(min=1e-8) - 1.0)


def _month_of_day(doy: Tensor) -> Tensor:
    """Mois (1 a 12) d'un jour de l'annee, calendrier non bissextile ; un jour d'ecart au plus."""
    fins = torch.tensor([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366], device=doy.device)
    return torch.bucketize(doy.long(), fins, right=False) + 1


def differentiable_flat_days_loss(q_obs: Tensor, q_sim: Tensor, seuil: float = 0.01,
                                  width: float = 0.003) -> Tensor:
    """Écart sur la PART DE JOURS PLATS, variation relative d'un jour à l'autre sous `seuil`.

    POURQUOI (2026-10-16, R382). L'hiver du modèle unique est plat (17 % de jours plats contre
    7 % observés) et l'entraînement reconstruit cette platitude même quand la physique la
    retire, parce qu'aucun terme ne la voit : les variations journalières comparent des
    écarts-types dominés par les crues, et une baisse régulière de 1 % par jour y pèse peu.
    Ce terme compare la grandeur même du diagnostic. La part est rendue dérivable par une
    sigmoïde de largeur `largeur` autour du seuil ; la variation est prise sur le logarithme
    du débit, donc relative. Écart absolu, premier ordre ; parfait = 0. Il punit aussi
    l'excès de nervosité (part simulée sous l'observée).
    """
    if q_obs.numel() < 3:
        return torch.zeros((), device=q_sim.device, dtype=q_sim.dtype)
    eps = (1e-3 * q_obs.median().abs()).clamp(min=1e-8)

    def _part(q):
        lq = torch.log(q.clamp(min=0.0) + eps)
        d = torch.sqrt((lq[1:] - lq[:-1]) ** 2 + 1e-8)
        return torch.sigmoid((seuil - d) / width).mean()

    return torch.abs(_part(q_sim) - _part(q_obs))


def differentiable_fdc_low_loss(q_obs: Tensor, q_sim: Tensor,
                                bas: float | None = None, ref: float | None = None) -> Tensor:
    """Ecart sur le SOUTIEN d'etiage : quantile bas rapporte a la mediane.

    Substitut differentiable et bon marche de l'indice d'ecoulement de base. Un modele
    dont toute l'eau d'ete passe par la nappe a un rapport Q20/Q50 proche de 1 ; un
    modele qui la fait passer par les chemins rapides l'a bien plus bas. C'est la seule
    facon de contraindre le CHEMIN de l'eau a partir des seuls debits observes : le
    2026-09-05, deux entrainements de la meme recette ont fait passer la part de la nappe
    de 32 a 70 % du debit d'ete sans qu'aucun terme de la perte ne s'en apercoive.
    """
    if q_obs.numel() < 10:
        return torch.zeros((), device=q_sim.device)
    # Quantiles absents : réglage loss_terms.fdc_low_quantiles (0,2 et 0,5).
    _qb, _qr = settings.get("loss_terms.fdc_low_quantiles")
    bas = _qb if bas is None else bas
    ref = _qr if ref is None else ref
    qb_o = torch.quantile(q_obs, bas); qr_o = torch.quantile(q_obs, ref)
    qb_s = torch.quantile(q_sim, bas); qr_s = torch.quantile(q_sim, ref)
    r_o = qb_o / qr_o.clamp(min=1e-8)
    r_s = qb_s / qr_s.clamp(min=1e-8)
    # ECART ABSOLU ET NON CARRE (2026-09-21). Un carre repond au SECOND ordre : pour un
    # prelevement estival de cinq pour cent du debit moyen, la forme au carre bouge de
    # 0,00022 et la forme absolue de 0,0124, soit cinquante-six fois plus. Comme ce terme est
    # celui qui voit le mieux un prelevement, et que c'est la grandeur que le projet doit
    # predire, le carre le rendait presque aveugle a sa raison d'etre.
    return torch.abs(r_s - r_o)


def differentiable_low_flow_loss(q_obs: Tensor, q_sim: Tensor, quantile: float | None = None,
                               eps: float | None = None) -> Tensor:
    """Écart absolu des logarithmes du débit sur les jours d'étiage observés.

    Les jours d'étiage sont ceux où le débit observé est sous son quantile `quantile` de la
    séquence vue. Le terme vise le NIVEAU des bas débits, ce que ni le KGE, aveugle aux
    minima, ni le soutien d'étiage, rapport de deux quantiles, ne contraignent. Écart
    absolu et non carré, pour répondre au premier ordre (règle du dépôt, 2026-09-21) ; le
    logarithme rend l'erreur relative, si bien qu'un étiage deux fois trop fort coûte
    autant en été sec qu'en été humide (2026-09-30).
    """
    ok = torch.isfinite(q_obs) & torch.isfinite(q_sim)
    if int(ok.sum()) < 10:
        return torch.zeros((), device=q_sim.device)
    # Absents : réglages loss_terms.low_flow_quantile (0,3) et loss_terms.low_flow_eps_m3s (1e-3).
    quantile = float(settings.get("loss_terms.low_flow_quantile")) if quantile is None else quantile
    eps = float(settings.get("loss_terms.low_flow_eps_m3s")) if eps is None else eps
    qo, qs = q_obs[ok], q_sim[ok]
    seuil = torch.quantile(qo, quantile)
    bas = qo <= seuil
    if int(bas.sum()) < 3:
        return torch.zeros((), device=q_sim.device)
    return torch.abs(torch.log(qs[bas].clamp(min=0.0) + eps) - torch.log(qo[bas] + eps)).mean()


def quantile_loss(y_obs: Tensor, q_pred: Tensor, taus: Tensor) -> Tensor:
    """Loss quantile multi-τ = moyenne des pinball sur K quantiles.

    Parameters
    ----------
    y_obs : Tensor, shape (...,) — observations en m³/s
    q_pred : Tensor, shape (..., K) — quantiles prédits (même unités que y)
    taus : Tensor, shape (K,) — niveaux de quantiles dans (0, 1)
    """
    if y_obs.numel() == 0:
        return torch.zeros((), device=q_pred.device)
    resid = y_obs.unsqueeze(-1) - q_pred                              # (..., K)
    taus_b = taus.to(q_pred.device).expand_as(resid)
    pinball = torch.maximum(taus_b * resid, (taus_b - 1.0) * resid)
    return pinball.mean()


def crps_from_quantiles(y_obs: Tensor, q_pred: Tensor, taus: Tensor) -> Tensor:
    """CRPS approximé depuis K quantiles (Gneiting & Ranjan 2011) :
    CRPS ≈ 2 · moyenne_τ(pinball_τ). Exact si τ uniforme dans (0,1) et K→∞."""
    return 2.0 * quantile_loss(y_obs, q_pred, taus)


def tws_anomaly_loss(
    storage_month: Tensor, grace_month: Tensor,
    sim_baseline: Tensor | float, grace_baseline: Tensor | float,
    sigma: Tensor | None = None,
) -> Tensor:
    """MSE (pondérée par l'incertitude) entre l'anomalie de stockage simulée et
    GRACE TWS. Centrage : on retire une baseline de chaque côté (GRACE et sim ont
    des références absolues différentes) → on compare les VARIATIONS.

        a_sim = storage_month − sim_baseline   (mm)
        a_obs = grace_month  − grace_baseline  (mm, anomalie GRACE)
        L = mean( (a_sim − a_obs)² [ / σ² ] )

    ``sim_baseline`` = moyenne long-terme du stockage simulé (mise à jour en
    running par époque, détachée). ``grace_baseline`` = moyenne GRACE sur les
    mois communs (constante précalculée).
    """
    if storage_month.numel() == 0:
        return torch.zeros((), device=storage_month.device)
    a_sim = storage_month - sim_baseline
    a_obs = grace_month - grace_baseline
    resid = a_sim - a_obs
    if sigma is not None:
        return (resid.pow(2) / (sigma ** 2 + 1.0)).mean()  # sigma: float ou tenseur
    return resid.pow(2).mean()



def peak_weighted_mse_loss(
    q_obs: Tensor, q_sim: Tensor, q_threshold: Tensor,
    station_var: Tensor | None = None,
) -> Tensor:
    """MSE restreinte aux pas de temps où Q_obs > seuil climato par station,
    normalisée par la variance Q par station pour échelle comparable au (1-NSE).

    Cible la pathologie « peak-shaving » diagnostiquée dans le PIT val 2019-2021
    (surplus u > 0.75) : donne du gradient explicite aux résidus en régime de
    pic, où le backbone calibré sur MSE+log_NSE rabote systématiquement.

    Parameters
    ----------
    q_obs : (T, S) débits observés
    q_sim : (T, S) débits simulés
    q_threshold : (S,) seuil par station (typiquement Q_p75 de la période d'entraînement)
    station_var : (S,) variance Q par station. Si fourni, normalise SE/var_s pour
                  que la magnitude finale soit O(0.1-1), comparable à (1-NSE).
                  Sans normalisation, la loss est en m³/s² brut et domine.

    Returns
    -------
    L_peak : MSE normalisée pooled sur les couples (t, s) tels que Q_obs[t,s] > q_threshold[s].
             Renvoie 0 (sans gradient) si aucun pic dans le chunk.

    Notes
    -----
    Chunk-safe : un chunk sans pic contribue 0, les chunks avec pics dominent.
    La pondération est binaire (seuil dur) — alternative continue : Q_obs^p.
    """
    if q_obs.numel() == 0:
        return torch.zeros((), device=q_sim.device)
    # Broadcast seuil (S,) sur (T, S)
    mask_peak = q_obs > q_threshold.unsqueeze(0)
    valid = mask_peak & ~torch.isnan(q_obs) & ~torch.isnan(q_sim)
    if not valid.any():
        return torch.zeros((), device=q_sim.device)
    # IMPORTANT : filtrer AVANT de calculer (q_obs - q_sim) ** 2. Sinon les
    # positions NaN produisent NaN dans le forward, et même si on les masque
    # avec [valid].mean(), le backward propage NaN × 0 = NaN à q_sim → tous
    # les grads zéroés en aval.
    qo = q_obs[valid]
    qs = q_sim[valid]
    sq_err = (qo - qs) ** 2
    if station_var is not None:
        # Index station pour chaque (t,s) valide
        S = q_obs.shape[1]
        s_idx = torch.arange(S, device=q_obs.device).unsqueeze(0).expand_as(q_obs)
        var_at_valid = station_var[s_idx[valid]] + 1e-8
        sq_err = sq_err / var_at_valid
    return sq_err.mean()


def differentiable_mse_loss(
    q_obs: Tensor, q_sim: Tensor, var: Tensor | None = None
) -> Tensor:
    """Mean squared error, optionally normalized by precomputed variance.

    When ``var`` is provided, computes MSE/var — equivalent to 1-NSE but with
    a fixed denominator, so it remains additive across temporal chunks.
    """
    mse = ((q_obs - q_sim) ** 2).mean()
    if var is not None:
        return mse / (var + 1e-8)
    return mse


def differentiable_nrmse_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Normalized RMSE: RMSE / mean(obs). Dimensionless, lower is better."""
    mse = ((q_obs - q_sim) ** 2).mean()
    return torch.sqrt(mse + 1e-8) / (q_obs.mean() + 1e-8)


def differentiable_log_mse_loss(
    q_obs: Tensor, q_sim: Tensor, eps: float | None = None
) -> Tensor:
    """MSE on log-transformed flows. Scale-invariant AND chunk-safe.

    Unlike log-NSE, this has no observation-variance denominator, so it remains
    additive across temporal chunks. The log transform naturally equalizes the
    gradient contribution of low-flow and high-flow periods. Décalage absent : réglage
    loss_terms.log_offset_m3s (1 m³/s).
    """
    if eps is None:
        eps = float(settings.get("loss_terms.log_offset_m3s"))
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    return ((log_obs - log_sim) ** 2).mean()


def differentiable_log_nse_loss(
    q_obs: Tensor, q_sim: Tensor, eps: float = 0.01
) -> Tensor:
    """1 - NSE on log-transformed flows (emphasises low-flow periods).

    Using log(Q + eps) puts more weight on getting baseflow right,
    which improves KGE's variability ratio component.
    """
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    num = ((log_obs - log_sim) ** 2).sum()
    denom = ((log_obs - log_obs.mean()) ** 2).sum()
    return num / (denom + 1e-8)


class CompositeKGELoss(nn.Module):
    """Composite KGE loss: alpha*(1-KGE) + (1-alpha)*(1-KGE_log).

    Per-station computation with optional station weights.
    Returns KGE components (r, beta, gamma) for diagnostic logging.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        eps: float = 1.0,
        per_station: bool = True,
        station_weights: Tensor | None = None,
        w_physics: float = 0.01,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.per_station = per_station
        self.w_physics = w_physics
        if station_weights is not None:
            self.register_buffer("station_weights", station_weights)
        else:
            self.station_weights: Tensor | None = None

    def forward(
        self,
        q_obs: Tensor,
        q_sim: Tensor,
        station_mask: Tensor,
        swe_obs: Tensor | None = None,
        swe_sim: Tensor | None = None,
        et_obs: Tensor | None = None,
        et_sim: Tensor | None = None,
        water_balance_residual: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        q_sim_at_stations = q_sim[:, station_mask]  # (T, n_stations)
        dev = q_sim.device
        zero = torch.tensor(0.0, device=dev)

        if self.per_station:
            n_stations = q_sim_at_stations.shape[1]
            valid = ~torch.isnan(q_obs) & ~torch.isnan(q_sim_at_stations)
            valid_counts = valid.sum(dim=0)
            keep = valid_counts >= 30

            n_keep = keep.sum().item()
            if n_keep == 0:
                loss = zero
                components = {
                    "r": zero, "beta": zero, "gamma": zero, "kge": zero,
                    "r_log": zero, "beta_log": zero, "gamma_log": zero, "kge_log": zero,
                }
            else:
                if self.station_weights is not None and len(self.station_weights) == n_stations:
                    w = self.station_weights[keep]
                    w = w / w.sum()
                else:
                    w = torch.full((n_keep,), 1.0 / n_keep, device=dev)

                keep_idx = keep.nonzero(as_tuple=True)[0]
                losses, r_vals, beta_vals, gamma_vals, kge_vals = [], [], [], [], []
                r_log_vals, beta_log_vals, gamma_log_vals, kge_log_vals = [], [], [], []

                for si in keep_idx:
                    v = valid[:, si]
                    q_o_v = q_obs[v, si]
                    q_s_v = q_sim_at_stations[v, si]
                    l, info = differentiable_composite_kge_loss(
                        q_o_v, q_s_v, alpha=self.alpha, eps=self.eps,
                    )
                    losses.append(l)
                    r_vals.append(info["r"])
                    beta_vals.append(info["beta"])
                    gamma_vals.append(info["gamma"])
                    kge_vals.append(info["kge"])
                    r_log_vals.append(info["r_log"])
                    beta_log_vals.append(info["beta_log"])
                    gamma_log_vals.append(info["gamma_log"])
                    kge_log_vals.append(info["kge_log"])

                loss = (torch.stack(losses) * w).sum()
                components = {
                    "r": (torch.stack(r_vals) * w).sum(),
                    "beta": (torch.stack(beta_vals) * w).sum(),
                    "gamma": (torch.stack(gamma_vals) * w).sum(),
                    "kge": (torch.stack(kge_vals) * w).sum(),
                    "r_log": (torch.stack(r_log_vals) * w).sum(),
                    "beta_log": (torch.stack(beta_log_vals) * w).sum(),
                    "gamma_log": (torch.stack(gamma_log_vals) * w).sum(),
                    "kge_log": (torch.stack(kge_log_vals) * w).sum(),
                }
        else:
            q_o = q_obs.reshape(-1)
            q_s = q_sim_at_stations.reshape(-1)
            valid_mask = ~torch.isnan(q_o) & ~torch.isnan(q_s)
            q_o, q_s = q_o[valid_mask], q_s[valid_mask]
            loss, components = differentiable_composite_kge_loss(
                q_o, q_s, alpha=self.alpha, eps=self.eps,
            )

        # Regularization terms
        if self.w_physics > 0 and water_balance_residual is not None:
            valid_wb = ~torch.isnan(water_balance_residual)
            if valid_wb.any():
                L_phys = (water_balance_residual[valid_wb] ** 2).mean()
                loss = loss + self.w_physics * L_phys
                components["physics_loss"] = L_phys

        components["total"] = loss
        return loss, components


def rolling_mean(x, window: int):
    """Moyenne glissante causale sur l'axe TEMPS (axe 0), fenêtre en pas de temps.

    Sert à apparier l'ETR simulée AUX COMPOSITES MOD16A2GF, qui sont des moyennes sur
    8 jours posées sur le seul jour de début de composite (bug d'appariement corrigé le
    2026-08-10 : on comparait une moyenne de 8 jours à une valeur journalière). Les
    `fenetre - 1` premiers pas sont laissés tels quels faute d'historique.
    """
    if window <= 1 or x.dim() < 1 or x.shape[0] < window:
        return x
    c = torch.nn.functional.avg_pool1d(
        x.transpose(0, -1).unsqueeze(0), kernel_size=window, stride=1,
        padding=0, count_include_pad=False).squeeze(0).transpose(0, -1)
    return torch.cat([x[:window - 1], c], dim=0)[:x.shape[0]]


class HydroLoss(nn.Module):
    """Multi-objective loss function.

    Parameters
    ----------
    w_nse, w_pbias, w_kge, w_mse, w_nrmse, w_log_nse : float
        Weights for the streamflow skill scores.
        w_mse is recommended for chunked training (additive across chunks).
    w_snow : float
        Weight for SWE reconstruction loss (0 if no SWE observations).
    w_et : float
        Weight for ET loss (0 if no flux-tower data).
    w_physics : float
        Weight for water balance closure penalty.
    per_station : bool
        If True, compute NSE/KGE/PBIAS per station and average (equal weight
        per station).  If False (default), pool all stations into one vector
        (dominated by the largest-flow station).
    """

    def __init__(
        self,
        w_nse: float = 1.0,
        w_pbias: float = 0.1,
        w_kge: float = 0.5,
        w_mse: float = 0.0,
        w_nrmse: float = 0.0,
        w_log_nse: float = 0.0,
        w_log_mse: float = 0.0,
        w_snow: float = 0.0,
        w_swe_mass: float = 0.0,
        w_et: float = 0.0,
        et_mode: str = "level",
        w_tws: float = 0.0,
        w_tws_clim: float = 0.0,
        w_quantile: float = 0.0,
        w_peak: float = 0.0,
        w_dq: float = 0.0,
        w_dq_log: float = 0.0,
        w_fdc_low: float = 0.0,
        w_low_flow: float = 0.0,
        w_r: float = 0.0,
        w_beta: float = 0.0,
        w_gamma: float = 0.0,
        w_peak_ratio: float = 0.0,
        w_recession: float = 0.0,
        w_flat: float = 0.0,
        w_nested: float = 0.0,
        flat_threshold: float = 0.01,
        flat_width: float = 0.003,
        flat_months: tuple = (12, 1, 2, 3),
        w_physics: float = 0.01,
        per_station: bool = False,
        station_weights: Tensor | None = None,
        station_var: Tensor | None = None,
        peak_threshold: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w_nse = w_nse
        self.w_pbias = w_pbias
        self.w_kge = w_kge
        self.w_mse = w_mse
        self.w_nrmse = w_nrmse
        self.w_log_nse = w_log_nse
        self.w_log_mse = w_log_mse
        self.w_snow = w_snow
        # CanSWE : masse du manteau mesurée au sol (R24). Calculé dans le
        # trainer, qui seul dispose du SWE simulé par nœud et de la table
        # site -> nœud. Complète w_snow (couverture MODIS), ne le remplace pas.
        self.w_swe_mass = w_swe_mass
        self.w_et = w_et
        # et_mode : "level" (MSE absolu, historique) ou "anomaly" (TENDANCE : les deux
        # séries sont centrées par nœud sur une ligne de base longue durée AVANT
        # d'arriver ici — centrage fait dans le trainer, qui détient l'état, comme
        # pour la TWS corrigée le 2026-08-10). Motivation : MOD16 sur-estime le NIVEAU
        # (~593 vs 450 mm/an aux tours de flux) ; en mode anomaly on ne contraint que
        # la saisonnalité, la phase et l'interannuel, pas le niveau biaisé.
        # "bassin" (2026-09-22) : la contrainte porte sur la MOYENNE du territoire, pas sur
        # chaque tronçon. Motif mesuré : sur 3150 nœuds au pas de huit jours, MOD16 porte UNE
        # direction indépendante et un rapport de participation de 1,1, c'est-à-dire une seule
        # courbe saisonnière. Prétendre en faire une contrainte spatiale est faux, et coûte
        # 3150 séries en mémoire sur une carte qui débordait déjà.
        # "forme" (2026-10-06) : centré PUIS divisé par l'écart-type de chaque côté, par
        # nœud, comme GRACE en forme. Le mode centré n'est pas aveugle au niveau : K_c
        # multiplie l'ET, donc son amplitude saisonnière, et le terme baissait K_c pour
        # réduire l'amplitude simulée, entraînant le niveau avec elle (R310). En forme, MOD16
        # ne contraint que la phase et la forme ; le niveau revient aux débits.
        if et_mode not in ("level", "anomaly", "bassin", "forme"):
            raise ValueError(f"et_mode inconnu : {et_mode!r} "
                             "(attendu 'level', 'anomaly', 'bassin' ou 'forme')")
        self.et_mode = et_mode
        self.w_tws = w_tws  # GRACE TWS (calculé dans le trainer, lu via loss_fn.w_tws)
        # Biais saisonnier GRACE par mois calendaire (R23). Le terme mensuel
        # ci-dessus juge à l'incertitude d'UNE observation (25 mm) et laisse
        # passer un biais systématique de 26 mm ; celui-ci le juge à
        # l'incertitude de la CLIMATOLOGIE (~5 mm). Calculé dans le trainer.
        self.w_tws_clim = w_tws_clim
        self.w_quantile = w_quantile  # Régression quantile (calculée dans le trainer)
        self.w_peak = w_peak  # Pondération pics (seuil climato Q_p75 par station)
        # FORME (2026-09-05) : variations journalieres et soutien d'etiage.
        self.w_dq = w_dq
        # Atténuation entre stations emboîtées (R467) : paires posées par le pilote.
        self.w_nested = w_nested
        self.nested_pairs = None
        self.nested_areas = None
        self.w_dq_log = w_dq_log
        self.w_fdc_low = w_fdc_low
        self.w_low_flow = w_low_flow
        # KGE DECOMPOSE (2026-09-20, proposition d'Essi). Porte entier, le KGE melange une
        # erreur de calendrier, une erreur de volume et une erreur d'amplitude en un seul
        # nombre, et l'optimiseur ne peut plus choisir laquelle reduire. Mesure du banc de
        # redondance sur 32 stations et 9 deformations : la recette a cinq termes laisse
        # 14,2 % de manque moyen et un angle mort de 48,2 % sur le soutien d'etiage ; les
        # trois facteurs separes plus l'etiage plus les variations, cinq termes aussi,
        # tombent a 2,8 % et 5,4 %. Poids nuls par defaut : rien ne change sans declaration.
        self.w_r = w_r
        self.w_beta = w_beta
        self.w_gamma = w_gamma
        # Terme de pics par RAPPORT des magnitudes, immunise contre l'aplatissement.
        self.w_peak_ratio = w_peak_ratio
        # Vitesse de vidange sur les jours de decrue : seule grandeur qui identifie la
        # partition entre ecoulement hypodermique et nappe.
        self.w_recession = w_recession
        self.w_flat = w_flat
        self.flat_threshold = flat_threshold
        self.flat_width = flat_width
        self.flat_months = tuple(int(m) for m in flat_months)
        # Jour de l'annee de chaque pas du bloc courant, pose par le trainer avant l'appel :
        # le terme de platitude ne regarde que les mois `plat_mois` du bloc (R382).
        self.doy_block = None
        self.w_physics = w_physics
        self.per_station = per_station
        if station_weights is not None:
            self.register_buffer("station_weights", station_weights)
        else:
            self.station_weights: Tensor | None = None
        if station_var is not None:
            self.register_buffer("station_var", station_var)
        else:
            self.station_var: Tensor | None = None
        if peak_threshold is not None:
            self.register_buffer("peak_threshold", peak_threshold)
        else:
            self.peak_threshold: Tensor | None = None

    def forward(
        self,
        q_obs: Tensor,
        q_sim: Tensor,
        station_mask: Tensor,
        swe_obs: Tensor | None = None,
        swe_sim: Tensor | None = None,
        et_obs: Tensor | None = None,
        et_sim: Tensor | None = None,
        water_balance_residual: Tensor | None = None,
        q_obs_hist: Tensor | None = None,
        q_sim_hist: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            q_obs:  (n_timesteps, n_stations) observed streamflow
            q_sim:  (n_timesteps, n_nodes) simulated; masked via station_mask
            station_mask: (n_stations,) bool — which nodes have observations
            ... optional auxiliary observations

        Returns:
            loss:       scalar total loss
            components: dict of named loss terms for logging
        """
        q_sim_at_stations = q_sim[:, station_mask]  # (T, n_stations)
        # Les trois facteurs du KGE sont initialises AVANT le branchement par station ou en
        # commun : la somme finale les lit dans les deux cas, alors que seul le chemin par
        # station les remplissait, d'ou une variable non liee sur l'autre.
        L_r = L_beta = L_gamma = L_peak_ratio = torch.tensor(0.0, device=q_sim.device)
        L_recession = torch.tensor(0.0, device=q_sim.device)
        L_flat = torch.tensor(0.0, device=q_sim.device)

        if self.per_station:
            n_stations = q_sim_at_stations.shape[1]
            dev = q_sim.device
            zero = torch.tensor(0.0, device=dev)

            # Validity mask: (T, S) — True where both obs and sim are valid
            valid = ~torch.isnan(q_obs) & ~torch.isnan(q_sim_at_stations)
            valid_counts = valid.sum(dim=0)  # (S,)
            # MINIMUM DE TRENTE OBSERVATIONS (revu le 2026-09-04). Le seuil protegeait les
            # statistiques de sequence (KGE, NSE) contre un bloc trop court. Avec
            # l'historique detache, les statistiques portent sur l'historique PLUS le
            # bloc : c'est cette longueur-la qu'il faut compter. Sans cela, un bloc de
            # quinze jours (douze vivants apres rodage) rendait une perte de debit NULLE,
            # et le trainer, qui ajoute le prior a chaque bloc, faisait alors un pas sur
            # le prior seul : 23 pas sur 24 par epoque, mesure sur le banc de sous-bassin.
            _n_hist = torch.zeros_like(valid_counts)
            if q_obs_hist is not None and q_sim_hist is not None and q_obs_hist.numel() > 0:
                _n_hist = (~torch.isnan(q_obs_hist) & ~torch.isnan(q_sim_hist)).sum(dim=0)
            _min_obs = int(settings.get("loss_terms.min_station_obs"))
            keep = ((valid_counts + _n_hist) >= _min_obs) & (valid_counts >= 1)
            n_keep = keep.sum().item()

            if n_keep == 0:
                logger.warning(
                    f"No stations have >= {_min_obs} valid observations in this chunk "
                    "(total stations: %d, max valid count: %s). "
                    "Loss will be zero with no gradient.",
                    n_stations,
                    int(valid_counts.max().item()) if n_stations > 0 else "N/A",
                )
                L_nse = L_pbias = L_kge = L_mse = L_nrmse = L_log_nse = L_log_mse = zero
                # Les termes de forme aussi, sans quoi un bloc sans aucune station gardee
                # leve une erreur au lieu de rendre une perte nulle (2026-09-09).
                L_dq = L_dql = L_fdc = zero
                L_low_flow = zero
            else:
                # MASQUE SANS NaN (2026-09-08). L'ancienne version posait des NaN DANS
                # le tenseur simule, puis s'appuyait sur nanmean et nansum. Ces reductions
                # ignorent les NaN a l'ALLER mais leur retropropagation rend un gradient
                # NaN : la perte restait finie, les debits aussi, et le gradient etait
                # empoisonne des qu'un bloc contenait une observation manquante. Mesure sur
                # la flotte du 6 septembre : 2000 blocs jetes sur le Saint-Laurent sud,
                # 1760 en Gaspesie, soit pres de la moitie des pas d'optimisation, alors
                # que les regions aux series completes n'en jetaient aucun. On masque donc
                # par multiplication et on normalise par le compte, ce qui donne la meme
                # valeur avec un gradient fini.
                q_o = torch.nan_to_num(q_obs[:, keep], nan=0.0)  # (T, S_keep)
                q_s = q_sim_at_stations[:, keep]                 # (T, S_keep), jamais NaN
                _m = valid[:, keep].to(q_s.dtype)                # 1 si observe, 0 sinon
                _n_val = _m.sum(dim=0).clamp(min=1.0)            # (S_keep,)

                # Weights for kept stations
                if self.station_weights is not None and len(self.station_weights) == n_stations:
                    w = self.station_weights[keep]
                    w = w / w.sum()
                else:
                    w = torch.full((n_keep,), 1.0 / n_keep, device=dev)

                # ── Vectorized MSE (chunk-safe) ──────────────────────────
                if self.w_mse > 0:
                    sq_err = ((q_o - q_s) * _m) ** 2            # (T, S_keep)
                    mse_per = sq_err.sum(dim=0) / _n_val        # (S_keep,)
                    if self.station_var is not None:
                        mse_per = mse_per / (self.station_var[keep] + 1e-8)
                    L_mse = (mse_per * w).sum()
                else:
                    L_mse = zero

                # ── Vectorized PBIAS ─────────────────────────────────────
                if self.w_pbias > 0:
                    diff_sum = ((q_s - q_o) * _m).sum(dim=0)    # (S_keep,)
                    obs_sum = (q_o * _m).sum(dim=0)              # (S_keep,)
                    pbias_per = (diff_sum / (obs_sum + 1e-8)).abs()
                    # VOLUME SUR LA SERIE CONTINUE (2026-10-09). Calcule sur le seul bloc de 45
                    # jours, le terme exigeait le volume observe dans CHAQUE fenetre et
                    # combattait tout transfert d'eau entre saisons : 0,44 de la perte
                    # d'entrainement de la Monteregie contre 0,06 sur la serie entiere qui sert
                    # de juge. Comme le KGE : historique detache plus bloc courant, gradient
                    # remis a l'echelle du jour vivant.
                    if (q_obs_hist is not None and q_sim_hist is not None and q_obs_hist.numel() > 0):
                        _ho = q_obs_hist[:, keep]
                        _hs = q_sim_hist[:, keep]
                        _hm = (~torch.isnan(_ho) & ~torch.isnan(_hs)).to(q_s.dtype)
                        _hd = ((torch.nan_to_num(_hs) - torch.nan_to_num(_ho)) * _hm).sum(dim=0).detach()
                        _hob = (torch.nan_to_num(_ho) * _hm).sum(dim=0).detach()
                        _ech = (_hm.sum(dim=0) + _n_val) / _n_val
                        _pb = ((diff_sum + _hd) / (obs_sum + _hob + 1e-8)).abs()
                        pbias_per = _pb.detach() + (_pb - _pb.detach()) * _ech
                    L_pbias = (pbias_per * w).sum()
                else:
                    L_pbias = zero

                # ── Vectorized log-MSE ───────────────────────────────────
                if self.w_log_mse > 0:
                    _lo = float(settings.get("loss_terms.log_offset_m3s"))
                    log_sq = ((torch.log(q_o + _lo)
                               - torch.log(q_s.clamp(min=0.0) + _lo)) * _m) ** 2
                    L_log_mse = ((log_sq.sum(dim=0) / _n_val) * w).sum()
                else:
                    L_log_mse = zero

                # ── Per-station loop only for metrics that need it ───────
                # NSE, KGE, NRMSE, log-NSE require per-station variance
                # or correlation — only compute if weight > 0
                L_nse = L_kge = L_nrmse = L_log_nse = zero
                L_dq = L_fdc = L_dql = zero
                L_low_flow = zero
                need_loop = (self.w_nse > 0 or self.w_kge > 0 or self.w_low_flow > 0
                             or self.w_nrmse > 0 or self.w_log_nse > 0
                             or self.w_dq > 0 or self.w_fdc_low > 0 or self.w_dq_log > 0
                             or self.w_r > 0 or self.w_beta > 0 or self.w_gamma > 0
                             or self.w_peak_ratio > 0 or self.w_recession > 0 or self.w_flat > 0)
                if need_loop:
                    nse_v, kge_v, nrmse_v, lnse_v = [], [], [], []
                    dq_v, fdc_v, dql_v = [], [], []
                    low_flow_v = []
                    r_v, beta_v, gamma_v = [], [], []
                    pr_v, rec_v, flat_v = [], [], []
                    keep_idx = keep.nonzero(as_tuple=True)[0]
                    # HISTORIQUE DÉTACHÉ (2026-09-03). Le KGE, le Nash-Sutcliffe et le
                    # NRMSE sont des statistiques de SÉQUENCE : moyennes, écarts-types et
                    # corrélation. Calculés sur un bloc d'entraînement de 45 jours, ils ne
                    # mesurent pas la même chose que sur la série continue qui sert à
                    # choisir le modèle. Mesuré sur les huit régions du rapport : le KGE
                    # médian par fenêtre de 45 jours vaut -0,01 à 0,27 quand le KGE
                    # continu vaut 0,55 à 0,78, et sur slno la corrélation entre les deux
                    # tombe à 0,05. La perte optimisait donc un objet sans rapport avec la
                    # métrique de sélection.
                    # Correction : les statistiques portent sur toute la séquence vue
                    # depuis le début de l'époque. L'historique est DÉTACHÉ, donc le
                    # gradient ne remonte que par le bloc courant, ce qui est correct :
                    # les blocs passés ont été simulés avec des paramètres antérieurs.
                    _hist = (q_obs_hist is not None and q_sim_hist is not None
                             and q_obs_hist.numel() > 0)
                    if _hist:
                        _vh = ~torch.isnan(q_obs_hist) & ~torch.isnan(q_sim_hist)
                    for j, si in enumerate(keep_idx):
                        v = valid[:, si]
                        q_o_v = q_obs[v, si]
                        q_s_v = q_sim_at_stations[v, si]
                        if _hist:
                            _vs = _vh[:, si]
                            if bool(_vs.any()):
                                q_o_v = torch.cat([q_obs_hist[_vs, si].detach(), q_o_v])
                                q_s_v = torch.cat([q_sim_hist[_vs, si].detach(), q_s_v])
                        # ECHELLE DU GRADIENT AVEC HISTORIQUE (2026-09-05). Une statistique
                        # de sequence sur N = N_hist + n jours a un gradient en 1/N par jour
                        # vivant ; la MSE du bloc l'a en 1/n. Le trainer multiplie ensuite
                        # la perte du bloc par n/N_train : le KGE recevait donc un facteur
                        # n/N de trop, environ 1 % pour 34 jours vivants sur 2900
                        # d'historique. Gamma, seul terme de toute la perte qui punit la
                        # platitude, etait de fait inerte (audit du 2026-09-05). On remet
                        # la sensibilite par jour vivant a l'echelle des termes locaux :
                        # la VALEUR imprimee reste celle de la serie complete (detachee).
                        _ech = 1.0
                        if _hist:
                            _n_viv = max(int(v.sum()), 1)
                            _ech_t = torch.tensor(float(q_o_v.numel()) / _n_viv, device=q_o_v.device)
                            _ech = _ech_t
                        def _reset(l_):
                            return l_.detach() + (l_ - l_.detach()) * _ech
                        if self.w_nse > 0:
                            nse_v.append(_reset(differentiable_nse_loss(q_o_v, q_s_v)))
                        if self.w_kge > 0:
                            kge_v.append(_reset(differentiable_kge_loss(q_o_v, q_s_v)))
                        if self.w_r > 0:
                            r_v.append(_reset(differentiable_r_loss(q_o_v, q_s_v)))
                        if self.w_beta > 0:
                            beta_v.append(_reset(differentiable_beta_loss(q_o_v, q_s_v)))
                        if self.w_gamma > 0:
                            gamma_v.append(_reset(differentiable_gamma_loss(q_o_v, q_s_v)))
                        if self.w_recession > 0:
                            rec_v.append(_reset(differentiable_recession_loss(q_o_v, q_s_v)))
                        if self.w_flat > 0 and self.doy_block is not None:
                            # BLOC SEUL ET MOIS CIBLES (2026-10-16, R382). Sur la serie entiere,
                            # l'optimiseur satisfaisait le terme en rendant l'ete nerveux et
                            # gardait l'hiver plat. Sur le seul bloc de 90 jours et les seuls
                            # jours des mois cibles, aucune compensation entre saisons.
                            _month = _month_of_day(self.doy_block.to(v.device))
                            _k = v & torch.isin(_month, torch.tensor(self.flat_months, device=v.device))
                            if int(_k.sum()) >= int(settings.get("loss_terms.flat_min_days")):
                                flat_v.append(differentiable_flat_days_loss(
                                    q_obs[_k, si], q_sim_at_stations[_k, si], self.flat_threshold, self.flat_width))
                            else:
                                # Meme longueur que les poids par station : zero hors des mois cibles.
                                flat_v.append(q_sim_at_stations[v, si].sum() * 0.0)
                        if self.w_peak_ratio > 0 and self.peak_threshold is not None:
                            _threshold = self.peak_threshold[si] if self.peak_threshold.numel() > 1                                 else self.peak_threshold
                            pr_v.append(_reset(differentiable_peak_ratio_loss(q_o_v, q_s_v, _threshold)))
                        if self.w_dq > 0:
                            dq_v.append(_reset(differentiable_dq_loss(q_o_v, q_s_v)))
                        if self.w_dq_log > 0:
                            dql_v.append(_reset(differentiable_dq_log_loss(q_o_v, q_s_v)))
                        if self.w_low_flow > 0:
                            low_flow_v.append(_reset(differentiable_low_flow_loss(q_o_v, q_s_v)))
                        if self.w_fdc_low > 0:
                            fdc_v.append(_reset(differentiable_fdc_low_loss(q_o_v, q_s_v)))
                            if settings.get("debug.kge_trace"):
                                # Trace de controle (2026-09-04) : le terme KGE de la perte
                                # valait 0.04 quand le KGE de l'annee d'entrainement
                                # mesure hors ligne valait 0.75. On imprime ce que le
                                # terme voit vraiment : longueur d'historique, du bloc
                                # vivant, et les trois composantes.
                                _r, _b, _g, _k = _kge_components(q_o_v.detach(), q_s_v.detach())
                                _nh = int(_vs.sum()) if _hist else 0
                                print(f"[kge-debug] station {int(si)} | hist {_nh} j | vivant "
                                      f"{int(v.sum())} j | KGE {float(_k):.3f} r {float(_r):.3f} "
                                      f"beta {float(_b):.3f} gamma {float(_g):.3f} | mu_obs "
                                      f"{float(q_o_v.mean()):.2f} mu_sim {float(q_s_v.mean()):.2f}",
                                      flush=True)
                        if self.w_nrmse > 0:
                            nrmse_v.append(differentiable_nrmse_loss(q_o_v, q_s_v))
                        if self.w_log_nse > 0:
                            lnse_v.append(differentiable_log_nse_loss(q_o_v, q_s_v))
                    if self.w_nse > 0 and nse_v:
                        L_nse = (torch.stack(nse_v) * w).sum()
                    if self.w_kge > 0 and kge_v:
                        L_kge = (torch.stack(kge_v) * w).sum()
                    if self.w_r > 0 and r_v:
                        L_r = (torch.stack(r_v) * w).sum()
                    if self.w_beta > 0 and beta_v:
                        L_beta = (torch.stack(beta_v) * w).sum()
                    if self.w_gamma > 0 and gamma_v:
                        L_gamma = (torch.stack(gamma_v) * w).sum()
                    if self.w_peak_ratio > 0 and pr_v:
                        L_peak_ratio = (torch.stack(pr_v) * w).sum()
                    if self.w_recession > 0 and rec_v:
                        L_recession = (torch.stack(rec_v) * w).sum()
                    if self.w_flat > 0 and flat_v:
                        L_flat = (torch.stack(flat_v) * w).sum()
                    if self.w_nrmse > 0 and nrmse_v:
                        L_nrmse = (torch.stack(nrmse_v) * w).sum()
                    if self.w_log_nse > 0 and lnse_v:
                        L_log_nse = (torch.stack(lnse_v) * w).sum()
                    if self.w_dq > 0 and dq_v:
                        L_dq = (torch.stack(dq_v) * w).sum()
                    if self.w_dq_log > 0 and dql_v:
                        L_dql = (torch.stack(dql_v) * w).sum()
                    if self.w_fdc_low > 0 and fdc_v:
                        L_fdc = (torch.stack(fdc_v) * w).sum()
                    if self.w_low_flow > 0 and low_flow_v:
                        L_low_flow = (torch.stack(low_flow_v) * w).sum()
        else:
            # Pooled metrics: flatten time x station (dominated by largest station)
            q_o = q_obs.reshape(-1)
            q_s = q_sim_at_stations.reshape(-1)

            # Mask NaN in both obs and sim (NaN sim comes from missing forcing or
            # extreme parameter values early in training)
            valid = ~torch.isnan(q_o) & ~torch.isnan(q_s)
            q_o, q_s = q_o[valid], q_s[valid]

            L_nse = differentiable_nse_loss(q_o, q_s)
            L_pbias = differentiable_pbias_loss(q_o, q_s)
            L_kge = differentiable_kge_loss(q_o, q_s)
            L_mse = differentiable_mse_loss(q_o, q_s)
            L_nrmse = differentiable_nrmse_loss(q_o, q_s)
            _zero = torch.tensor(0.0, device=q_s.device)
            L_log_nse = differentiable_log_nse_loss(q_o, q_s) if self.w_log_nse > 0 else _zero
            L_log_mse = differentiable_log_mse_loss(q_o, q_s) if self.w_log_mse > 0 else _zero
            L_dq = differentiable_dq_loss(q_o, q_s) if self.w_dq > 0 else _zero
            L_dql = differentiable_dq_log_loss(q_o, q_s) if self.w_dq_log > 0 else _zero
            L_fdc = differentiable_fdc_low_loss(q_o, q_s) if self.w_fdc_low > 0 else _zero
            L_low_flow = differentiable_low_flow_loss(q_o, q_s) if self.w_low_flow > 0 else _zero

        # Pondération pics (seuil climato Q_p75 par station) : MSE restreinte
        # aux pas de temps où Q_obs dépasse le seuil. Vise la queue haute
        # diagnostiquée comme sous-dispersée dans le PIT val.
        if self.w_peak > 0 and self.peak_threshold is not None:
            L_peak = peak_weighted_mse_loss(
                q_obs, q_sim_at_stations, self.peak_threshold,
                station_var=self.station_var,
            )
        else:
            L_peak = torch.tensor(0.0, device=q_sim.device)

        L_nested = (differentiable_nested_loss(q_obs, q_sim_at_stations, self.nested_pairs, self.nested_areas)
                    if self.w_nested > 0 and self.nested_pairs is not None else torch.tensor(0.0, device=q_sim.device))
        loss = (self.w_nse * L_nse + self.w_pbias * L_pbias
                + self.w_kge * L_kge + self.w_mse * L_mse
                + self.w_nrmse * L_nrmse
                + self.w_log_nse * L_log_nse
                + self.w_log_mse * L_log_mse
                + self.w_peak * L_peak
                + self.w_dq * L_dq
                + self.w_dq_log * L_dql
                + self.w_fdc_low * L_fdc
                + self.w_low_flow * L_low_flow
                + self.w_r * L_r + self.w_beta * L_beta + self.w_gamma * L_gamma
                + self.w_peak_ratio * L_peak_ratio + self.w_recession * L_recession
                + self.w_flat * L_flat
                + self.w_nested * L_nested)
        components = {"nse_loss": L_nse, "pbias_loss": L_pbias,
                      "dq_loss": L_dq, "dq_log_loss": L_dql, "fdc_bas_loss": L_fdc, "etiage_loss": L_low_flow,
                      "kge_loss": L_kge, "mse_loss": L_mse,
                      "r_loss": L_r, "beta_loss": L_beta, "gamma_loss": L_gamma,
                      "peak_ratio_loss": L_peak_ratio, "recession_loss": L_recession, "plat_loss": L_flat,
                      "nrmse_loss": L_nrmse,
                      "log_nse_loss": L_log_nse,
                      "log_mse_loss": L_log_mse,
                      "peak_loss": L_peak,
                      "nested_loss": L_nested}

        if self.w_snow > 0 and swe_obs is not None and swe_sim is not None:
            # Meme piege que pour l'evapotranspiration : sous MEANDRE_DIAG_CPU=1 le simule
            # arrive sur le processeur et l'observe sur la carte.
            if swe_sim.device != swe_obs.device:
                swe_sim = swe_sim.to(swe_obs.device)
            valid = ~torch.isnan(swe_obs) & ~torch.isnan(swe_sim)
            if valid.any():
                L_snow = ((swe_obs[valid] - swe_sim[valid]) ** 2).mean()
                loss = loss + self.w_snow * L_snow
                components["snow_loss"] = L_snow

        if self.w_et > 0 and et_obs is not None and et_sim is not None:
            # APPARIEMENT TEMPOREL (corrigé le 2026-08-10). MOD16A2GF fournit une MOYENNE
            # sur 8 jours, que le chargeur pose sur le SEUL jour de début de composite ;
            # on la comparait à l'ETR simulée de CE jour précis. Confronter une moyenne
            # de 8 jours à une valeur journalière ajoute du bruit pur et fausse toute
            # statistique de variance. On moyenne donc le simulé sur la même fenêtre.
            # `et_window` = loss_terms.et_window_days (8) par défaut ; 1 restaure l'ancien comportement.
            et_sim = rolling_mean(et_sim, int(getattr(self, "et_window", settings.get("loss_terms.et_window_days"))))
            # MEANDRE_DIAG_CPU=1 accumule les diagnostics sur le processeur pour tenir dans
            # la memoire de la carte : l'evapotranspiration simulee arrive alors sur un
            # appareil different de l'observee et la comparaison levait une exception, ce
            # qui rendait tout entrainement impossible sous ce reglage (mesure 2026-09-19).
            if et_sim.device != et_obs.device:
                et_sim = et_sim.to(et_obs.device)
            if self.et_mode == "bassin":
                # Moyenne sur les nœuds avant comparaison : un nœud sans observation ne doit
                # pas tirer la moyenne, d'où le compte des valides au dénominateur.
                _vo = ~torch.isnan(et_obs)
                _n = _vo.sum(dim=1).clamp(min=1)
                et_obs = torch.nan_to_num(et_obs, nan=0.0).sum(dim=1) / _n
                et_sim = (et_sim * _vo).sum(dim=1) / _n
                et_obs = torch.where(_vo.any(dim=1), et_obs, torch.full_like(et_obs, float("nan")))
            valid = ~torch.isnan(et_obs) & ~torch.isnan(et_sim)
            if valid.any():
                L_et = ((et_obs[valid] - et_sim[valid]) ** 2).mean()
                loss = loss + self.w_et * L_et
                components["et_loss"] = L_et

        if self.w_physics > 0 and water_balance_residual is not None:
            valid = ~torch.isnan(water_balance_residual)
            if valid.any():
                L_phys = (water_balance_residual[valid] ** 2).mean()
                loss = loss + self.w_physics * L_phys
                components["physics_loss"] = L_phys

        components["total"] = loss
        return loss, components


# ---------------------------------------------------------------------------
# CRPS loss for ensemble calibration
# ---------------------------------------------------------------------------


# Alias de compatibilite (nettoyage 2026-08-24, convention anglaise pour le code).


# Anciennes clés françaises de la section [loss], renommées le 2026-10-28 (identifiants en
# anglais). Une configuration ou une recette archivée qui les porte se relit telle quelle.
LEGACY_LOSS_KEYS = {"w_etiage": "w_low_flow", "w_fdc_bas": "w_fdc_low", "w_plat": "w_flat", "plat_seuil": "flat_threshold", "plat_mois": "flat_months", "plat_largeur": "flat_width"}


def normalize_loss_section(section: dict) -> dict:
    """Section [loss] avec les anciennes clés converties ; refuse une clé donnée sous ses deux noms."""
    out = dict(section)
    for old, new in LEGACY_LOSS_KEYS.items():
        if old in out:
            if new in out:
                raise ValueError(f"[loss] porte {old} et {new} : garder seulement {new}")
            out[new] = out.pop(old)
    return out

