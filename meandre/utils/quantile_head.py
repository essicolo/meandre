"""Tête de régression quantile per-nœud.

Prédit K offsets δ_τ depuis μ (médiane par construction = μ, donc δ_0.5 = 0
implicite — on ne prédit jamais le quantile 0.5). Architecture :

    spatial_params (N, P) → MLP → 2K sorties (K log-width centers + K Q-slopes)
    log_w_τ(t, n) = a_τ(n) + b_τ(n) · log(|Q(t, n)| + ε)
    w_τ = exp(log_w_τ)                              # largeurs positives
    Monotonie par cumsum de chaque côté de la médiane :
      δ_τ<0.5 = −cumsum(w des plus proches de 0.5 vers les extrêmes)
      δ_τ>0.5 = +cumsum                              (idem)

→ q_τ(t, n) = μ(t, n) + δ_τ(t, n), strictement croissant en τ.

Hétéroscédasticité : les largeurs grossissent avec |Q| via b_τ · log|Q|
(comme le SpatialNoiseHead actuel pour σ). Init b_τ ≈ 1 → mise à l'échelle
quasi-linéaire avec Q ; a_τ ≈ −1 → largeurs initiales modestes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QuantileHead(nn.Module):
    """Per-node quantile head with monotone offsets from μ.

    Parameters
    ----------
    n_spatial_params : int
        Number of spatial parameters per node (input to the MLP).
    hidden : int
        Hidden layer width.
    taus : tuple of floats in (0, 1) excluding 0.5
        Quantile levels to predict. Sorted ascending internally.
    eps : float
        Stabiliser inside ``log(|Q| + ε)``.
    """

    def __init__(
        self,
        n_spatial_params: int = 36,
        hidden: int = 32,
        taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95),
        eps: float = 1.0,
    ) -> None:
        super().__init__()
        taus_sorted = sorted(set(taus))
        if any(t <= 0.0 or t >= 1.0 or t == 0.5 for t in taus_sorted):
            raise ValueError(f"taus must be in (0,1) and exclude 0.5; got {taus}")
        self.taus = tuple(taus_sorted)
        self.K = len(self.taus)
        self.n_lower = sum(1 for t in self.taus if t < 0.5)
        self.n_upper = self.K - self.n_lower
        self.eps = eps

        # MLP : spatial_params → 2K (K width-centers a_τ + K Q-slopes b_τ)
        self.net = nn.Sequential(
            nn.Linear(n_spatial_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.K),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            # a_τ ≈ −1 → largeur init exp(−1) ≈ 0.37
            self.net[-1].bias[: self.K] = -1.0
            # b_τ ≈ 1 → mise à l'échelle hétéroscédastique log|Q|
            self.net[-1].bias[self.K:] = 1.0

    def forward(self, spatial_params: Tensor, Q: Tensor) -> Tensor:
        """
        Parameters
        ----------
        spatial_params : Tensor, shape (N, P)
        Q : Tensor, shape (T, N) — débit prédit (μ)

        Returns
        -------
        offsets : Tensor, shape (T, N, K) — δ_τ tel que q_τ = μ + δ_τ.
                  Strictement croissant le long de self.taus.
        """
        raw = self.net(spatial_params)              # (N, 2K)
        a = raw[:, : self.K]                         # (N, K) log-width centers
        b = raw[:, self.K:]                          # (N, K) Q-slopes
        log_q = torch.log(Q.abs() + self.eps)       # (T, N)

        # log_w (T, N, K) = a + b · log|Q|, broadcast
        log_w = a.unsqueeze(0) + b.unsqueeze(0) * log_q.unsqueeze(-1)
        w = log_w.exp()                              # (T, N, K) > 0

        # ENVELOPPE MULTIPLICATIVE, EN LOG (2026-09-11). Les largeurs cumulees etaient
        # appliquees ADDITIVEMENT au debit, si bien que le cote bas pouvait passer sous
        # zero : mesure sur cinq regions, le cinquieme centile predit etait negatif ou nul
        # dans 21 a 44 % des pas de temps, et la classe inferieure du diagramme de
        # Talagrand etait vide PAR CONSTRUCTION, une observation ne pouvant pas descendre
        # sous un debit negatif. Un debit est strictement positif et sa distribution est
        # log-normale : les largeurs se cumulent donc dans le LOGARITHME du debit, ce qui
        # rend chaque quantile strictement positif et l'enveloppe dissymetrique comme la
        # variable. Le contrat exterieur ne change pas, la fonction rend toujours des
        # ecarts a ajouter a la mediane.
        # MEANDRE_QUANTILE_ADDITIF=1 restitue l'ancienne forme, pour comparaison seulement.
        w_lower = w[..., : self.n_lower]             # (T, N, n_lower)
        cum_lower = -w_lower.flip(-1).cumsum(dim=-1).flip(-1)   # <= 0, ordre τ ascendant
        w_upper = w[..., self.n_lower:]              # (T, N, n_upper)
        cum_upper = w_upper.cumsum(dim=-1)           # >= 0, ordre τ ascendant
        cum = torch.cat([cum_lower, cum_upper], dim=-1)         # (T, N, K)
        import os as _os
        if _os.environ.get("MEANDRE_QUANTILE_ADDITIF", "0") == "1":
            return cum
        # ECHELLE (2026-09-11, deuxieme passe). Les largeurs `w` sont calibrees en unites
        # de DEBIT : les reutiliser telles quelles comme largeurs en LOGARITHME les fait
        # exploser. Mesure sur quatre regions apres la premiere version de ce correctif :
        # le cumul atteignait la borne de 30, soit un quatre-vingt-quinzieme centile a
        # 10^13 fois la mediane et un cinquieme centile numeriquement nul, donc une queue
        # basse toujours vide et une couverture de 96 % pour un intervalle annonce a 90.
        # On ramene donc la largeur dans un domaine logarithmique raisonnable : un facteur
        # ECH par niveau, et un cumul borne a BORNE, soit au plus un facteur e^BORNE entre
        # un quantile extreme et la mediane. A l'initialisation le cinquieme centile vaut
        # environ 0,9 fois la mediane, et l'entrainement elargit si les observations le
        # demandent.
        # TROISIEME PASSE, et cette fois un test l'a attrapee avant l'entrainement. Les
        # largeurs valent exp(a + b.log Q), soit une PUISSANCE du debit : en unites de
        # debit c'est voulu, une grande riviere a une enveloppe plus large en valeur
        # absolue. Reprises comme largeurs logarithmiques, elles font croitre la largeur
        # RELATIVE avec le debit, et le test l'a mesure : a 1 m3/s l'enveloppe allait de
        # 0,90 a 1,12 fois la mediane, a 100 m3/s de 0,018 a 55. On passe donc par une
        # fonction douce du meme predicteur lineaire, qui croit comme log Q et non comme
        # une puissance de Q, si bien que la largeur relative ne derive que lentement.
        _ECH, _BORNE = 0.05, 4.0
        s = torch.nn.functional.softplus(log_w)                 # (T, N, K) > 0, doux
        s_lower = s[..., : self.n_lower]
        cum_l = -s_lower.flip(-1).cumsum(dim=-1).flip(-1)
        s_upper = s[..., self.n_lower:]
        cum_u = s_upper.cumsum(dim=-1)
        cum_log = (_ECH * torch.cat([cum_l, cum_u], dim=-1)).clamp(-_BORNE, _BORNE)
        q_pos = Q.clamp(min=0.0).unsqueeze(-1)
        return q_pos * (torch.exp(cum_log) - 1.0)
