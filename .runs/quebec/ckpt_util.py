"""Utilitaire d'évaluation : un champion porte des codes latents (effet aléatoire par
nœud) qui n'existent QUE pour sa région d'origine. Les activer sur une autre région est
impossible (dimensions différentes) ; les désactiver sur sa propre région sous-estime le
score (mesuré sur gasp/CaSR brut le 3 août : 0.686 sans, 0.702 avec). Ce test regarde le
point de contrôle et tranche automatiquement, au lieu d'un drapeau fixe faux dans un cas
sur deux."""
import torch


def a_des_latents(ckpt_path: str, n_nodes: int) -> bool:
    """True si le checkpoint porte des codes latents dimensionnés pour CETTE région."""
    try:
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = c["state_dict"] if isinstance(c, dict) and "state_dict" in c else c
        z = sd.get("spatial_encoder.latent_codes")
        return z is not None and z.shape[0] == n_nodes
    except Exception:
        return False
