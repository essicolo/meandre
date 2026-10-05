"""Les prélèvements suivent la tranche du forçage dans chaque simulation de l'entraîneur.

La simulation lit les prélèvements à l'indice de pas compté depuis le début de l'appel.
Jusqu'au 2026-10-05, l'entraîneur passait la série complète avec le forçage d'un bloc :
chaque bloc lisait les prélèvements des premiers jours de la série, nuls en 2000, si bien
que tous les modèles ont été entraînés et validés sans prélèvements.
"""
import re
from pathlib import Path

import torch

from meandre.routing.withdrawals import WithdrawalData


def test_tranche_alignee():
    net = torch.arange(10, dtype=torch.float32).repeat(3, 1).T
    w = WithdrawalData(net=net, net_gw=-net)
    t = w.slice(4, 7)
    assert t.net.shape == (3, 3)
    assert float(t.net_withdrawal(0)[0]) == 4.0
    assert float(t.gw_withdrawal(2)[0]) == -6.0


def test_entraineur_ne_passe_jamais_la_serie_complete():
    src = Path("meandre/training/trainer.py").read_text(encoding="utf-8")
    appels = re.findall(r"withdrawals=data\.withdrawals[^\n]*", src)
    assert appels, "aucun appel trouvé"
    for a in appels:
        assert ".slice(" in a, f"série complète passée à une simulation tranchée : {a}"
