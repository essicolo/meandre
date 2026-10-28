# loss-design

Études de la fonction de perte : redondance des termes, ordre de réponse, équilibrage des poids, bancs de perte et d'étiage (septembre 2026, voir CLAUDE.md, section sur la fonction de perte).

| script | rôle |
|---|---|
| `low-flow-bench.py` | La fonction objectif voit-elle un prélèvement en étiage |
| `loss-bench.py` | Banc de FONCTION DE PERTE : quel terme voit quel defaut, sans entrainer |
| `balance-weights.py` | Poser les poids de la perte par ÉQUILIBRAGE mesuré, au lieu de les régler au jugé |
| `response-order.py` | À quel ORDRE chaque terme de la perte répond-il à une petite erreur |
| `loss-redundancy.py` | Combien d'information DISTINCTE portent les termes de la perte |

Ces fichiers viennent de `.runs/quebec`, rangés le 28 octobre 2026. Ceux qui importent le pilote ou ses chargeurs suivent leur nouvel emplacement dans le paquet.
