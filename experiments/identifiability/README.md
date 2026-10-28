# identifiability

Identifiabilité : ce que chaque source de données apporte, indice de débit de base observé, plafond de percolation préféré par station.

| script | rôle |
|---|---|
| `baseflow-index-identifiability.py` | Quelles covariables prédisent l'indice d'écoulement de base d'une station |
| `observed-baseflow-index.py` | Indice d'écoulement de base MESURÉ sur les hydrogrammes observés |
| `snow-information.py` | Combien de contraintes INDÉPENDANTES chaque source de neige apporte-t-elle |
| `data-information.py` | Combien de directions indépendantes TOUTES les observations portent-elles ensemble |
| `preferred-ceiling.py` | Plafond de percolation préféré par station, et ce qui le prédit |

Ces fichiers viennent de `.runs/quebec`, rangés le 28 octobre 2026. Ceux qui importent le pilote ou ses chargeurs suivent leur nouvel emplacement dans le paquet.
