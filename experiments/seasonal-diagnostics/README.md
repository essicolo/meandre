# seasonal-diagnostics

Diagnostics saisonniers sans simulation ou en passe avant : évapotranspiration, été, étiage, pluie CaSR contre stations, humidité du sol, fermeture des bilans.

| script | rôle |
|---|---|
| `low-flow-diagnostics.py` | Sens de l'erreur sur les jours d'étiage, par saison, à partir d'un export ETL_DUMP_Q |
| `et-sources-compared.py` | Evapotranspiration mensuelle de trois sources sur les noeuds d'un territoire : SSEBop v6.1 |
| `summer-hydrotel-vs-meandre.py` | L'ensemble Hydrotel a-t-il l'ete d'un territoire ? Rapport simule sur observe par mois |
| `territory-evaluation.py` | Tableau récapitulatif Québec : held-out 2022-2024 par région, méandre vs Hydrotel BRUT |
| `basin-closure.py` | Fermeture du bilan d'eau annuel par bassin jaugé, sans modèle (2026-10-09) |
| `c3s-soil-moisture.py` | Humidité du sol satellitaire C3S (ESA CCI, combinée) sur les nœuds d'un territoire (2026-10-09) |
| `station-indicators.py` | Indicateurs d'étiage et KGE par station, à partir des séries aux stations exportées par le |
| `casr-rain-vs-stations.py` | La pluie d'été de CaSR est-elle trop forte sur les sous-bassins du banc ? Comparaison aux stations GHCN |
| `station-seasons.py` | Rapport simule sur observe par saison, MEDIANE DES STATIONS, pour des exports de stations |
| `contrasted-stations.py` | Par station, deux territoires : KGE et ete de meandre et d'Hydrotel contre les attributs du bassin amont |
| `field-trajectory.py` | Trajectoire d'une sortie du champ au fil des époques, moyennée sur le bassin amont de stations |

Ces fichiers viennent de `.runs/quebec`, rangés le 28 octobre 2026. Ceux qui importent le pilote ou ses chargeurs suivent leur nouvel emplacement dans le paquet.
