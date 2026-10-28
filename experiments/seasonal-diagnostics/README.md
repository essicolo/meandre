# seasonal-diagnostics

Diagnostics saisonniers sans simulation ou en passe avant : évapotranspiration, été, étiage, pluie CaSR contre stations, humidité du sol, fermeture des bilans.

| script | rôle |
|---|---|
| `diag_etiage.py` | Sens de l'erreur sur les jours d'étiage, par saison, à partir d'un export ETL_DUMP_Q |
| `et_sources_comparees.py` | Evapotranspiration mensuelle de trois sources sur les noeuds d'un territoire : SSEBop v6.1 |
| `ete_hydrotel_vs_meandre.py` | L'ensemble Hydrotel a-t-il l'ete d'un territoire ? Rapport simule sur observe par mois |
| `eval_regions.py` | Tableau récapitulatif Québec : held-out 2022-2024 par région, méandre vs Hydrotel BRUT |
| `fermeture_bassins.py` | Fermeture du bilan d'eau annuel par bassin jaugé, sans modèle (2026-10-09) |
| `humidite_sol_c3s.py` | Humidité du sol satellitaire C3S (ESA CCI, combinée) sur les nœuds d'un territoire (2026-10-09) |
| `indicateurs_stations.py` | Indicateurs d'étiage et KGE par station, à partir des séries aux stations exportées par le |
| `pluie_casr_vs_stations.py` | La pluie d'été de CaSR est-elle trop forte sur les sous-bassins du banc ? Comparaison aux stations GHCN |
| `saisons_stations.py` | Rapport simule sur observe par saison, MEDIANE DES STATIONS, pour des exports de stations |
| `stations_contrastees.py` | Par station, deux territoires : KGE et ete de meandre et d'Hydrotel contre les attributs du bassin amont |
| `trajectoire_champ.py` | Trajectoire d'une sortie du champ au fil des époques, moyennée sur le bassin amont de stations |

Ces fichiers viennent de `.runs/quebec`, rangés le 28 octobre 2026. Ceux qui importent le pilote ou ses chargeurs suivent leur nouvel emplacement dans le paquet.
