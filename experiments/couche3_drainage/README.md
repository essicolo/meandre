# Couche 3 tirée du sol et forme par classe de drainage

Expérience des 27 et 28 octobre 2026. Les verdicts sont au registre, de R410 à R435. Ce dossier garde ce qu'il faut pour rejouer les passes.

## Résultat retenu

La forme de la couche 3 se choisit par tronçon selon la classe de drainage. Les tronçons mal drainés, de classe supérieure à 4,5, gardent la loi à seuil d'Hydrotel. Les autres prennent une vidange de Darcy, dont la conductivité est apprise sous un prior centré sur la classe de drainage. Sans réentraîner, la perte 2019-2024 baisse de 5,5 % en Montérégie et de 0,8 % en Outaouais, et reste égale au Saint-Laurent sud-ouest (R432). Cette forme est portée dans la configuration provinciale `.runs/quebec/config/modele-provincial-2026-11.toml`.

## Contenu

| fichier | rôle | registre |
|---|---|---|
| `sonde_conflit.py` | dérivée des erreurs de crue et d'étiage par champ, par horizon et par groupe de stations ; appelée par le pilote (`MEANDRE_SONDE_CONFLIT=1`) | R410, R411, R415 |
| `config/tmp-darcy.toml` | couche 3 entièrement de Darcy | R416 à R423 |
| `config/tmp-darcy-substratum.toml` | conductivité imposée par la classe de drainage | R422, réfuté |
| `config/tmp-darcy-hybride.toml` | latéral à seuil, percolation de Darcy | R425, réfuté |
| `config/tmp-darcy-aniso.toml` | percolation au substratum, latéral sur la conductivité de la couche | R426, réfuté |
| `config/tmp-darcy-lateral.toml` | latéral de Darcy, percolation à seuil | R427, réfuté |
| `config/tmp-darcy-mixte.toml` | forme par classe de drainage | R432, retenu |

Les constructeurs de tables lues par la configuration restent dans `.runs/quebec` : `audit_stations.py` (rattachement des stations, R430) et `epaisseur_depots.py`.

## Rejouer une passe

Depuis la racine du dépôt, modèle gelé, sans entraînement :

```bash
ETL_CONFIG=experiments/couche3_drainage/config/tmp-darcy-mixte.toml ETL_REGION=mont ETL_EPOCHS=0 ETL_HELDOUT=2019-01-01,2024-12-31 MEANDRE_RAPPORT_AIRES=1 MEANDRE_TABLE_RATTACHEMENT={DERIVES}/auxiliaires/rattachement-stations.csv MEANDRE_PRIORS=1 MEANDRE_PRIOR_CLES=K_sat_3 MEANDRE_PRIOR_KSAT3_CENTRE=substratum MEANDRE_PRIOR_DEPART_CENTRE=K_sat_3 MEANDRE_PRIOR_CLES_NIVEAU=K_sat_3 ETL_WARM_FROM=.runs/quebec/checkpoints/best-quebec-etl-qv848-e32.pt python .runs/quebec/etl_run.py
```

Les points de reprise entraînés sous prior du dépôt et les journaux sont sur Ubuntu, dans `planml-out`.
