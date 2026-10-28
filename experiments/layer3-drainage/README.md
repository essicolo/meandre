# Couche 3 tirée du sol et forme par classe de drainage

Expérience des 27 et 28 octobre 2026. Les verdicts sont au registre, de R410 à R435. Ce dossier garde ce qu'il faut pour rejouer les passes.

## Résultat retenu

La forme de la couche 3 se choisit par tronçon selon la classe de drainage. Les tronçons mal drainés, de classe supérieure à 4,5, gardent la loi à seuil d'Hydrotel. Les autres prennent une vidange de Darcy, dont la conductivité est apprise sous un prior centré sur la classe de drainage. Sans réentraîner, la perte 2019-2024 baisse de 5,5 % en Montérégie et de 0,8 % en Outaouais, et reste égale au Saint-Laurent sud-ouest (R432). Cette forme est portée dans la configuration provinciale `.runs/quebec/config/modele-provincial-2026-11.toml`.

## Contenu

| fichier | rôle | registre |
|---|---|---|
| `conflict_probe.py` | dérivée des erreurs de crue et d'étiage par champ, par horizon et par groupe de stations ; appelée par le pilote (`MEANDRE_SONDE_CONFLIT=1`) | R410, R411, R415 |
| `config/darcy.toml` | couche 3 entièrement de Darcy | R416 à R423 |
| `config/darcy-substratum.toml` | conductivité imposée par la classe de drainage | R422, réfuté |
| `config/hybrid.toml` | latéral à seuil, percolation de Darcy | R425, réfuté |
| `config/anisotropic.toml` | percolation au substratum, latéral sur la conductivité de la couche | R426, réfuté |
| `config/darcy-lateral.toml` | latéral de Darcy, percolation à seuil | R427, réfuté |
| `config/drainage-class.toml` | forme par classe de drainage | R432, retenu |

Les constructeurs de tables lues par la configuration restent dans `.runs/quebec` : `audit_stations.py` (rattachement des stations, R430) et `epaisseur_depots.py`.

Le registre cite ces fichiers sous leurs premiers noms. Ils ont été renommés le 28 octobre pour suivre la règle des identifiants en anglais :

- `sonde_conflit.py` : `conflict_probe.py` ;
- `tmp-darcy.toml` : `darcy.toml` ;
- `tmp-darcy-substratum.toml` : `darcy-substratum.toml` ;
- `tmp-darcy-hybride.toml` : `hybrid.toml` ;
- `tmp-darcy-aniso.toml` : `anisotropic.toml` ;
- `tmp-darcy-lateral.toml` : `darcy-lateral.toml` ;
- `tmp-darcy-mixte.toml` : `drainage-class.toml` ;
- table `rattachement-stations.csv` : `station-attachment.csv`.

## Rejouer une passe

Depuis la racine du dépôt, modèle gelé, sans entraînement :

```bash
ETL_CONFIG=experiments/layer3_drainage/config/drainage-class.toml ETL_REGION=mont ETL_EPOCHS=0 ETL_HELDOUT=2019-01-01,2024-12-31 MEANDRE_RAPPORT_AIRES=1 MEANDRE_TABLE_RATTACHEMENT={DERIVES}/auxiliaires/station-attachment.csv MEANDRE_PRIORS=1 MEANDRE_PRIOR_CLES=K_sat_3 MEANDRE_PRIOR_KSAT3_CENTRE=substratum MEANDRE_PRIOR_DEPART_CENTRE=K_sat_3 MEANDRE_PRIOR_CLES_NIVEAU=K_sat_3 ETL_WARM_FROM=.runs/quebec/checkpoints/best-quebec-etl-qv848-e32.pt python .runs/quebec/etl_run.py
```

Les points de reprise entraînés sous prior du dépôt et les journaux sont sur Ubuntu, dans `planml-out`.
