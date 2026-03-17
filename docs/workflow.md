

```
┌─────────────────────┐     ┌─────────────────────┐
│  Branche PHYSITEL    │     │  Branche open data   │
│  (troncons.parquet,  │     │  (DEM, land cover,   │
│   UHRH polygons,     │     │   soil via APIs)     │
│   HYDROTEL files)    │     │                      │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         ▼                           ▼
   ┌─────────────────────────────────────┐
   │         basin.duckdb                │
   │  (nodes, edges, territorial,        │
   │   withdrawals)                      │
   └──────────────┬──────────────────────┘
                  │
                  │  + forcing.nc (ERA5 ou stations)
                  │  + observations (HYDAT, débits)
                  ▼
   ┌─────────────────────────────────────┐
   │         Entraînement                │
   │  basin.duckdb + forcing.nc + obs    │
   └──────────────┬──────────────────────┘
                  │
                  ▼
              best.pt
                  │
                  │  + forcing.nc (nouveau scénario)
                  ▼
   ┌─────────────────────────────────────┐
   │         Prédiction                  │
   │  best.pt + basin.duckdb + forcing.nc│
   └─────────────────────────────────────┘
```