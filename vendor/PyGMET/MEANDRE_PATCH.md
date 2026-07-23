# PyGMET vendoré pour meandre (pipeline reproductible CaSR+PyGMET)

Source : github.com/NCAR/PyGMET (clone 2026-07-23).

## Patch appliqué (src/regression.py)
`least_squares_ludcmp` -> `least_squares_numpy` aux 2 sites d'appel (l.192, l.235).
Le solveur maison ludcmp lève ZeroDivisionError sur bloc local rang-déficient ;
`least_squares_numpy` a le garde-fou `det==0 -> coeffs 0`. Résultats identiques
sur matrices bien conditionnées.

## Usage
Entrées générées par .runs/quebec/build_pygmet_inputs.py (stations ECCC + MNT PHYSITEL).
`python vendor/PyGMET/src/main.py <config.toml>`
Config déterministe (forçage) : ensemble_flag=false ; prédicteurs lat/lon/elev
(les pentes rendaient des blocs singuliers). Ensemble (phase 2 incertitude) = ensemble_flag=true.
