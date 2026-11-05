"""Clone fidèle des modules d'Hydrotel 4.3.6 (C++, INRS-ETE), porté en PyTorch.

Chaque module a été validé unité hydrologique par unité hydrologique contre le binaire
(neige en degrés-jours modifiés, gel de Rankinen, sol BV3C2 à trois couches, milieux humides
isolés). Les ETP de McGuinness, de Linacre et d'Hydro-Québec ont été retirées le 2026-11-05 :
le modèle n'emploie que Penman-Monteith (`meandre/vertical/evapotranspiration.py`). La colonne verticale
(`meandre/vertical/hydrotel_column.py`) les orchestre ; sans réglage, elle reproduit le binaire.
"""
