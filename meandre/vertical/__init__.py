# Colonne native (VerticalColumn) + ses modules (snow/frost/interception/soil/
# aquifer/wetland natifs) RETIRÉS 2026-06-27 (ETP/baseflow déficients ; remplacés
# par la HydrotelColumn fidèle dans hydrotel_column.py, qui s'appuie sur le
# sous-paquet hydrotel_clone/). Seul ETModule subsiste (Penman-Monteith, utilisé
# par le mode et_mode="penman" de la colonne hydrotel).
from meandre.vertical.evapotranspiration import ETModule

__all__ = ["ETModule"]
