"""Contrat de domaine : ce que le modèle sait du monde (2026-10-26, étape 1 de l'architecture cible)."""
from meandre.domain.domain import Domain, Observation
from meandre.domain.manifest import Manifest, Source, fingerprint, register, current
from meandre.domain.vocabulary import ATTRIBUTES, check_columns, field_columns, physical_columns

__all__ = ["Domain", "Observation", "Manifest", "Source", "fingerprint", "register", "current", "ATTRIBUTES", "check_columns", "field_columns", "physical_columns"]
