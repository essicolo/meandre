"""La recette provinciale déclare tous les réglages qui définissent le modèle : rien n'est pris en silence dans le code."""
import re
import tomllib
from pathlib import Path

from meandre.utils.settings import SCHEMA

RECIPE = Path(__file__).resolve().parents[1] / ".runs" / "quebec" / "recipe.toml"
# Outils de lancement, sans effet sur le modèle : diagnostics, sondes, débogage, banc.
TOOLS = {"diagnostics", "probe", "debug", "bench"}
TOOL_KEYS = {"forcing.synthetic"}


def test_every_model_setting_is_written_in_the_recipe():
    text = RECIPE.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    missing = []
    for key in SCHEMA:
        section, name = key.split(".", 1)
        if section in TOOLS or key in TOOL_KEYS:
            continue
        if name in (data.get(section) or {}):
            continue
        # Un réglage sans valeur par défaut s'écrit en commentaire, « # nom = (absent) », dans sa section.
        block = re.search(rf"^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M)
        if block and re.search(rf"^# {re.escape(name)} = \(absent\)", block.group(1), re.M):
            continue
        missing.append(key)
    assert not missing, f"réglages absents de {RECIPE.name} : {missing}"


def _declared(text, section, name):
    """Une clé posée dans sa section, ou écrite en commentaire « # nom = (absent) »."""
    if name in (tomllib.loads(text).get(section) or {}):
        return True
    block = re.search(rf"^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M)
    return bool(block and re.search(rf"^# {re.escape(name)} = \(absent\)", block.group(1), re.M))


def test_pilot_sections_are_written_in_the_recipe():
    import inspect

    from meandre.training.loss import HydroLoss

    text = RECIPE.read_text(encoding="utf-8")
    pilot = (RECIPE.parents[2] / "meandre" / "run" / "pilot.py").read_text(encoding="utf-8")
    missing = []
    for var, section in (("tcfg", "training"), ("mcfg", "model")):
        for name in set(re.findall(rf'\b{var}\.get\("([a-z_0-9]+)"', pilot)):
            if not _declared(text, section, name):
                missing.append(f"[{section}] {name}")
    for name in inspect.signature(HydroLoss.__init__).parameters:
        if name.startswith("w_") and not _declared(text, "loss", name):
            missing.append(f"[loss] {name}")
    # Cles de [loss] qui ne sont pas des poids (et_mode, forme du terme de platitude...), lues par l'assemblage du territoire.
    territory = (RECIPE.parents[2] / "meandre" / "run" / "territory_data.py").read_text(encoding="utf-8")
    for name in set(re.findall(r'\blcfg\.get\("([a-z_0-9]+)"', territory)):
        if not name.startswith("w_") and not _declared(text, "loss", name):
            missing.append(f"[loss] {name}")
    assert not missing, f"cles lues par le pilote, absentes de {RECIPE.name} : {missing}"
