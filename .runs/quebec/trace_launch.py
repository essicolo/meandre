"""Ce que lit réellement un lancement du pilote : variables, clés du TOML, fichiers, modules.

Exécute `etl_run.py` tel quel, dans le même processus, sous le nom de module `traced_script`
(le pilote ne dépend pas de `__main__`, et ce nom garde le traceur visible du compilateur de
PyTorch), et note au passage :
  - chaque variable d'environnement du projet consultée (ETL_, MEANDRE_, JOINT_, PROV_),
    sa valeur au moment de la lecture, ou son absence, et le fichier et la ligne qui la lisent ;
  - chaque clé lue dans un fichier TOML chargé par `tomllib.load` ;
  - chaque fichier de données ouvert, par Python, DuckDB ou xarray ;
  - chaque module du dépôt importé.
Le bilan est écrit en JSON, y compris quand le pilote se termine par `os._exit`.

    python .runs/quebec/trace_launch.py <bilan.json> [script, défaut etl_run.py]
"""
import json
import os
import runpy
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
PREFIXES = ("ETL_", "MEANDRE_", "JOINT_", "PROV_")
OUTPUT = sys.argv[1]
SCRIPT = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/.runs/quebec/etl_run.py"

env_reads = {}
toml_keys = {}
opened_files = set()


def relative(path):
    """Chemin relatif au dépôt quand il y est, absolu sinon, avec des barres obliques."""
    p = os.path.abspath(path).replace("\\", "/")
    return os.path.relpath(p, REPO).replace("\\", "/") if p.startswith(REPO) else p


def reader_location():
    """Fichier et ligne du dépôt qui lit la variable.

    None quand la lecture n'en est pas une : pose de la recette, ou bibliothèque tierce qui
    parcourt tout l'environnement (dask le fait à l'importation).
    """
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename.replace("\\", "/")
        if name.endswith("meandre/utils/recette.py") or "/.venv/" in name:
            return None
        if name.startswith(REPO) and not name.endswith("trace_launch.py"):
            return f"{relative(name)}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


# Variables : toute lecture passe par _Environ.__getitem__ (get, in et getenv compris).
_original_getitem = type(os.environ).__getitem__


def traced_getitem(self, key):
    # Pendant la compilation par torch.compile, lire la pile casse le graphe et ralentit le
    # calcul au point de le bloquer : la lecture passe alors sans être notée. Une variable lue
    # seulement dans du code compilé échappe donc au bilan.
    compiler = getattr(sys.modules.get("torch"), "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return _original_getitem(self, key)
    tracked = isinstance(key, str) and key.startswith(PREFIXES)
    location = reader_location() if tracked else None
    try:
        value = _original_getitem(self, key)
    except KeyError:
        if location:
            env_reads.setdefault(key, {"value": None, "read_by": []})["read_by"].append(location)
        raise
    if location:
        entry = env_reads.setdefault(key, {"value": None, "read_by": []})
        entry["value"] = value
        entry["read_by"].append(location)
    return value


type(os.environ).__getitem__ = traced_getitem


# Clés du TOML : le document chargé devient un dictionnaire qui note chaque clé consultée.
class TracedDict(dict):
    def __init__(self, data, prefix):
        super().__init__({k: wrap(v, f"{prefix}.{k}" if prefix else k) for k, v in data.items()})
        self._prefix = prefix

    def _record(self, key):
        toml_keys[f"{self._prefix}.{key}" if self._prefix else str(key)] = True

    def __getitem__(self, key):
        self._record(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._record(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._record(key)
        return super().__contains__(key)

    # Une copie par dict(section) ou un parcours passe par __iter__ ou keys() : toute la
    # section compte alors comme lue, faute de pouvoir suivre la copie.
    def __iter__(self):
        for key in super().keys():
            self._record(key)
        return super().__iter__()

    def keys(self):
        for key in super().keys():
            self._record(key)
        return super().keys()

    def items(self):
        for key in super().keys():
            self._record(key)
        return super().items()


def wrap(value, prefix):
    if isinstance(value, dict):
        return TracedDict(value, prefix)
    if isinstance(value, list):
        return [wrap(x, f"{prefix}[{i}]") for i, x in enumerate(value)]
    return value


_original_toml_load = tomllib.load


def traced_toml_load(f, *args, **kwargs):
    opened_files.add(relative(getattr(f, "name", "?")))
    return TracedDict(_original_toml_load(f, *args, **kwargs), "")


tomllib.load = traced_toml_load


# Fichiers : ouvertures Python par le crochet d'audit, bibliothèques compilées par enveloppe.
def record_file(path):
    try:
        p = os.path.abspath(os.fspath(path)).replace("\\", "/")
    except TypeError:
        return
    if any(s in p for s in (".venv", "site-packages", "/Python3", "/python3", "AppData")) or p.endswith(".pyc"):
        return
    opened_files.add(relative(p))


def audit_hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
        record_file(args[0])


sys.addaudithook(audit_hook)


def wrap_reader(module_name, function_name):
    try:
        module = __import__(module_name, fromlist=[function_name])
    except ImportError:
        return
    original = getattr(module, function_name)

    def reader(path=None, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            record_file(path)
        return original(path, *args, **kwargs) if path is not None else original(*args, **kwargs)

    setattr(module, function_name, reader)


for _module, _function in (("duckdb", "connect"), ("xarray", "open_dataset"), ("xarray", "open_zarr"), ("xarray", "open_mfdataset")):
    wrap_reader(_module, _function)


def write_report():
    modules = sorted({relative(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).replace("\\", "/").startswith(REPO) and ".venv" not in m.__file__})
    reads = {k: {"value": v["value"], "read_by": sorted(set(v["read_by"]))} for k, v in sorted(env_reads.items())}
    report = {"script": relative(SCRIPT), "env_reads": reads, "toml_keys": sorted(toml_keys), "opened_files": sorted(opened_files), "modules": modules}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


_original_exit = os._exit


def traced_exit(code):
    write_report()
    sys.stdout.flush()
    _original_exit(code)


os._exit = traced_exit
sys.argv = [SCRIPT]
try:
    runpy.run_path(SCRIPT, run_name="traced_script")
finally:
    write_report()
