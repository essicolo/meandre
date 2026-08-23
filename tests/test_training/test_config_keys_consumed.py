"""Toute cle de config que le pilote ignore doit etre CONNUE, jamais silencieuse.

Motif : trois fois en 24 heures (dettes #14, #15, #16 du registre), une construction
par enumeration a perdu en silence quelque chose d'ajoute ailleurs. Le cas #16 est
l'archetype : `patience = 8` pose dans 53 configs avec dix lignes de justification, et
jamais transmis a TrainingConfig parce que l'enumeration du pilote ne reprenait pas la
cle -- l'arret anticipe annonce a Essi n'a jamais existe, et le run aux-A a fait dix
epoques d'effondrement sans s'arreter. Le meme balayage a aussi attrape `grad_clip`,
lu sous le mauvais nom (`clip_grad_norm`) depuis toujours.

Ce test compare les cles du bloc [training] des TOML aux cles que `etl_run.py` consomme
reellement (analyse du source : `tcfg.get(...)` et `tcfg[...]`). Une cle TOML ni
consommee ni presente dans la LISTE D'EXCEPTIONS ci-dessous fait echouer la suite. Pour
ignorer une cle volontairement, il faut l'inscrire ici AVEC sa raison : c'est le prix
du silence.
"""
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:                                     # Python < 3.11
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PILOTE = REPO / ".runs" / "quebec" / "etl_run.py"
CONFIGS = sorted((REPO / ".runs" / "quebec" / "config").glob("*-v4.toml"))

# Cles [training] IGNOREES PAR LE PILOTE QUEBECOIS, chacune avec sa raison. Ce bloc est
# un registre, pas une poubelle : toute entree sans raison verifiable doit sauter.
IGNOREES_SCIEMMENT = {
    "n_epochs": "remplace par ETL_EPOCHS (l'experimentation fixe la duree au lancement)",
    "best_metric": "code en dur kge_median dans le pilote : les runs quebecois se comparent tous dessus",
    "val_every": "code en dur 1 : la validation par epoque est le protocole des diagnostics",
    "warm_start": "le pilote gere la reprise par ETL_TAG et le chargement explicite du checkpoint",
    "lr_finetune": "pas de warm-start TOML dans ce pilote (voir warm_start)",
    "lr_new_features_mult": "idem, mecanisme de warm-start non utilise ici",
    "weight_decay": "jamais active au Quebec ; l'optimiseur du trainer garde son defaut",
    "compile_modules": "compile_column retire le 2026-08-19 (dette #8) ; cle morte a purger des TOML",
    "autopilot_beta_penalty": "autopilot du pilote ne module pas les penalites beta/gamma",
    "autopilot_gamma_penalty": "idem",
    "autopilot_gamma_threshold": "idem (seul beta_threshold participe au drift handler)",
    "enable_residual_epoch": "correcteur residuel LEGACY, inactif (CLAUDE.md)",
    "enable_temporal_epoch": "GRU temporel LEGACY, inactif",
    "enable_travel_epoch": "idem",
    "residual_warmup_epochs": "correcteur residuel LEGACY",
    "train_with_param_noise": "stack ParamNoise DEPRECATED 2026-05-11",
    "param_noise_target_sigma": "idem",
    "w_param_noise_kl": "idem",
    "w_concrete_kl": "ConcreteDropout DEPRECATED 2026-05-11",
    "tta_warmup_epochs": "mecanisme abandonne avec le stack ensembliste",
}


def _consumed_keys() -> set:
    src = PILOTE.read_text(encoding="utf-8")
    return (set(re.findall(r'tcfg\.get\("([a-z_0-9]+)"', src))
            | set(re.findall(r'tcfg\["([a-z_0-9]+)"\]', src)))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_toute_cle_training_est_consommee_ou_declaree(config):
    toml_keys = set(tomllib.loads(config.read_text(encoding="utf-8"))["training"].keys())
    silencieuses = toml_keys - _consumed_keys() - set(IGNOREES_SCIEMMENT)
    assert not silencieuses, (
        f"{config.name} : cles [training] que le pilote IGNORE EN SILENCE : "
        f"{sorted(silencieuses)}. Soit les consommer dans etl_run.py, soit les inscrire "
        f"dans IGNOREES_SCIEMMENT avec leur raison.")


def test_patience_est_reellement_consommee():
    """Verrou nominal de la dette #16 : la cle qui a coute l'arret anticipe."""
    assert "patience" in _consumed_keys()


def test_grad_clip_est_lu_sous_son_vrai_nom():
    """Verrou du second bug attrape par ce balayage : la cle lue sous un autre nom."""
    src = PILOTE.read_text(encoding="utf-8")
    assert 'tcfg.get("grad_clip"' in src
    assert 'tcfg.get("clip_grad_norm"' not in src


def test_la_liste_d_exceptions_ne_couvre_pas_des_cles_consommees():
    """Une cle a la fois consommee ET declaree ignoree est un mensonge de registre."""
    doubles = set(IGNOREES_SCIEMMENT) & _consumed_keys()
    assert not doubles, f"retirees de IGNOREES_SCIEMMENT : {sorted(doubles)}"
