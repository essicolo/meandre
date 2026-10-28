"""La recette d'une exécution : capturer ce qui définit réellement un modèle.

POURQUOI CE MODULE (inventaire du 2026-09-03). La configuration effective d'une
exécution ne se lit ni dans le fichier TOML ni dans le point de reprise, mais dans le
bloc de variables d'environnement du script shell qui l'a lancée. Sur les 54 fichiers de
configuration du domaine québécois, DEUX sont réellement chargés par le code, et le
pilote d'entraînement lit à lui seul 79 variables d'environnement, dont l'essentiel de la
physique : loi des ancrages, seuil pluie-neige, amplitude de fonte, aquifère, milieux
humides, lacs, bornes du routage. Aucun de ces soixante-dix réglages n'avait de
représentation persistée ailleurs que dans un script.

La conséquence a un nom dans le registre : « un point de reprise seul ne définit PAS un
modèle », et elle a coûté cinq évaluations faussées en quatre jours. La fiche d'exécution
posée le 2026-08-17 a traité la partie modèle, neuf drapeaux d'état. Ce module traite la
partie recette : toutes les variables d'environnement du projet effectivement posées,
plus l'empreinte du code, écrites DANS le point de reprise et à côté de lui en TOML
lisible.

Le fichier TOML produit est un document, pas une entrée : il enregistre ce qui a servi.
Le rendre exécutable en entrée est l'étape suivante, et elle demande que les pilotes
lisent la configuration au lieu de l'environnement.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

# Préfixes des variables qui pilotent le projet. PROV et JOINT sont les pilotes
# provincial et conjoint, ETL le pilote régional, MEANDRE le paquet lui-même.
PREFIXES = ("ETL_", "MEANDRE_", "JOINT_", "PROV_")

# Jamais capturé : secrets et jetons. La recette est destinée à être lue et partagée.
FORBIDDEN = ("TOKEN", "SECRET", "PASSWORD", "MOTDEPASSE", "KEY", "CODE")


def _est_secret(field_name: str) -> bool:
    return any(m in field_name.upper() for m in FORBIDDEN)


def _fingerprint_code() -> dict:
    """Révision git et état de l'arbre, si le dépôt est disponible."""
    out = {}
    root = Path(__file__).resolve().parents[2]
    for key, cmd in (("revision", ["git", "rev-parse", "HEAD"]),
                     ("branche", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                out[key] = r.stdout.strip()
        except Exception:
            pass
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["arbre_propre"] = (r.stdout.strip() == "")
    except Exception:
        pass
    return out


def capture_recipe() -> dict:
    """Toutes les variables du projet posées dans l'environnement, plus le contexte.

    Seules les variables EFFECTIVEMENT posées sont capturées : une variable absente
    signifie que le défaut du code s'applique, et le défaut est dans le code, pas ici.
    """
    variables = {n: v for n, v in os.environ.items()
                 if n.startswith(PREFIXES) and not _est_secret(n)}
    context = {"python": platform.python_version(), "hote": platform.node(),
                "systeme": platform.system()}
    try:
        import torch
        # str() OBLIGATOIRE : torch.__version__ est un objet TorchVersion, que le
        # chargement securise de torch (weights_only, defaut depuis 2.6) refuse de
        # deserialiser. Sans cette conversion, tout point de reprise ecrit avec une
        # recette devient illisible.
        context["torch"] = str(torch.__version__)
        context["cuda"] = str(torch.version.cuda or "aucun")
    except Exception:
        pass
    from meandre.utils import settings
    from meandre.domain import manifest as _manifest
    _m = _manifest.current()
    return {"variables": dict(sorted(variables.items())), "settings": settings.snapshot(),
            "contexte": context, "code": _fingerprint_code(), "domain": _m.as_dict() if _m else {}}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_recipe_toml(path, recette: dict, record: dict | None = None,
                        init_kwargs: dict | None = None) -> Path | None:
    """Écrit la recette en TOML lisible. Retourne le chemin, ou None en cas d'échec.

    L'échec n'est JAMAIS fatal : perdre le document accompagnant ne doit pas faire
    perdre le point de reprise.
    """
    try:
        p = Path(path).with_suffix(".recipe.toml")
        lines = [
            "# Recette d'exécution de meandre, écrite automatiquement à la sauvegarde.",
            "# Ce document enregistre ce qui a produit le point de reprise du même nom :",
            "# un point de reprise seul ne définit pas un modèle.",
            "",
            "[contexte]",
        ]
        for k, v in sorted(recette.get("contexte", {}).items()):
            lines.append(f"{k} = {_toml_value(v)}")
        if recette.get("code"):
            lines += ["", "[code]"]
            for k, v in sorted(recette["code"].items()):
                lines.append(f"{k} = {_toml_value(v)}")
        if record:
            lines += ["", "# État du modèle au moment de la sauvegarde : occupation du sol,",
                       "# milieux humides, phénologie, noyau de versant, lacs, aquifère, ETP.",
                       "[modele]"]
            for k, v in sorted(record.items()):
                if v is not None:
                    lines.append(f"{k} = {_toml_value(v)}")
        if init_kwargs:
            lines += ["", "# Arguments de construction du modèle.", "[construction]"]
            for k, v in sorted(init_kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    lines.append(f"{k} = {_toml_value(v)}")
        var = recette.get("variables", {})
        lines += ["", "# Variables d'environnement du projet effectivement posées.",
                   "# Une variable absente signifie que le défaut du code s'applique.",
                   "[recipe]"]
        for k, v in var.items():
            lines.append(f"{k} = {_toml_value(v)}")
        if not var:
            lines.append("# aucune : l'exécution tournait entièrement sur les défauts du code")
        setting_values = recette.get("settings") or {}
        if setting_values:
            lines += ["", "# Réglages du paquet qui s'écartent de leur défaut (meandre/utils/settings.py).", "[settings]"]
            for k, v in sorted(setting_values.items()):
                lines.append(f'"{k}" = {_toml_value(v)}')
        for k, src in sorted((recette.get("domain") or {}).items()):
            lines += ["", f"[domain.{k}]"]
            for kk, vv in src.items():
                lines.append(f"{kk} = {_toml_value(vv)}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p
    except Exception:
        return None


def apply_recipe(section: dict | None) -> list[str]:
    """Pose une recette lue en TOML dans l'environnement, SANS jamais écraser.

    C'est la brique qui rend les réglages exprimables en fichier. Le pilote lit
    aujourd'hui sa physique dans l'environnement, en soixante-dix endroits dispersés ;
    réécrire ces soixante-dix lectures serait long et risqué. Poser les valeurs du
    fichier par `setdefault` avant que le pilote ne commence donne le même résultat en
    une ligne, et conserve l'ordre de priorité attendu :

        1. la variable d'environnement, si elle est posée (un lancement ponctuel prime) ;
        2. la valeur du fichier de recette ;
        3. le défaut du code.

    Un script de grappe existant, qui pose son bloc de variables, continue donc de se
    comporter exactement comme avant. Retourne les clés effectivement appliquées.
    """
    if not section:
        return []
    posees = []
    for key, value in section.items():
        field_name = str(key)
        if not field_name.startswith(PREFIXES):
            raise ValueError(
                f"recette : '{field_name}' ne porte aucun prefixe du projet {PREFIXES}. "
                f"Une cle sans prefixe ne serait lue par personne.")
        if field_name in os.environ:
            continue
        if isinstance(value, bool):
            text_value = "1" if value else "0"
        else:
            text_value = str(value)
        os.environ[field_name] = text_value
        posees.append(field_name)
    return sorted(posees)


def compare_recipe(saved: dict | None, ignore: tuple[str, ...] = ()) -> list[str]:
    """Écarts entre la recette sauvegardée et l'environnement courant.

    Retourne une liste de phrases lisibles, vide si tout concorde. Les chemins de
    déploiement sont ignorés par défaut : ils changent légitimement d'une machine à
    l'autre et ne modifient pas la physique.
    """
    if not saved or not isinstance(saved, dict):
        return []
    neutral = ("MEANDRE_DATA", "MEANDRE_PLATEFORMES", "MEANDRE_PLATFORMS",
               "MEANDRE_RQH", "ETL_TAG", "ETL_DUMP_Q", "ETL_DUMP_REACH",
               "ETL_WARM_FROM", "ETL_MELT_DIR", "ETL_SOIL_DIR") + tuple(ignore)
    old = {k: v for k, v in saved.get("variables", {}).items() if k not in neutral}
    current_value = {n: v for n, v in os.environ.items()
               if n.startswith(PREFIXES) and n not in neutral and not _est_secret(n)}
    gaps = []
    from meandre.domain import manifest as _manifest
    _m = _manifest.current()
    if _m is not None:
        gaps += _m.differences(saved.get("domain"))
    for k in sorted(set(old) | set(current_value)):
        a, c = old.get(k), current_value.get(k)
        if a == c:
            continue
        if a is None:
            gaps.append(f"{k} posee a {c!r} maintenant, absente a l'entrainement")
        elif c is None:
            gaps.append(f"{k} valait {a!r} a l'entrainement, absente maintenant")
        else:
            gaps.append(f"{k} valait {a!r} a l'entrainement, vaut {c!r} maintenant")
    return gaps
