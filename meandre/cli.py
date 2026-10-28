"""Point d'entrée : `python -m meandre run <recette.toml> [options]`.

La recette définit le modèle ; le lancement ne pose que le territoire, les époques, l'étiquette et
le point de départ. Chaque option devient la variable d'environnement que lit le pilote ; une
variable déjà posée dans l'environnement garde la priorité, comme pour tout réglage déclaré.
"""
import argparse
import os
import runpy
import sys

# option du lancement -> variable lue par le pilote
LAUNCH_OPTIONS = {
    "region": "ETL_REGION",
    "epochs": "ETL_EPOCHS",
    "tag": "ETL_TAG",
    "warm_from": "ETL_WARM_FROM",
    "heldout": "ETL_HELDOUT",
    "dump_q": "ETL_DUMP_Q",
    "seed": "ETL_SEED",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="meandre", description="Lance meandre sur une recette.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="entraîner ou évaluer le modèle défini par une recette")
    run.add_argument("recipe", help="fichier TOML de la recette, par exemple .runs/quebec/recipe.toml")
    run.add_argument("--region", help="territoire, ou liste séparée par des virgules (domaine provincial)")
    run.add_argument("--epochs", help="nombre d'époques ; 0 pour une évaluation sans entraînement")
    run.add_argument("--tag", help="étiquette ajoutée au nom du point de reprise")
    run.add_argument("--warm-from", dest="warm_from", help="point de reprise de départ")
    run.add_argument("--heldout", help="période d'évaluation, « début,fin »")
    run.add_argument("--dump-q", dest="dump_q", help="fichier npz des débits aux stations")
    run.add_argument("--seed", help="graine")
    args = parser.parse_args(argv)
    if args.command == "run":
        os.environ.setdefault("ETL_CONFIG", args.recipe)
        for option, variable in LAUNCH_OPTIONS.items():
            value = getattr(args, option)
            if value is not None:
                os.environ.setdefault(variable, str(value))
        sys.argv = ["meandre.run.pilot"]
        runpy.run_module("meandre.run.pilot", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
