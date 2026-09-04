#!/bin/bash
cd /mnt/c/Users/parse01/documents-locaux/GitHub/meandre
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
rm -f .runs/slso/_hydrotel_wsl_run.log
exec .venv-wsl/bin/python .runs/slso/slso.py .runs/slso/config/slso-physitel-hydrotel-wsl.toml > .runs/slso/_hydrotel_wsl_run.log 2>&1
