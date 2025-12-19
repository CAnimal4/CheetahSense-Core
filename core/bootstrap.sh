#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Activate with: source .venv/bin/activate"
