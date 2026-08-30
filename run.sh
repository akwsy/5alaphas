#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python -m src.train "$@"
