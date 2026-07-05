#!/bin/bash
# Idempotent environment bootstrap for whest-kprop-experiments.
# Ensures uv is installed and the project venv is synced. Safe to run on
# every session start: when everything is already in place it exits in ~1s.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "bootstrap: uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"
uv sync --group dev --quiet
echo "bootstrap: uv $(uv --version | awk '{print $2}') ready, venv synced at $REPO_DIR/.venv"
