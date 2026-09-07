#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_FILE="$UNIT_DIR/qc-scm-service-manager.service"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
TUNNEL_ENV=""
if [[ -n "${QC_SCM_TUNNEL_URL:-}" ]]; then
    TUNNEL_ENV="Environment=QC_SCM_TUNNEL_URL=${QC_SCM_TUNNEL_URL}"
fi
sed \
    -e "s|@PROJECT_ROOT@|${PROJECT_ROOT}|g" \
    -e "s|@PYTHON@|${PYTHON_BIN}|g" \
    -e "s|@TUNNEL_ENV@|${TUNNEL_ENV}|g" \
    "$PROJECT_ROOT/dashboard/service-manager.service.in" > "$UNIT_FILE"

systemctl --user daemon-reload
systemctl --user enable --now qc-scm-service-manager.service
printf 'Service Manager enabled: %s\n' "$UNIT_FILE"
