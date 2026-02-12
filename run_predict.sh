#!/usr/bin/env bash
set -euo pipefail

echo "run_predict.sh is deprecated (legacy predict.py removed). Delegating to run_live.sh."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_live.sh" "$@"
