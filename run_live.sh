#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_NAME="live_via_ml_array.py"
GPU_ID="${GPU_ID:-0}"
COMMON_ARGS=()
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --gpu requires a GPU id (e.g. 0, 1, all)." >&2
        exit 1
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --gpu=*)
      GPU_ID="${1#*=}"
      shift
      ;;
    -b|--build)
      COMMON_ARGS+=("--build")
      shift
      ;;
    --)
      shift
      SCRIPT_ARGS+=("$@")
      break
      ;;
    *)
      SCRIPT_ARGS+=("$1")
      shift
      ;;
  esac
done

export PROJECT_DIR
export SCRIPT_NAME
export GPU_DEVICE="$GPU_ID"

echo "Running live script on GPU: $GPU_ID"
"$PROJECT_DIR/common.sh" "${COMMON_ARGS[@]}" --predict -- "${SCRIPT_ARGS[@]}"
