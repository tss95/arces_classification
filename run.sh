#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_NAME="train.py"
MODEL_CONFIG="${MODEL_CONFIG:-alexnet.yaml}"
GPU_ID="${GPU_ID:-0}"
DETERMINISTIC_OVERRIDE="${DETERMINISTIC_OVERRIDE:-}"
COMMON_ARGS=()
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --model requires a model yaml name." >&2
        exit 1
      fi
      MODEL_CONFIG="$2"
      shift 2
      ;;
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
    --deterministic)
      DETERMINISTIC_OVERRIDE="true"
      shift
      ;;
    --non-deterministic)
      DETERMINISTIC_OVERRIDE="false"
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

MODEL_CONFIG="${MODEL_CONFIG%.yaml}.yaml"
MODEL_NAME="$(basename "${MODEL_CONFIG%.yaml}")"
MODEL_CONFIG_PATH="$PROJECT_DIR/config/models/$MODEL_CONFIG"
if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
  echo "ERROR: Model config not found: $MODEL_CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi

export PROJECT_DIR
export SCRIPT_NAME
export GPU_DEVICE="$GPU_ID"
export MODEL_CONFIG
if [[ -n "$DETERMINISTIC_OVERRIDE" ]]; then
  export DETERMINISTIC_OVERRIDE
fi

echo "Running train script with model: $MODEL_NAME ($MODEL_CONFIG) on GPU: $GPU_ID"
"$PROJECT_DIR/common.sh" "${COMMON_ARGS[@]}" -- "${SCRIPT_ARGS[@]}"
