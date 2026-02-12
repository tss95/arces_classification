#!/bin/bash
# Default training entrypoint
export SCRIPT_NAME=code_test.py
export DATA_DIR="/nobackup2/tord/arces_classification_pytorch"
export PROJECT_DIR="/staff/tord/Workspace/arces_classification"
export BASE_DIR="/nobackup2/tord/arces_classification_pytorch"
# Allow overriding the model config via -m/--model or $MODEL_CONFIG
DEFAULT_MODEL_CONFIG="alexnet.yaml"
MODEL_CONFIG="${MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
GPU_ID="${GPU_ID:-1}"
DETERMINISTIC_OVERRIDE="${DETERMINISTIC_OVERRIDE:-}"
PASSTHROUGH_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -m|--model)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --model requires a model yaml name (e.g., cnn_dense.yaml or s4)." >&2
        exit 1
      fi
      MODEL_CONFIG="$2"
      shift 2
      ;;
    -g|--gpu)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --gpu requires a GPU id (0 or 1)." >&2
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
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$GPU_ID" =~ ^[01]$ ]]; then
  echo "ERROR: --gpu must be 0 or 1 (got: $GPU_ID)." >&2
  exit 1
fi

# Normalize to yaml filename and basename
MODEL_CONFIG="${MODEL_CONFIG%.yaml}.yaml"
MODEL_NAME="$(basename "${MODEL_CONFIG%.yaml}")"

# Default to online W&B when a key is present; allow offline fallback otherwise
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi

# Ensure PROJECT_DIR is set before touching configs
if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "ERROR: PROJECT_DIR is not set. export PROJECT_DIR=/path/to/repo" >&2
  exit 1
fi

MODEL_CONFIG_PATH="$PROJECT_DIR/config/models/$MODEL_CONFIG"
if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
  echo "ERROR: Model config not found: $MODEL_CONFIG_PATH" >&2
  exit 1
fi

# Keep MODEL_CONFIG available to the container
export MODEL_CONFIG

# Update the base data_config.yaml to point at the requested model
CONFIG_MAIN="$PROJECT_DIR/config/data_config.yaml"
if [[ -f "$CONFIG_MAIN" ]]; then
  sed -i "s/^model_name:.*/model_name: \"$MODEL_NAME\"/" "$CONFIG_MAIN"
else
  echo "ERROR: data_config.yaml not found at $CONFIG_MAIN" >&2
  exit 1
fi

echo "Running train script with model: $MODEL_NAME ($MODEL_CONFIG) on GPU: $GPU_ID"
export GPU_DEVICE="device=${GPU_ID}"
if [[ -n "$DETERMINISTIC_OVERRIDE" ]]; then
  export DETERMINISTIC_OVERRIDE
  echo "Deterministic override: ${DETERMINISTIC_OVERRIDE}"
fi
source common.sh "${PASSTHROUGH_ARGS[@]}"
