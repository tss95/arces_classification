#!/usr/bin/env bash
set -euo pipefail

# Generic local Docker launcher for this repo (no rsync staging).
#
# Required env:
#   SCRIPT_NAME   script in repo root to execute (e.g. train.py, live_via_ml_array.py)
#
# Optional env:
#   PROJECT_DIR            host repo path (default: directory of this script)
#   DATA_DIR               host data root path used by config data_paths (default: PROJECT_DIR, with legacy fallback)
#   IMAGE_NAME             docker image tag (default: arces_classification_pytorch:latest)
#   DOCKER_USER            uid:gid in container (default: current user)
#   MODEL_CONFIG           model yaml name/path override for training entrypoints
#   INFERENCE_REPO_DIR     host path to ml_array_data_classification repo; mounted to /inference_repo when present
#   WANDB_*                forwarded if set
#   DETERMINISTIC_OVERRIDE forwarded if set
#   PREDICT_MODE           set internally from --predict and forwarded to runtime config layer
#
# Args:
#   -b|--build          force image rebuild
#   -g|--gpu <id|all>   GPU selector for docker --gpus
#   --predict           set runtime predict mode for this run
#   --                  all following args are passed to the python entry script

if [[ -z "${SCRIPT_NAME:-}" ]]; then
  echo "ERROR: SCRIPT_NAME is not set." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

if [[ ! -f "$PROJECT_DIR/$SCRIPT_NAME" ]]; then
  echo "ERROR: Script not found: $PROJECT_DIR/$SCRIPT_NAME" >&2
  exit 1
fi

LEGACY_DATA_DIR="/nobackup2/$(whoami)/arces_classification_pytorch"
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "$PROJECT_DIR/loaded_classifier_nofilt" ]]; then
    DATA_DIR="$PROJECT_DIR"
  elif [[ -d "$LEGACY_DATA_DIR/loaded_classifier_nofilt" ]]; then
    DATA_DIR="$LEGACY_DATA_DIR"
  else
    DATA_DIR="$PROJECT_DIR"
  fi
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-arces_classification_pytorch:latest}"
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"
GPU_DEVICE="${GPU_DEVICE:-all}"
FORCE_BUILD=0
PREDICT_MODE="False"
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build)
      FORCE_BUILD=1
      shift
      ;;
    -g|--gpu)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --gpu requires an argument (e.g. 0, 1, all)." >&2
        exit 1
      fi
      GPU_DEVICE="$2"
      shift 2
      ;;
    --gpu=*)
      GPU_DEVICE="${1#*=}"
      shift
      ;;
    --predict)
      PREDICT_MODE="True"
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

if [[ "$FORCE_BUILD" -eq 1 ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Building docker image: $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" -f "$PROJECT_DIR/docker.dockerfile" "$PROJECT_DIR"
fi

CONTAINER_PROJECT_DIR="/workspace"
CONTAINER_DATA_DIR="/workspace_data"
CONTAINER_WANDB_DIR="/workspace/wandb"
CONTAINER_MPLCONFIGDIR="/workspace/.cache/matplotlib"

MOUNT_ARGS=(
  -v "$PROJECT_DIR:$CONTAINER_PROJECT_DIR"
)

if [[ "$DATA_DIR" == "$PROJECT_DIR" ]]; then
  CONTAINER_DATA_DIR="$CONTAINER_PROJECT_DIR"
else
  MOUNT_ARGS+=(-v "$DATA_DIR:$CONTAINER_DATA_DIR")
fi

if [[ -z "${INFERENCE_REPO_DIR:-}" ]]; then
  SIBLING_REPO="$(dirname "$PROJECT_DIR")/ml_array_data_classification"
  if [[ -d "$SIBLING_REPO" ]]; then
    INFERENCE_REPO_DIR="$SIBLING_REPO"
  fi
fi

if [[ -n "${INFERENCE_REPO_DIR:-}" && -d "${INFERENCE_REPO_DIR:-}" ]]; then
  MOUNT_ARGS+=(-v "$INFERENCE_REPO_DIR:/inference_repo")
  export INFERENCE_REPO_DIR="/inference_repo"
fi

mkdir -p "$PROJECT_DIR/wandb" "$PROJECT_DIR/.cache/matplotlib"

echo "Launching $SCRIPT_NAME in Docker"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "GPU_DEVICE=$GPU_DEVICE"
echo "IMAGE_NAME=$IMAGE_NAME"

DOCKER_TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_FLAGS=(-it)
fi

ENV_ARGS=(
  -e PROJECT_DIR="$CONTAINER_PROJECT_DIR"
  -e DATA_DIR="$CONTAINER_DATA_DIR"
  -e WANDB_DIR="$CONTAINER_WANDB_DIR"
  -e MPLCONFIGDIR="$CONTAINER_MPLCONFIGDIR"
)

OPTIONAL_ENV_VARS=(
  WANDB_API_KEY
  WANDB_MODE
  WANDB_ENTITY
  WANDB_PROJECT
  DETERMINISTIC_OVERRIDE
  INFERENCE_REPO_DIR
  MODEL_CONFIG
)

for var_name in "${OPTIONAL_ENV_VARS[@]}"; do
  if [[ -n "${!var_name:-}" ]]; then
    ENV_ARGS+=(-e "$var_name=${!var_name}")
  fi
done
ENV_ARGS+=(-e "PREDICT_MODE=${PREDICT_MODE}")

docker run --rm \
  --ipc=host \
  --gpus="$GPU_DEVICE" \
  -u "$DOCKER_USER" \
  "${ENV_ARGS[@]}" \
  "${DOCKER_TTY_FLAGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  -w "$CONTAINER_PROJECT_DIR" \
  "$IMAGE_NAME" \
  python "$CONTAINER_PROJECT_DIR/$SCRIPT_NAME" "${SCRIPT_ARGS[@]}"
