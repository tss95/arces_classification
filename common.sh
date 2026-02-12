# DEFINE THESE
USER=$(whoami)

# Validate required environment variables
if [ -z "$PROJECT_DIR" ]; then
  echo "ERROR: PROJECT_DIR is not set. export PROJECT_DIR=/path/to/repo" >&2
  exit 1
fi
if [ -z "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR is not set. export DATA_DIR=/path/to/data" >&2
  exit 1
fi

PROJECT_PATH="$PROJECT_DIR"
CONFIG_PATH="$PROJECT_DIR/config"
SATURN_OUTPUT_DIR=$PROJECT_PATH/output
CONFIG_MAIN="$CONFIG_PATH/data_config.yaml"
PROJECTNAME="arces_classification_pytorch"
SCRIPT_LOCATION="$PROJECT_PATH"
REQUIREMENTS="$PROJECT_PATH/requirements.txt"
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"
CONTAINER_PROJECT_DIR="${CONTAINER_PROJECT_DIR:-/tf}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/tf/data}"
CONTAINER_WANDB_DIR="${CONTAINER_WANDB_DIR:-/tf/wandb}"
CONTAINER_MPLCONFIGDIR="${CONTAINER_MPLCONFIGDIR:-/tf/.cache/matplotlib}"

GLOBAL_CONFIG="$PROJECT_PATH/global_config.py"
PROJECT_SETUP="$PROJECT_PATH/project_setup.py"
LOGGER="$CONFIG_PATH/logging_config.py"


# Create directory if it doesn't exist
mkdir_if_not_exist() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
  fi
}

# Custom Sync Function
sync_directories() {
    local source_dir=$1
    local target_dir=$2
    # Mirroring directory structure and syncing files
    rsync -av --delete --progress "$source_dir/" "$target_dir/"
}

# LOGIC (DO NOT CHANGE)
# Allow override of BASE_DIR; default to /nobackup2/$USER/$PROJECTNAME
BASE_DIR=${BASE_DIR:-/nobackup2/$USER/$PROJECTNAME}
mkdir_if_not_exist "$BASE_DIR/data"
mkdir_if_not_exist "$BASE_DIR/data/data"
mkdir_if_not_exist "$BASE_DIR/data/metadata"
mkdir_if_not_exist "$BASE_DIR/data/loaded_classifier"
mkdir_if_not_exist "$BASE_DIR/data/loaded_classifier_nofilt"
mkdir_if_not_exist "$BASE_DIR/output"
mkdir_if_not_exist "$BASE_DIR/logs"
mkdir_if_not_exist "$BASE_DIR/src"
mkdir_if_not_exist "$BASE_DIR/inference"
mkdir_if_not_exist "$BASE_DIR/config"
mkdir_if_not_exist "$BASE_DIR/config/models"
mkdir_if_not_exist "$BASE_DIR/output/plots"
mkdir_if_not_exist "$BASE_DIR/output/models"
mkdir_if_not_exist "$BASE_DIR/output/predictions"
mkdir_if_not_exist "$BASE_DIR/output/logs"
mkdir_if_not_exist "$BASE_DIR/wandb"
mkdir_if_not_exist "$BASE_DIR/.cache"
mkdir_if_not_exist "$BASE_DIR/.cache/matplotlib"

# Default mode is not predict
PREDICT_MODE="False"
# Initialize variable to indicate whether to force build
FORCE_BUILD="False"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--predict) PREDICT_MODE="True" ;;
        -b) FORCE_BUILD="True" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Rsync of data directory (support both flat and nested layouts)
DATA_SRC="$DATA_DIR/data"
METADATA_SRC="$DATA_DIR/metadata"
LOADED_SRC="$DATA_DIR/loaded_classifier"
LOADED_SRC_NOFILT="$DATA_DIR/loaded_classifier_nofilt"
SYNC_INFERENCE_REPO="${SYNC_INFERENCE_REPO:-1}"
INFERENCE_REPO_DIR="${INFERENCE_REPO_DIR:-$(dirname "$PROJECT_DIR")/ml_array_data_classification}"
INFERENCE_TARGET_DIR="$BASE_DIR/inference/ml_array_data_classification"

# If metadata/loaded_classifier are nested under data/, fall back to that structure
if [ ! -d "$METADATA_SRC" ] && [ -d "$DATA_DIR/data/metadata" ]; then
  METADATA_SRC="$DATA_DIR/data/metadata"
fi
if [ ! -d "$LOADED_SRC" ] && [ -d "$DATA_DIR/data/loaded_classifier" ]; then
  LOADED_SRC="$DATA_DIR/data/loaded_classifier"
fi
if [ ! -d "$LOADED_SRC_NOFILT" ] && [ -d "$DATA_DIR/data/loaded_classifier_nofilt" ]; then
  LOADED_SRC_NOFILT="$DATA_DIR/data/loaded_classifier_nofilt"
fi

if [ -d "$DATA_SRC" ]; then
  rsync -ahr --include='eventclass_*' --exclude='*' "$DATA_SRC/" "$BASE_DIR/data/data/"
else
  echo "WARN: $DATA_SRC not found; skipping data sync"
fi
if [ -d "$METADATA_SRC" ]; then
  rsync -ahr --include='*snrupdate*' --include='*arrivals_update*' --exclude='*' "$METADATA_SRC/" "$BASE_DIR/data/metadata/"
else
  echo "WARN: $METADATA_SRC not found; skipping metadata sync"
fi
if [ -d "$LOADED_SRC" ]; then
  rsync -ahr "$LOADED_SRC/" "$BASE_DIR/data/loaded_classifier/"
else
  echo "WARN: $LOADED_SRC not found; skipping loaded_classifier sync"
fi
if [ -d "$LOADED_SRC_NOFILT" ]; then
  rsync -ahr "$LOADED_SRC_NOFILT/" "$BASE_DIR/data/loaded_classifier_nofilt/"
else
  echo "WARN: $LOADED_SRC_NOFILT not found; skipping loaded_classifier_nofilt sync"
fi

if [ "$SYNC_INFERENCE_REPO" = "1" ]; then
  if [ -d "$INFERENCE_REPO_DIR" ]; then
    mkdir_if_not_exist "$INFERENCE_TARGET_DIR"
    rsync -ahr --delete --exclude='.git/' "$INFERENCE_REPO_DIR/" "$INFERENCE_TARGET_DIR/"
    if command -v git >/dev/null 2>&1; then
      INFERENCE_COMMIT=$(git -C "$INFERENCE_REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
      echo "$INFERENCE_COMMIT" > "$INFERENCE_TARGET_DIR/.inference_commit"
    fi
  else
    echo "WARN: $INFERENCE_REPO_DIR not found; skipping inference repo sync"
  fi
else
  echo "Inference repo sync disabled (SYNC_INFERENCE_REPO=$SYNC_INFERENCE_REPO)"
fi


echo "Options: $@"
# Now use the PREDICT_MODE variable to alter script behavior
if [ "$PREDICT_MODE" = "True" ]; then
    echo "Only loading validation set"
    # Add your predict mode-specific commands here
else
    echo "Loading training and validation set"
    # Add your normal mode-specific commands here
fi

sed -i "s/predict: .*/predict: $PREDICT_MODE/" $CONFIG_MAIN

echo "Folders created"

cp $SCRIPT_LOCATION/$SCRIPT_NAME $BASE_DIR/run_script.py
rsync -ahr $GLOBAL_CONFIG $BASE_DIR/global_config.py
rsync -ahr $PROJECT_SETUP $BASE_DIR/project_setup.py
#rsync -ahr $PROJECT_PATH/data/maps/* $BASE_DIR/data/maps/
rsync -ahr $LOGGER $BASE_DIR/config/logging_config.py
rsync -ahr $CONFIG_MAIN $BASE_DIR/config/data_config.yaml
rsync -ahr $CONFIG_PATH/models/* $BASE_DIR/config/models/
cp -r $PROJECT_PATH/src $BASE_DIR                                                                                                                                                                                                                                                                                                                                                                   

# Syncing configs and requirements
rsync -ahr $REQUIREMENTS $BASE_DIR/requirements.txt
cp $PROJECT_PATH/docker.dockerfile $BASE_DIR/docker.dockerfile
cp -v $PROJECT_PATH/.dockerignore $BASE_DIR/.dockerignore
cp $PROJECT_PATH/.docker_bashrc $BASE_DIR/.docker_bashrc

# Paths
SOURCE_OUTPUT_DIR=$SATURN_OUTPUT_DIR
TARGET_OUTPUT_DIR=$BASE_DIR/output

# Sync Source to Target before Docker run
echo "Syncing files from $SOURCE_OUTPUT_DIR to $TARGET_OUTPUT_DIR"
sync_directories "$SOURCE_OUTPUT_DIR" "$TARGET_OUTPUT_DIR"
echo "Files synced"


# Compute the hash of the local Dockerfile and requirements.txt
HASH_ON_LOCAL_MACHINE=$(sha256sum $PROJECT_PATH/docker.dockerfile $REQUIREMENTS | awk '{ print $1 }')

# Compute the hash of the current Dockerfile and requirements.txt on the GPU machine
HASH_ON_GPU_MACHINE=$(sha256sum $BASE_DIR/docker.dockerfile $BASE_DIR/requirements.txt | awk '{ print $1 }')

echo "Hash computed."

# Rebuild the Docker image if the hashes are different
if [ "$HASH_ON_GPU_MACHINE" != "$HASH_ON_LOCAL_MACHINE" ] || [ "$FORCE_BUILD" = "True" ]; then
  cp $PROJECT_PATH/docker.dockerfile $BASE_DIR/docker.dockerfile
  cp $REQUIREMENTS $BASE_DIR/requirements.txt
  docker build -t $PROJECTNAME:latest -f $BASE_DIR/docker.dockerfile .
fi

# Check if WANDB_API_KEY is set and export it
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WANDB_API_KEY is not set. Continuing without it."
    WANDB_EXPORT=""
else
    echo "Exporting WANDB_API_KEY to Docker container."
    WANDB_EXPORT="export WANDB_API_KEY=$WANDB_API_KEY &&"
fi

echo "Debug: BASE_DIR=$BASE_DIR, PROJECTNAME=$PROJECTNAME"
docker run -e PROJECT_DIR=$CONTAINER_PROJECT_DIR -e DATA_DIR=$CONTAINER_DATA_DIR -e WANDB_API_KEY -e WANDB_MODE -e WANDB_ENTITY -e WANDB_PROJECT -e DETERMINISTIC_OVERRIDE -e WANDB_DIR=$CONTAINER_WANDB_DIR -e MPLCONFIGDIR=$CONTAINER_MPLCONFIGDIR -it --ipc=host --rm --gpus=${GPU_DEVICE} -u ${DOCKER_USER} -v $BASE_DIR:/tf -w $CONTAINER_PROJECT_DIR $PROJECTNAME:latest bash -c "$WANDB_EXPORT
                                                                                                     if [ -r /root/.bashrc ]; then source /root/.bashrc; fi &&
                                                                                                     find /tf/data -name 'Thumbs.db' -type f -delete &&
                                                                                                     python /tf/run_script.py &&
                                                                                                     chmod -R 777 /tf/* &&
                                                                                                     bash"


# Extract model name using Python script
model_name=$(python3 get_model_name.py $CONFIG_MAIN)

echo "Syncing files from $TARGET_OUTPUT_DIR to $SOURCE_OUTPUT_DIR"

sync_directories "$TARGET_OUTPUT_DIR" "$SOURCE_OUTPUT_DIR"
echo "Output synced"

#find "$SOURCE_OUTPUT_DIR/$model_name" -type d -empty -delete
#echo "Deleted empty directories in $SOURCE_OUTPUT_DIR/$model_name"
