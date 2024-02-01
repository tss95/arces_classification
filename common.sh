# DEFINE THESE
USER=$(whoami)
PROJECT_PATH=$ROOT_DIR
DATA_PATH=$STORAGE_DIR/data/loaded
CONFIG_PATH=$ROOT_DIR/config
CONFIG_MAIN=$CONFIG_PATH/data_config.yaml
PROJECTNAME=arces_classification
SCRIPT_LOCATION=$PROJECT_PATH
REQUIREMENTS=$PROJECT_PATH/requirements.txt

GLOBAL_CONFIG=$PROJECT_PATH/global_config.py
PROJECT_SETUP=$PROJECT_PATH/project_setup.py
LOGGER=$CONFIG_PATH/logging_config.py

OUTPUT=$STORAGE_DIR/output

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
BASE_DIR=/nobackup2/$USER/$PROJECTNAME
mkdir_if_not_exist $BASE_DIR/data
mkdir_if_not_exist $BASE_DIR/output
mkdir_if_not_exist $BASE_DIR/logs
mkdir_if_not_exist $BASE_DIR/src
mkdir_if_not_exist $BASE_DIR/config
mkdir_if_not_exist $BASE_DIR/config/models
mkdir_if_not_exist $OUTPUT/plots
mkdir_if_not_exist $OUTPUT/models
mkdir_if_not_exist $OUTPUT/predictions
mkdir_if_not_exist $OUTPUT/logs
mkdir_if_not_exist $BASE_DIR/data/maps

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
rsync -ahr $PROJECT_PATH/data/maps/* $BASE_DIR/data/maps/
rsync -ahr $LOGGER $BASE_DIR/config/logging_config.py
rsync -ahr $CONFIG_MAIN $BASE_DIR/config/data_config.yaml
cp -r $PROJECT_PATH/src $BASE_DIR

# Syncing configs and requirements
rsync -ahr $REQUIREMENTS $BASE_DIR/requirements.txt
cp $PROJECT_PATH/docker.dockerfile $BASE_DIR/docker.dockerfile
cp -v $PROJECT_PATH/.dockerignore $BASE_DIR/.dockerignore
cp $PROJECT_PATH/.docker_bashrc $BASE_DIR/.docker_bashrc

# Paths
SOURCE_OUTPUT_DIR=$STORAGE_DIR/output
TARGET_OUTPUT_DIR=$BASE_DIR/output

# Sync Source to Target before Docker run
echo "Syncing files from $OUTPUT to $TARGET_OUTPUT_DIR"
sync_directories "$OUTPUT" "$TARGET_OUTPUT_DIR"
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
  docker build -t $PROJECTNAME:latest-gpu -f $BASE_DIR/docker.dockerfile .
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
docker run -e TF_CPP_MIN_LOG_LEVEL=3 -it --ipc=host --rm --gpus ${GPU_DEVICE} -v $BASE_DIR:/tf $PROJECTNAME:latest-gpu bash -c "$WANDB_EXPORT
                                                                                                                                source /root/.bashrc &&
                                                                                                                                find /tf/data -name 'Thumbs.db' -type f -delete &&
                                                                                                                                export TF_CPP_MIN_LOG_LEVEL=3 &&
                                                                                                                                python /tf/run_script.py &&
                                                                                                                                chmod -R 777 /tf/output/* &&
                                                                                                                                bash"


# Extract model name using Python script
model_name=$(python3 get_model_name.py $CONFIG_MAIN)

echo "Syncing files from $TARGET_OUTPUT_DIR to $SOURCE_OUTPUT_DIR"

sync_directories "$TARGET_OUTPUT_DIR" "$SOURCE_OUTPUT_DIR"
echo "Output synced"

find "$SOURCE_OUTPUT_DIR/$model_name" -type d -empty -delete
echo "Deleted empty directories in $SOURCE_OUTPUT_DIR/$model_name"
