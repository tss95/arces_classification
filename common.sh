# DEFINE THESE
USER=$(whoami)
PROJECT_PATH=/staff/tord/Workspace/arces_classification
DATA_PATH=$PROJECT_PATH/data/loaded
CONFIG_PATH=$PROJECT_PATH/config
CONFIG_MAIN=$CONFIG_PATH/data_config.yaml
CONFIG_MODEL=$CONFIG_PATH/model_config.yaml
PROJECTNAME=arces_classification
SCRIPT_LOCATION=$PROJECT_PATH
REQUIREMENTS=$PROJECT_PATH/requirements.txt

GLOBAL_CONFIG=$PROJECT_PATH/global_config.py
PROJECT_SETUP=$PROJECT_PATH/project_setup.py
LOGGER=$PROJECT_PATH/config/logging_config.py

DATA_LOCATION=$DATA_PATH
OUTPUT=$PROJECT_PATH/output

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
mkdir_if_not_exist $BASE_DIR/Classes
mkdir_if_not_exist $BASE_DIR/config
mkdir_if_not_exist $BASE_DIR/config/models
mkdir_if_not_exist $OUTPUT/plots
mkdir_if_not_exist $OUTPUT/models
mkdir_if_not_exist $OUTPUT/predictions
mkdir_if_not_exist $OUTPUT/logs
mkdir_if_not_exist $BASE_DIR/data/maps




echo "Folders created"

cp $SCRIPT_LOCATION/$SCRIPT_NAME $BASE_DIR/run_script.py
rsync -ahr $GLOBAL_CONFIG $BASE_DIR/global_config.py
rsync -ahr $PROJECT_SETUP $BASE_DIR/project_setup.py
rsync -ahr $PROJECT_PATH/data/maps/* $BASE_DIR/data/maps/
rsync -ahr $LOGGER $BASE_DIR/config/logging_config.py
rsync -agr $CONFIG_MAIN $BASE_DIR/config/data_config.yaml
rsync -ahr $PROJECT_PATH/config/models/$MODEL_CONFIG $BASE_DIR/config/models/$MODEL_CONFIG
cp -r $PROJECT_PATH/Classes $BASE_DIR
#rsync -ahr $DATA_PATH/* $BASE_DIR/data
rsync -ahr $CONFIG_MAIN $BASE_DIR/config/data_config.yaml
rsync -ahr $REQUIREMENTS $BASE_DIR/requirements.txt
cp $PROJECT_PATH/docker.dockerfile $BASE_DIR/docker.dockerfile
cp -v $PROJECT_PATH/.dockerignore $BASE_DIR/.dockerignore
cp $PROJECT_PATH/.docker_bashrc $BASE_DIR/.docker_bashrc

# Paths
SOURCE_OUTPUT_DIR=$PROJECT_PATH/output
TARGET_OUTPUT_DIR=$BASE_DIR/output

# Sync Source to Target before Docker run
echo "Syncing files from $SOURCE_OUTPUT_DIR to $TARGET_OUTPUT_DIR"
sync_directories "$SOURCE_OUTPUT_DIR" "$TARGET_OUTPUT_DIR"

echo "Files synced"

# Initialize variable to indicate whether to force build
FORCE_BUILD=false

# Parse command-line options
while getopts "b" OPTION; do
  case $OPTION in
    b)
      FORCE_BUILD=true
      ;;
    *)
      echo "Usage: $0 [-b]"
      exit 1
      ;;
  esac
done

# Compute the hash of the local Dockerfile and requirements.txt
HASH_ON_LOCAL_MACHINE=$(sha256sum $PROJECT_PATH/docker.dockerfile $REQUIREMENTS | awk '{ print $1 }')

# Compute the hash of the current Dockerfile and requirements.txt on the GPU machine
HASH_ON_GPU_MACHINE=$(sha256sum $BASE_DIR/docker.dockerfile $BASE_DIR/requirements.txt | awk '{ print $1 }')

echo "Hash computed."

# Rebuild the Docker image if the hashes are different
if [ "$HASH_ON_GPU_MACHINE" != "$HASH_ON_LOCAL_MACHINE" ] || [ "$FORCE_BUILD" = true ]; then
  cp $PROJECT_PATH/docker.dockerfile $BASE_DIR/docker.dockerfile
  cp $REQUIREMENTS $BASE_DIR/requirements.txt
  docker build -t $PROJECTNAME:latest-gpu -f $BASE_DIR/docker.dockerfile .
fi


wandb_api_key=aeb7c08e886c823c11e185436e5391546de850d0
# '"device=0"' uses only one, '"device=0,1"' will use both
echo "Debug: BASE_DIR=$BASE_DIR, PROJECTNAME=$PROJECTNAME"
docker run -e TF_CPP_MIN_LOG_LEVEL=3 -it --ipc=host --rm --gpus ${GPU_DEVICE} -v $BASE_DIR:/tf $PROJECTNAME:latest-gpu bash -c "export WANDB_API_KEY=$wandb_api_key &&
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
