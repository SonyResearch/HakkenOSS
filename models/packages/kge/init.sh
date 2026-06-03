ENV_FILE=.env

DATA_BASE_DIR=~/Documents/GitHub/data/
DATA_ROOT_FOLDER=~/Documents/GitHub/data/hakken_bio/v2


CONFIG_PATH="${PWD}/config"

TRAINED_MODEL_PATH="${PWD}/outputs/production/kge/complex/v1/"

# Create file if it does not exist and add a dummy line
if [ ! -f "$ENV_FILE" ]; then
    touch $ENV_FILE
    echo "DATA_BASE_DIR=${DATA_BASE_DIR}" >> $ENV_FILE
    echo "DATA_ROOT_FOLDER=${DATA_ROOT_FOLDER}" >> $ENV_FILE
    echo "CONFIG_PATH=${CONFIG_PATH}" >> $ENV_FILE
    echo "TRAINED_MODEL_PATH=${TRAINED_MODEL_PATH}" >> $ENV_FILE
fi

if [ ! -d "$TRAINED_MODEL_PATH" ]; then
    mkdir -p $TRAINED_MODEL_PATH
fi
