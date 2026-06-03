ENV_FILE=.env

DATA_BASE_DIR=~/Documents/GitHub/data
DATA_ROOT_FOLDER=${DATA_ROOT}/hakken_bio/v2

KGE_RUN_PATH=~/Documents/GitHub/project_spaice_ds/packages/pip/kge/experiments/digital_science/test

CACHED_DATA_FOLDER=${KGE_RUN_PATH}/cached_kg
CONFIG_PATH="${PWD}/config"

UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME=name.surname@sony.com
UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD=YOUR_PASSWORD

# Create file if it does not exist and add a dummy line
if [ ! -f "$ENV_FILE" ]; then
    touch $ENV_FILE
    echo "DATA_BASE_DIR=${DATA_BASE_DIR}" >> $ENV_FILE
    echo "DATA_ROOT_FOLDER=${DATA_ROOT_FOLDER}" >> $ENV_FILE
    echo "KGE_RUN_PATH=${KGE_RUN_PATH}" >> $ENV_FILE
    echo "CACHED_DATA_FOLDER=${CACHED_DATA_FOLDER}" >> $ENV_FILE
    echo "CONFIG_PATH=${CONFIG_PATH}" >> $ENV_FILE
    echo "UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME=${UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME}" >> $ENV_FILE
    echo "UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD=${UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD}" >> $ENV_FILE
fi


