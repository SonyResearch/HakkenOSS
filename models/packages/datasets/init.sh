ENV_FILE=.env

DATA_PATH=TODO

# Create file if it does not exist and add a dummy line
if [ ! -f "$ENV_FILE" ]; then
    touch $ENV_FILE
    echo "DATA_PATH=${DATA_PATH}" >> $ENV_FILE

fi

