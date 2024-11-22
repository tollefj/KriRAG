#!/bin/bash
if [ ! -d ~/LLM_STORE ] || [ -z "$(ls -A ~/LLM_STORE/*.gguf 2>/dev/null)" ]; then
    echo "Error: ~/LLM_STORE does not exist or contains no .gguf LLM files."
    echo "Download and place a .gguf in ~/LLM_STORE and update the variable "MODEL_NAME" (in `run-krirag.sh`) accordingly."
    exit 1
else
    echo "Found .gguf files in ~/LLM_STORE:"
    ls -1 ~/LLM_STORE/*.gguf
fi

MODEL_NAME="gemma-2-9b-it-Q5_K_M"

if ! docker network ls | grep -q krirag-net; then
  docker network create krirag-net
fi

cleanup_container() {
    local CONTAINER_NAME=$1

    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping existing container: $CONTAINER_NAME..."
        docker stop $CONTAINER_NAME
    fi

    if docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        echo "Removing existing container: $CONTAINER_NAME..."
        docker rm $CONTAINER_NAME
    fi
}

API_NAME="krirag-api"
UI_NAME="krirag-ui"
MODEL_PATH="/models/$MODEL_NAME.gguf"
NGPU="100"   # Number of GPU layers, just max it at 100
N_CONTEXT_LEN="4096"
USER="toffdock"

docker pull $USER/krirag-api
docker pull $USER/krirag-ui

cleanup_container $API_NAME
docker run -d \
    --gpus all \
    --name $API_NAME \
    --network krirag-net \
    -p 8502:8502 \
    -v ~/LLM_STORE:/models \
    $USER/krirag-api \
    -m "$MODEL_PATH" \
    --port 8502 -n $N_CONTEXT_LEN -ngl $NGPU \

cleanup_container $UI_NAME
docker run \
    --gpus all \
    --name $UI_NAME \
    --network krirag-net \
    -p 8501:8501 \
    $USER/krirag-ui
