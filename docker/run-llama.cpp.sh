#!/bin/bash

DEFAULT_MODEL="llms/gemma-2-9b-it-Q5_K_M.gguf"

# Check for model argument
if [ -z "$1" ]; then
  echo "No model specified. Using default model: $DEFAULT_MODEL"
  MODEL_FILE="$DEFAULT_MODEL"
else
  MODEL_FILE="$1"
fi

IMAGE_FILE="./docker/llm-image.tar"
MODELS_DIR="$(cd "$(dirname "$0")/../" && pwd)"

# Check if the Docker image file exists
if [ ! -f "$IMAGE_FILE" ]; then
  echo "Error: Docker image file $IMAGE_FILE does not exist."
  exit 1
fi

echo "Loading Docker image from $IMAGE_FILE..."
docker load -i "$IMAGE_FILE" || { echo "Docker load failed"; exit 1; }

OSTYPE=$(uname -s)
# Determine host IP based on OS
if [ "$OSTYPE" == "Darwin" ]; then
  HOST_IP=$(ipconfig getifaddr en0 2>/dev/null)
elif [ "$OSTYPE" == "Linux" ]; then
  HOST_IP=$(hostname -I | awk '{print $1}')
else
  echo "Unsupported OS: $OSTYPE"
  exit 1
fi

# Ensure HOST_IP is set
if [ -z "$HOST_IP" ]; then
  echo "Failed to retrieve host IP address."
  exit 1
fi

PORT=8000
CTX_LEN=4096
GPU_LAYERS=100

docker run -v "$MODELS_DIR:/models" -p 8000:8000 \
  ghcr.io/ggerganov/llama.cpp:server -m "/models/$MODEL_FILE" \
  --port $PORT --host "$HOST_IP" -n $CTX_LEN -ngl $GPU_LAYERS

echo "Server is running on $HOST_IP:$PORT"
echo "SSH: ssh <user>@$HOST_IP"
echo "IP: http://$HOST_IP:$PORT"
