#!/bin/bash

DEFAULT_MODEL="llms/gemma-2-9b-it-Q5_K_M.gguf"
# DEFAULT_MODEL="llms/qwen2.5-1.5b-instruct-q5_k_m.gguf"

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
docker load -i "$IMAGE_FILE" || {
  echo "Docker load failed!"
  exit 1
}

OSTYPE=$(uname -s)
if [ "$OSTYPE" == "Darwin" ]; then
  HOST_IP=$(ipconfig getifaddr en0 2>/dev/null)
elif [ "$OSTYPE" == "Linux" ]; then
  HOST_IP=$(hostname -I | awk '{print $1}')
else
  HOST_IP="localhost"
fi

PORT=8000
CTX_LEN=4096
GPU_LAYERS=100

echo "server is running on $HOST_IP:$PORT"
echo "ssh: ssh <user>@$HOST_IP"
echo "web: http://$HOST_IP:$PORT"

docker run -v "$MODELS_DIR:/models" \
  -p "$PORT:$PORT" \
  --gpus all \
  local/llama.cpp:server-cuda \
  -m "~/LLM_STORE\gemma-2-9b-it-Q5_K_M.gguf" \
  --port $PORT -n $CTX_LEN -ngl $GPU_LAYERS
