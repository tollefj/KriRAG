#!/bin/bash
if ! python3 -c "import torch" &>/dev/null; then
    echo "Installing torch..."
    python3 -m pip install torch
fi
echo "Verifying CUDA support..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available. Using CUDA image..."
    image="ghcr.io/ggerganov/llama.cpp:server-cuda"
else
    echo "CUDA is not available. Using CPU-only image..."
    image="ghcr.io/ggerganov/llama.cpp:server"
fi

echo "Downloading image $image"
docker pull $image

docker save -o docker/llm-image.tar $image
