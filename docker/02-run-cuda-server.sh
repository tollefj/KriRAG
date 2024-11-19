#!/bin/bash
docker load < docker/server-cuda.tar

MODEL_PATH="/models/gemma-2-9b-it-Q5_K_M.gguf"
NGPU="100"   # Number of GPU layers, just max it at 100
N_CONTEXT_LEN="4096"

# Run the Llama.cpp server
docker run -v ~/LLM_STORE:/models \
  -p 8000:8000 \
  --gpus all \
  local/llama.cpp:server-cuda \
  -m "$MODEL_PATH" \
  --port 8000 -n $N_CONTEXT_LEN -ngl $NGPU 