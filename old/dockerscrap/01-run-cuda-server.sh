#!/bin/bash
docker run -v ~/LLM_STORE/:/models \
  -p 8000:8000 \
  --gpus all \
  local/llama.cpp:server-cuda \
  -m "models/gemma-2-9b-it-Q5_K_M.gguf" \
  --port 8000 -n 4096 -ngl 100
