#!/bin/bash
git clone git@github.com:ggerganov/llama.cpp.git
cd llama.cpp
docker build -t krirag-api -f .devops/llama-server-cuda.Dockerfile .
# cd ..
# docker save local/llama.cpp:server-cuda > docker/server-cuda.tar