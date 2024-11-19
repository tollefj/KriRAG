git clone git@github.com:ggerganov/llama.cpp.git
cd llama.cpp
docker build -t local/llama.cpp:server-cuda -f .devops/llama-server-cuda.Dockerfile .

docker run --gpus all -v llms:/llms local/llama.cpp:server-cuda -m /llms/gemma-2-9b-it-Q5_K_M.gguf --port 8000 --host 0.0.0.0 -n 512 --n-gpu-layers 100
# just test the image:
docker run --gpus all -v llms:/llms local/llama.cpp:server-cuda --help

docker run \
  --gpus all \
  -v ~/LLM_STORE:/llms local/llama.cpp:server-cuda \
  -m /llms/gemma-2-9b-it-Q5_K_M.gguf \
  --port 8181 \
  --host 0.0.0.0 \
  -n 512 \
  --n-gpu-layers 100 \

docker run --gpus all -v /llms:/llms -it local/llama.cpp:server-cuda /bin/bash
