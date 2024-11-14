import os

import torch


def check_cuda_support():
    return torch.cuda.is_available()


image = (
    "ghcr.io/ggerganov/llama.cpp:server-cuda"
    if check_cuda_support()
    else "ghcr.io/ggerganov/llama.cpp:server"
)

print(f"Downloading image {image}")
os.system(f"docker pull {image}")
os.system(f"docker save -o llm-image.tar {image}")
