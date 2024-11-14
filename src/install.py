# ------------------------------------------------------------------------------
# File: install.py
# Description: installation script for KriRAG to support offline usage
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import nltk
from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name

from utils.generic import init_dotenv

embedding_model = os.getenv("EMBEDDING_MODEL")

ignore_patterns = ["*.msgpack", "*.safetensors", "*.onnx", "onnx/*", "*.h5"]
only_safetensors = ["*.msgpack", "*.bin", "*.onnx", "onnx/*", "*.h5"]

known_safetensor_huggingface = ["intfloat", "google-bert"]


# modified from
# https://github.com/huggingface/huggingface_hub/issues/1240
def download_model(
    repo_id,
    save_path,
    output_dir="./src/models",
    ignore_patterns=ignore_patterns,
):
    print(f"Attempting to download {repo_id}")
    destination = Path(output_dir) / save_path
    if destination.exists():
        print(f"Model already exists at {destination}")
        return
    print(f"Output destination: {destination}")

    # Download and copy without symlinks
    tmp_ignore = deepcopy(ignore_patterns)
    for user in known_safetensor_huggingface:
        if user in repo_id:
            tmp_ignore = deepcopy(only_safetensors)
            print(f"Using safetensors for {repo_id}")
    downloaded = snapshot_download(
        repo_id,
        ignore_patterns=tmp_ignore,
        cache_dir=output_dir,
    )
    shutil.copytree(downloaded, destination)
    # Remove all downloaded files
    cache_folder = Path(output_dir) / repo_folder_name(
        repo_id=repo_id, repo_type="model"
    )
    shutil.rmtree(cache_folder)
    return destination


def install(custom_environments: List[str] = []):
    # environments are paths to .env files with custom overridden variables
    init_dotenv(custom_environments)

    print("Downloading nltk libs...")
    nltk.download("punkt", download_dir="src/models")
    nltk.download("punkt_tab", download_dir="src/models")

    for env_path in ["EMBEDDING_MODEL"]:
        print(f"Downloading {env_path}...")
        try:
            download_model(
                repo_id=os.getenv(env_path),
                save_path=env_path,
                ignore_patterns=ignore_patterns,
            )
        except Exception as e:
            print(f"Cancelled: {e}")


if __name__ == "__main__":
    install(sys.argv[1:])  # allow user to input any number of custom environment paths
