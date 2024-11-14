# ------------------------------------------------------------------------------
# File: sbert_util.py
# Description: a simple loader for sbert models
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import os

from sentence_transformers import SentenceTransformer
from torch.cuda import is_available


def get_sbert(trust_remote_code: bool = True):
    device = "cuda" if is_available() else "cpu"

    embedding_model = os.getenv("EMBEDDING_OUTPUT")
    if len(embedding_model) == 0:
        print("No model specified. Loading default model")
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(
        embedding_model,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    return model
