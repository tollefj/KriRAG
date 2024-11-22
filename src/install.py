import nltk
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# https://huggingface.co/intfloat/multilingual-e5-base/discussions/24
pr_number = 24
model = SentenceTransformer(
    EMBEDDING_MODEL,
    revision=f"refs/pr/{pr_number}",
    backend="openvino",  # we optimize cpu-inference to reduce docker container image (w/ cuda drivers etc.)
)

del model

nltk.download("punkt")
nltk.download("punkt_tab")

