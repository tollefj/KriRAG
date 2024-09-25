# KriRAG

Code for the paper titled:
*Enhancing Criminal Investigation Analysis with Summarization and Memory-based Retrieval-Augmented Generation: A Comprehensive Evaluation of Real Case Data*

Results from manual evaluation are found in the [experiments](experiments/) directory.

**Note: This study discusses case files that contain unsettling information and language pertaining to violent crimes.**

![KriRAG UI](assets/1.png)
![KriRAG UI and config](assets/2.png)
![KriRAG UI and output](assets/3.png)
![KriRAG UI and output](assets/4.png)


## Installation

First, download a suitable torch version for your system. Any >2 version should suffice.

CPU only:

`pip install torch --index-url https://download.pytorch.org/whl/cpu`

With CUDA:

`pip install torch`

```bash
make install
make download  # places cache and models in the same directory, allowing for easy offline usage.
```

You need to download an LLM of choice and run it using the llama.cpp library, or with any other library supporting the same open-ai like API.
The script for running the Gemma-27B model used in this work is located in `scripts/LLM_GEMMAQ5_27.sh`.

## Running

The simplest way of running the system is through the UI using [streamlit](https://streamlit.io/).

```bash
streamlit run ui.py
```

Otherwise, feel free to inspect the source code in `src`.

### Environment variables

Any program utilizing huggingface models should use `load_dotenv()` for the correct environment variables.
See the `utils.generic` module for info, allowing overriding variables from custom env files.

```python
from utils.generic import init_dotenv
init_dotenv(custom_environments=".my-env-file")
# your program
```

The default env variables are located in the `.env` file:

```YAML
HF_HOME=models
NLTK_DATA=models
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_OUTPUT=models/EMBEDDING_MODEL
path_llm=models/llm.gguf
```
