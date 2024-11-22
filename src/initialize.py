import os
import zipfile
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import streamlit as st
from chromadb import PersistentClient
from chromadb.types import Collection
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer

from utils.chroma import get_client

EMBEDDING_MODEL = "sbert"
LANG = "english"

valid_exts = [".txt", ".json", ".jsonl"]

with st.spinner("Loading SentenceTransformer model..."):
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        backend="openvino",  # we optimize cpu-inference to reduce docker container image (w/ cuda drivers etc.)
        local_files_only=True,
    )


def load_documents(
    doc_path: str,
    lang: str = LANG,
) -> pd.DataFrame:
    # iterate all files in the folder
    doc_paths = []
    for _file in os.listdir(doc_path):
        if _file.endswith(tuple(valid_exts)):
            doc_paths.append(os.path.join(doc_path, _file))

    parsed_data = []
    for dp in doc_paths:
        with open(dp, "r", encoding="utf-8") as f:
            _doc = f.readlines()
            parsed_data.extend(parse_document(_doc, lang))

    df = pd.DataFrame(parsed_data)
    return df


def load_txt_from_folder(folder_path: str, lang: str = LANG) -> pd.DataFrame:
    # walk all files in the directory!
    parsed_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    _doc = f.readlines()
                    parsed_data.extend(parse_document(_doc, lang))

    return pd.DataFrame(parsed_data)


def load_single_document(docs: List[str], lang: str = LANG) -> pd.DataFrame:
    # docs is a list of strings/sentences
    parsed_data = parse_document(docs, lang)
    df = pd.DataFrame(parsed_data)
    return df


def parse_document(docs: List[str], lang: str = LANG) -> List[str]:
    docs = [d.replace("\n", "") for d in docs]

    def sentencize(text):
        return nltk.sent_tokenize(text, language=lang)

    parsed_data = []
    sentences = []
    paragraph_sent_map = []
    for d_id, paragraph in enumerate(docs):
        d_sents = sentencize(paragraph)
        paragraph_sent_map.extend([d_id] * len(d_sents))
        sentences.extend(d_sents)

    for s_id, sent in enumerate(sentences):
        _d_id = paragraph_sent_map[s_id]
        parsed_data.append(
            {
                "id": f"DOC_{_d_id + 1}",  # for LLM referencing
                "page_id": _d_id,
                "sent_id": s_id,
                "text": sent,
            }
        )

    return parsed_data


@st.cache_data
def load_and_cache_documents(uploaded_file):
    ext = uploaded_file.name.split(".")[-1] if uploaded_file else None
    df = pd.DataFrame()
    if ext == "txt":
        uploaded_file = uploaded_file.read().decode("utf-8")
        uploaded_file = uploaded_file.split("\n")
        df = load_single_document(uploaded_file)
    elif ext == "zip":
        with zipfile.ZipFile(uploaded_file, "r") as z:
            z.extractall("temp")
            df = load_txt_from_folder("temp")

    if df.empty:
        raise ValueError("No data found in the uploaded file.")

    print(f"Loaded {len(df)} documents.")
    # columns: id/page_id/sent_id/text
    num_pages = df["page_id"].nunique()
    num_sents = df.shape[0]
    st.info(f"Found {num_pages} documents and {num_sents} sentences")
    return {
        "data": df.to_dict(orient="records"),
        "num_pages": num_pages,
        "num_sents": num_sents,
    }


def populate_collection(
    data: List[Dict[str, any]],
    collection_name: str,
    delete=False,
    BATCH_SIZE=32,
) -> Tuple[PersistentClient, Collection]:
    client, collection = get_client(
        persist=True,  # persist: store to disk (under the `chroma` folder)
        delete=delete,  # WARNING: enable ONLY if doing changes to the data
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    if collection.count() == 0:
        document_meta = []
        meta_text = "Adding metadata..."
        meta_bar = st.progress(0, text=meta_text)
        for percent_complete, row in enumerate(data):
            document_meta.append(
                {
                    "document": row["id"],
                    "sent_id": row["sent_id"],
                    "page_id": row["page_id"],
                }
            )
            current_perc = (percent_complete + 1) / len(data)
            current_perc = min(current_perc, 1.0)
            meta_bar.progress(current_perc)
        meta_bar.empty()

        documents = [d["text"] for d in data]

        with st.spinner("Computing embeddings..."):
            embeddings = embedding_model.encode(
                documents,
                show_progress_bar=True,
                batch_size=BATCH_SIZE,
            ).tolist()

        # a reference key for each document
        ids = []
        for d_id, d in enumerate(data):
            ids.append(f"{d_id}-{d['id']}-{d['page_id']}-{d['sent_id']}")

        batches = create_batches(
            api=client,
            ids=ids,
            embeddings=embeddings,
            metadatas=document_meta,
            documents=documents,
        )

        # i: index, e: embedding, m: metadata, d: document
        for i, e, m, d in batches:
            collection.add(i, e, m, d)

    return client, collection
