import os
from typing import Dict, List, Tuple

import streamlit as st
import torch
from chromadb import PersistentClient
from chromadb.types import Collection
from chromadb.utils.batch_utils import create_batches

from pandas import DataFrame

from utils.chroma import get_client
from utils.sbert_util import get_sbert


def initialize(df: DataFrame) -> Dict[str, any]:
    """Initialize the document analysis process."""
    os.makedirs("data/generated", exist_ok=True)

    if df.empty:
        return None
    # columns: id/page_id/sent_id/text
    num_pages = df["page_id"].nunique()
    num_sents = df.shape[0]
    st.info(f"Found {num_pages} documents and {num_sents} sentences")
    return {
        "data": df.to_dict(orient="records"),
        "num_pages": num_pages,
        "num_sents": num_sents,
    }


def populate(
    data: List[Dict[str, any]],
    collection_name: str,
    delete=False,
    BATCH_SIZE=32,
) -> Tuple[PersistentClient, Collection]:
    embedding_model = get_sbert()

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

    del embedding_model
    torch.cuda.empty_cache()

    return client, collection
