# ------------------------------------------------------------------------------
# File: rag.py
# Description: rag pipeline for KriRAG
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import jsonlines
import streamlit as st
from chromadb.types import Collection

from llm import (
    ask_llm,
    memory_prompt,
    parse_llm_output,
    pred,
    question_and_reason_prompt,
)
from utils.batch_util import get_sentence_batches
from utils.chroma import get_matching_documents


class RagConfig:
    LANG = "en"
    TOP_N = 10
    LLM_CTX_LEN = 8168
    NEW_TOKENS = 2048
    OUTPUT_DIR = "output"
    TOKEN_LEN = LLM_CTX_LEN // 4


class RAGProcessor:
    def __init__(
        self,
        collection: Collection,
        ip_address: str,
        port: int,
        config: RagConfig,
    ):

        self.collection = collection
        self.ip_address = ip_address
        self.port = port
        self.config = config

    def run(self, queries: List[str]) -> str:
        print(locals())
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        metadata = self.collection.get()["metadatas"]
        documents = {d["document"] for d in metadata}
        top_n = self.config.TOP_N if self.config.TOP_N != -1 else len(documents)

        case_folder = f"RAG_Top{top_n}_{start_time}"
        rag_path = os.path.join(self.config.OUTPUT_DIR, case_folder)
        os.makedirs(rag_path, exist_ok=True)

        for query in queries:
            self.process_query(query, top_n, rag_path)
        return rag_path

    def process_query(self, query: str, top_n: int, rag_path: str):
        st.write(f"Processing query: {query}")
        print(f"Query: {query}")
        matched_docs = get_matching_documents(
            collection=self.collection,
            query=query,
            n_results=top_n,
        )
        print(f"Reduced from {top_n} to {len(matched_docs)} documents")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = re.sub(r"[^\w\s]", "", query).replace(" ", "-")
        output_path = os.path.join(rag_path, f"{timestamp}_{filename}.jsonl")

        QUERY_MEMORY: List[str] = []
        with jsonlines.open(output_path, "w") as writer:
            progress_bar = st.progress(
                0, text=f"Processing {len(matched_docs)} documents..."
            )

            for i, doc_id in enumerate(matched_docs):
                current_percentage = (i + 1) / len(matched_docs)
                progress_bar.progress(
                    current_percentage, text=f"Document {i + 1}/{len(matched_docs)}"
                )
                self.process_document(
                    doc_id,
                    query,
                    QUERY_MEMORY,
                    writer,
                )

    def process_document(
        self,
        doc_id: str,
        query: str,
        query_memory: List[str],
        writer: jsonlines.Writer,
    ):
        matching_docs = self.collection.get(where={"document": {"$in": [doc_id]}})
        texts = matching_docs["documents"]
        print(f"Doc {doc_id} has {len(texts)} sentences")

        batches = get_sentence_batches(texts, self.config.TOKEN_LEN)

        skip_first_memory = False
        for batch, batch_texts in batches["batches"].items():
            full_text = " ".join(batch_texts)
            prev_info = ""
            if skip_first_memory:
                prev_info = self.get_previous_info(query, doc_id, query_memory)
            print(f"Prev info: {prev_info}")
            llm_output = ask_llm(
                query=query,
                text=full_text,
                ip_address=self.ip_address,
                port=self.port,
                extra=prev_info,
                doc_id=doc_id,
                tokens=self.config.NEW_TOKENS,
                prompt_source=question_and_reason_prompt,
                verbose=False,
                lang=RagConfig.LANG,
            )
            summary = self.extract_summary(llm_output)
            query_memory = [prev_info, summary]
            print(f"Memory: {query_memory}")
            record = self.create_record(
                doc_id,
                batch,
                query,
                llm_output,
                batches["map"][batch],
                full_text,
                query_memory,
            )
            writer.write(record)
            self.display_results(doc_id, query, llm_output, full_text, query_memory)
            skip_first_memory = True

    def get_previous_info(
        self,
        query: str,
        doc_id: str,
        query_memory: List[str],
    ) -> str:
        prev_info = pred(
            instruction=memory_prompt.format(
                previous_information=query_memory,
                query=query,
                DOC_ID=doc_id,
            ),
            ip_address=self.ip_address,
            port=self.port,
            max_tokens=1000,
            use_schema="summary",
        )
        parsed = parse_llm_output(prev_info)
        print(f"Previous info: {parsed}")
        return parsed.get("summary", "") if isinstance(parsed, dict) else ""

    def extract_summary(self, llm_output: Dict[str, Any]) -> str:
        return llm_output.get("summary", "")

    def create_record(
        self,
        doc_id: str,
        batch: int,
        query: str,
        llm_output: Dict[str, Any],
        sentences: List[str],
        text: str,
        memory: List[str],
    ) -> Dict[str, Any]:
        return {
            "id": doc_id,
            "batch": batch,
            "query": query,
            "llm_output": llm_output,
            "sentences_in_batch": sentences,
            "text": text,
            "memory": memory,
        }

    def display_results(
        self,
        doc_id: str,
        query: str,
        llm_output: Dict[str, Any],
        full_text: str,
        memory: List[str],
    ):
        with st.expander(
            f"Results for {doc_id} (relevance score: {llm_output.get('score', 'N/A')}/3)"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Query\n{query}")
                st.markdown("#### Generated questions")
                for q in llm_output.get("questions", []):
                    if "question" in q:
                        st.markdown(f"- {q['question']}")
                st.markdown(f"#### Summary\n{llm_output.get('summary', '')}")
                st.markdown("#### Memory")
                for info in memory:
                    st.markdown(f"- {info}")
            with col2:
                st.markdown("#### Full text")
                st.write(full_text)
            st.divider()
