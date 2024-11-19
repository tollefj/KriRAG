# ------------------------------------------------------------------------------
# File: ui.py
# Description: main ui for KriRAG (streamlit)
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

from utils.generic import init_dotenv, init_sqlite

init_dotenv()
init_sqlite()

import streamlit as st
from pandas import DataFrame

from combine import meta_summary
from initialize import load_documents, load_single_document
from prepare import initialize, populate
from rag import run_rag

import zipfile
from datetime import datetime
import os
import json
import pandas as pd

st.set_page_config(page_title="KriRAG", layout="wide")

default_queries = [
    "persons with residence and connections to the address (the crime scene) as owner, tenant, visitor, etc.",
    "how did the victim die (what is the cause of death?)",
    "details about the murder weapon (what is the murder weapon?)",
    "the victim's involvement in conflict or argument prior to death",
]

##############################################
# STREAMLIT APP
##############################################
# css hack to remove top header
st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
    display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("KriRAG")
subtext = "query-based analysis for criminal investigations"
st.sidebar.header("Server Configuration")
ip_address = st.sidebar.text_input("IP Address of API", value="127.0.0.1")
port = st.sidebar.number_input("API Port", value=8502, step=1)

# Add listeners for changes
if st.sidebar.button("Update Configuration"):
    st.session_state.ip_address = ip_address
    st.session_state.port = port
    st.success(f"Configuration updated to IP: {ip_address}, Port: {port}")

st.markdown(f"**{subtext}**")
st.divider()

is_initialized = False
query_area_exists = False
col1, col2 = st.columns(2)

file_options = {
    "file": "File upload (txt)",
    "folder": "Folder location",
}
file_options_inv = {v: k for k, v in file_options.items()}


df = DataFrame()

try:
    with col1:
        st.write("### Data:")
        st.write("Select between a case folder or document upload")

        _options = list(file_options.values())
        file_option = st.selectbox(
            label="Select data source",
            options=_options,
            placeholder="Select...",
            index=0,
        )

        if file_option == file_options["folder"]:
            df = load_documents(
                st.text_input(
                    "Location of case files:",
                )
            )

        elif file_option == file_options["file"]:
            _uploaded = st.file_uploader("Upload a file", type=["txt"])
            if _uploaded:
                _uploaded = _uploaded.read().decode("utf-8")
                _uploaded = _uploaded.split("\n")
                df = load_single_document(_uploaded)

        if not df.empty:
            initialization = initialize(df)
            data = initialization["data"]

            is_initialized = True

    with col2:
        st.markdown("### Queries:")
        if not is_initialized:
            st.warning("Please select data source first.")
        else:
            query_area = st.text_area(
                "Your queries (one per line):",
                "\n\n".join(default_queries),
                height=400,
                key="queries",
            )
            query_area_exists = True

except FileNotFoundError as e:
    st.error(f"Error: {e}")

st.divider()
if is_initialized and query_area_exists:
    to_delete = st.checkbox(
        "Delete previously computed data (local database)",
        value=False,
    )
    collection_name = (
        st.text_input(label="Name of your experiment:", value="KriRAG experiment 1")
        .replace(" ", "-")
        .lower()
    )
    queries = query_area.split("\n\n")
    queries = [q.strip() for q in queries if q]

    top_n = st.slider(
        "Number of candidate documents for each query.",
        1,
        min(initialization["num_pages"], 100),
        1,
    )
    st.write(
        "Note: a higher slider value will increase processing time, but will likely find more relevant documents."
    )

    rag_started = False
    btn = st.button("Run KriRAG", disabled=rag_started)
    if btn:
        rag_started = True
        with st.spinner("Analyzing..."):
            _, collection = populate(
                data, collection_name=collection_name, delete=to_delete
            )
            rag_path = run_rag(
                queries=queries,
                collection=collection,
                ip_address=ip_address,
                port=port,
                lang="en",
                top_n=top_n,
                llm_ctx_len=8168,
                new_tokens=4096,
            )

        with st.spinner("Processing findings..."):
            meta = meta_summary(rag_path, ip_address=ip_address, port=port)
            st.write("### Meta-summary of queries:")
            for m_id, meta_dict in enumerate(meta):
                st.write(f"Query: {meta_dict['query']}")
                st.write(f"{meta_dict['summary']}")
                st.divider()

        st.info(f"Analysis Complete! Download the CSV below.")
        rag_started = False

        
        all_data = []
        for file in sorted(os.listdir(rag_path)):
            if file.endswith(".jsonl"):
                file_path = os.path.join(rag_path, file)
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        data["id"] = file
                        all_data.append(data)
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(rag_path, "combined_results.csv")
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
            label="Download Results",
            data=f,
            file_name=f"{collection_name.replace(' ', '_')}_{timestamp}.csv",
            mime="text/csv",
            )
