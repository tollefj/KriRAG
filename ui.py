from utils.generic import init_dotenv, init_sqlite

init_dotenv()
init_sqlite()

import streamlit as st
from pandas import DataFrame

from src.combine import meta_summary
from src.initialize import load_documents, load_single_document
from src.prepare import initialize, populate
from src.rag import run_rag

st.set_page_config(page_title="KriRAG")

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
# st.markdown("#### query-based analysis for criminal investigations")
subtext = "query-based analysis for criminal investigations"
# nicer html-formatted off-green color:
# st.markdown(
#     f'<p style="color:#4CAF50;">{subtext}</p>',
#     unsafe_allow_html=True,
# )
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

        # selection box between Folder and File upload
        st.write("### Data:")
        st.write("Select between a case folder or document upload")

        # concat "" and file option keys:
        # _options = [""] + list(file_options.values())
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
                lang="en",
                top_n=top_n,
                llm_ctx_len=8168,
                new_tokens=4096,
            )

        with st.spinner("Processing findings..."):
            meta = meta_summary(rag_path)
            st.write("### Meta-summary of queries:")
            for m_id, meta_dict in enumerate(meta):
                st.write(f"Query: {meta_dict['query']}")
                st.write(f"{meta_dict['summary']}")
                st.divider()

        st.info(f"Analysis Complete! You can find the files in `{rag_path}`")
        rag_started = False
