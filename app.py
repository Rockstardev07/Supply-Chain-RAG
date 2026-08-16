"""
app.py — Streamlit interface: upload PDFs, index them, ask questions,
see the answer with sources underneath.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ingest import get_collection, ingest_files
from rag import DEFAULT_TOP_K, ask

st.set_page_config(
    page_title="Supply Chain Assistant",
    layout="centered",
    initial_sidebar_state="expanded",  # sidebar starts open; click the "<" arrow at its top edge to collapse/hide it
)
st.title("Supply Chain Assistant")
st.caption(
    "Ask a question in plain English. The answer is drawn from whichever "
    "indexed document holds it — the quarterly review, the policy "
    "handbook, or both."
)

# --- Sidebar: upload + index ------------------------------------------------
with st.sidebar:
    st.header("Upload & index")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Index documents", disabled=not uploaded_files):
        with st.spinner("Reading, chunking, and embedding..."):
            tmp_dir = tempfile.mkdtemp()
            saved_paths = []
            for f in uploaded_files:
                path = Path(tmp_dir) / f.name
                path.write_bytes(f.getbuffer())
                saved_paths.append(str(path))

            result = ingest_files(saved_paths)

        st.success(f"{result['files']} files processed, {result['chunks']} chunks stored.")

    st.divider()
    try:
        current_count = get_collection().count()
        st.caption(f"Currently indexed: {current_count} chunks in Chroma.")
    except Exception:
        st.caption("Currently indexed: 0 chunks in Chroma.")

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=3,
        max_value=10,
        value=DEFAULT_TOP_K,
        help=(
            "Higher values retrieve more information and can help "
            "with questions involving multiple documents."
        )
    )

# --- Main: ask (continuous chat) ---------------------------------------
st.header("Ask your questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            st.subheader("Answer")
            st.write(message["content"])

            sources = message.get("sources", [])
            st.subheader("Sources")
            if sources:
                for s in sources:
                    st.markdown(f"- **{s['file']}**, page {s['page']}")
            else:
                st.caption("No sources retrieved.")

# Chat input box
question = st.chat_input(
    "e.g. What is the approval authority for a purchase order worth ₹1.4 crore?"
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating your answer..."):
            try:
                result = ask(question, top_k=top_k)
            except Exception as e:
                result = {
                    "answer": f"Sorry, I couldn't generate an answer.\n\n**Error:** `{str(e)}`",
                    "sources": [],
                }

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            for s in result["sources"]:
                st.markdown(f"- **{s['file']}**, page {s['page']}")
        else:
            st.caption("No sources retrieved.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )