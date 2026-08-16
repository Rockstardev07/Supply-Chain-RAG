"""
ingest.py — load PDFs, chunk them, embed the chunks, and store them in a
persisted ChromaDB collection.

This file has exactly one job: turn a list of PDF file paths into searchable
vectors on disk. It does not answer questions — that happens in rag.py.
"""

import hashlib
from pathlib import Path

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_providers import EMBEDDING_MODEL_NAME as EMBEDDING_MODEL
from llm_providers import embed_texts

# ---- Configuration -----------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supplychain_docs"

# Chunk size ~1100 chars / 150 overlap: large enough that a full scorecard
# row or a full penalty clause survives inside one chunk (tables fall apart
# below ~1000 chars once pdfplumber flattens them to text), small enough
# that four or five chunks still fit comfortably in the prompt.
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150


# ---- Step 1: load -------------------------------------------------------
def load_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF, page by page.

    Returns a list of {"text": ..., "source": filename, "page": page_number}
    dicts, one per non-empty page. Page numbers are 1-indexed so they match
    what a human sees when they open the PDF.
    """
    pages = []
    filename = Path(file_path).name
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": filename, "page": i})
    return pages


# ---- Step 2: chunk -------------------------------------------------------
def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split each page's text into overlapping chunks, keeping source/page
    metadata attached to every chunk so retrieval can cite it later."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return chunks


def _chunk_id(chunk: dict) -> str:
    """Deterministic ID from source + page + text, so re-indexing the same
    PDF content upserts the existing rows instead of creating duplicates.
    (A random uuid4 here was the original bug: it made every index click
    add a fresh copy of every chunk, even for content already stored.)"""
    raw = f"{chunk['source']}|{chunk['page']}|{chunk['text']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---- Step 3 & 4: embed + store -------------------------------------------
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_files(file_paths: list[str]) -> dict:
    """Full pipeline for a list of PDF paths. Returns a small summary dict
    like {"files": 2, "chunks": 96} — the same shape the Streamlit app and
    the /ingest API endpoint report to the user.

    Uses upsert with a deterministic id per chunk, so indexing the same
    file twice (clicking "Index documents" again, or restarting the app
    and re-uploading) updates the existing rows instead of doubling them.
    The stored-chunk count only grows when genuinely new content is added.
    """
    all_chunks = []
    for path in file_paths:
        pages = load_pdf(path)
        all_chunks.extend(chunk_pages(pages))

    if not all_chunks:
        return {"files": len(file_paths), "chunks": 0}

    texts = [c["text"] for c in all_chunks]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    collection = get_collection()
    collection.upsert(
        ids=[_chunk_id(c) for c in all_chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
    )

    return {"files": len(file_paths), "chunks": len(all_chunks)}


if __name__ == "__main__":
    # Convenience: `python ingest.py` indexes every PDF already sitting in
    # data/, which is how the documents get loaded.
    pdf_paths = [str(p) for p in Path("data").glob("*.pdf")]
    if not pdf_paths:
        print("No PDFs found in data/. Add the PDFs there first.")
    else:
        result = ingest_files(pdf_paths)
        print(f"{result['files']} files processed, {result['chunks']} chunks stored.")
