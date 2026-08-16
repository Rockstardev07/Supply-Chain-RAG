"""
api/main.py — optional FastAPI backend exposing /ingest, /ask, /stats.

Run with:  uvicorn api.main:app --reload
Docs at:   http://localhost:8000/docs
"""

import sys
import tempfile
from pathlib import Path
from typing import List

# Allow running `uvicorn api.main:app` from the project root while still
# importing ingest.py and rag.py, which live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from ingest import COLLECTION_NAME, get_collection, ingest_files
from llm_providers import ANSWER_MODEL_NAME, ANSWER_PROVIDER_ORDER
from llm_providers import EMBEDDING_MODEL_NAME as EMBEDDING_MODEL
from rag import DEFAULT_TOP_K, ask

app = FastAPI(title="Meridian Supply Chain RAG API")


class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...)):
    tmp_dir = tempfile.mkdtemp()
    saved_paths = []
    for f in files:
        path = Path(tmp_dir) / f.filename
        path.write_bytes(await f.read())
        saved_paths.append(str(path))

    result = ingest_files(saved_paths)
    return result  # {"files": 2, "chunks": 96}


@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    result = ask(payload.question, top_k=payload.top_k)
    return result  # {"answer": "...", "sources": [{"file": "...", "page": 3}]}


@app.get("/stats")
async def stats_endpoint():
    collection = get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": ANSWER_MODEL_NAME,
        "answering_provider_order": ANSWER_PROVIDER_ORDER,
    }
