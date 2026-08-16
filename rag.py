"""
rag.py — retrieve the most relevant chunks for a question and ask the
answering model to respond using only those chunks.
"""

import chromadb

from ingest import CHROMA_DIR, COLLECTION_NAME
from llm_providers import ANSWER_MODEL_NAME as ANSWER_MODEL
from llm_providers import ask_llm, embed_texts

DEFAULT_TOP_K = 8  # deliberately on the higher side — see README on why
# cross-document questions need top_k around 5-6 or higher rather than 3-4.
TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are an internal assistant for a supply chain company. "
    "Answer only from the context provided below. If the context does not "
    "contain the answer, say the information is not available in the "
    "uploaded documents. Do not use outside knowledge, and do not guess. "
    "When you use a number or a clause, mention which document it came from. "

    "If a question depends on a classification or condition, briefly show "
    "both possibilities, then check the context for evidence that satisfies "
    "either one. If the evidence supports a conclusion, state it clearly "
    "as the final answer instead of leaving it open."
)

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Embed the question and pull the top_k closest chunks out of Chroma."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Full ask pipeline. Returns {"answer": ..., "sources": [...]} — the
    same shape the Streamlit app and the /ask API endpoint return."""
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "The information is not available in the uploaded documents "
            "(no documents have been indexed yet).",
            "sources": [],
        }

    context = build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    answer = ask_llm(SYSTEM_PROMPT, user_message, temperature=TEMPERATURE)

    # De-duplicate sources (several chunks can come from the same page).
    seen = set()
    sources = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": c["source"], "page": c["page"]})

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    q = input("Question: ")
    result = ask(q)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['file']}, page {s['page']}")
