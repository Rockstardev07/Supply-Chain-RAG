"""
llm_providers.py — a small provider abstraction so this project can run on
Google Gemini's free tier, with OpenAI as an optional fallback.

The embedding provider is chosen once at startup (whichever key is
present, Gemini preferred) and used for the entire collection: embeddings
from different providers do not share a vector space, and ChromaDB
requires one consistent vector dimensionality per collection. Switching
which key is set after documents have already been indexed requires
deleting chroma_db/ and re-indexing from scratch.

The answering step has no such constraint. ask_llm() tries each
configured Gemini model in order, then falls back to OpenAI if configured
and every Gemini attempt fails. This exists because Google has been
retiring Gemini model IDs faster than their published shutdown dates in
2026, so a single hardcoded model name is not reliable on its own.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY (recommended — free, get one "
        "at https://aistudio.google.com/apikey) and/or OPENAI_API_KEY in "
        "your .env file."
    )

# ---- Model names ------------------------------------------------------
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
# Gemini embeddings default to 3072 dims (Matryoshka-truncatable); 768 keeps
# the Chroma index small and fast for a project this size with negligible
# quality loss for short policy/report chunks.
GEMINI_EMBEDDING_DIM = 768

# Tried in order. gemini-3.6-flash is the current stable GA default;
# gemini-3.5-flash-lite is kept as a same-provider fallback since Google
# has been shutting down Gemini model IDs ahead of their published dates.
GEMINI_CHAT_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o"

# ---- Embedding provider: chosen ONCE, used for the whole collection ----
if GEMINI_API_KEY:
    EMBEDDING_PROVIDER = "gemini"
    EMBEDDING_MODEL_NAME = GEMINI_EMBEDDING_MODEL
else:
    EMBEDDING_PROVIDER = "openai"
    EMBEDDING_MODEL_NAME = OPENAI_EMBEDDING_MODEL

# ---- Answering providers: tried in order, first configured key first ---
ANSWER_PROVIDER_ORDER = []
if GEMINI_API_KEY:
    ANSWER_PROVIDER_ORDER.append("gemini")
if OPENAI_API_KEY:
    ANSWER_PROVIDER_ORDER.append("openai")

ANSWER_MODEL_NAME = GEMINI_CHAT_MODELS[0] if GEMINI_API_KEY else OPENAI_CHAT_MODEL


def _gemini_client():
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts with whichever provider was selected at
    startup (see EMBEDDING_PROVIDER above — it does not change per call).

    task_type should be "RETRIEVAL_DOCUMENT" when indexing chunks and
    "RETRIEVAL_QUERY" when embedding a question. Gemini uses this to
    produce better-matched vectors for retrieval; it's ignored by OpenAI.
    """
    batch_size = 100

    if EMBEDDING_PROVIDER == "gemini":
        from google.genai import types

        client = _gemini_client()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=GEMINI_EMBEDDING_DIM,
                ),
            )
            vectors.extend([e.values for e in response.embeddings])
        return vectors

    client = _openai_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def ask_llm(system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
    """Ask the answering model. Tries each Gemini model in GEMINI_CHAT_MODELS
    in order, then falls back to OpenAI if configured and every Gemini
    attempt fails.

    Gemini's newer models (3.x) have deprecated the temperature/top_p/top_k
    sampling parameters — they're no longer sent to Gemini. Determinism is
    instead handled through the system prompt's instruction to answer only
    from the supplied context. The temperature parameter is still applied
    for the OpenAI fallback, where it remains a valid parameter.
    """
    errors = []

    if "gemini" in ANSWER_PROVIDER_ORDER:
        from google.genai import types

        client = _gemini_client()
        for model_name in GEMINI_CHAT_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
                return response.text
            except Exception as e:  # noqa: BLE001 - deliberately broad, we fall back
                errors.append(f"gemini ({model_name}): {e}")
                continue

    if "openai" in ANSWER_PROVIDER_ORDER:
        try:
            client = _openai_client()
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            errors.append(f"openai: {e}")

    raise RuntimeError(
        "All configured answering providers failed:\n" + "\n".join(errors)
    )
