from typing import List
import asyncio
import logging
import os
import time
import requests
from core.config import settings
from dotenv import load_dotenv
load_dotenv()

import numpy as np

# Temporary local embedding mode to bypass Gemini rate limits
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "768"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini embedding configuration
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = "gemini"
DEFAULT_EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "embedding-001")
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()
EXPECTED_DIM = getattr(settings, "EMBED_MODEL_DIM", 768)
_ALLOWED_GEMINI_768_MODELS = {"embedding-001"}
if EMBEDDING_MODEL not in _ALLOWED_GEMINI_768_MODELS:
    logger.warning(
        "Embedding model '%s' is not a known Gemini 768-dim model; falling back to '%s'.",
        EMBEDDING_MODEL,
        DEFAULT_EMBEDDING_MODEL,
    )
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or getattr(settings, "GOOGLE_API_KEY", None)
)
_GEMINI_TIMEOUT_SECS = 30
_last_known_dim: int | None = EXPECTED_DIM

if not GEMINI_API_KEY and not USE_LOCAL_EMBEDDINGS:
    raise RuntimeError("GEMINI_API_KEY missing - required for Gemini embeddings.")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _zero_embedding(dim: int | None = None) -> List[float]:
    """Generate a zero vector for fallback consistency."""
    global _last_known_dim
    target_dim = dim or _last_known_dim or EXPECTED_DIM
    _last_known_dim = target_dim
    return [0.0] * target_dim


def _normalize_embedding(vector: List[float]) -> List[float]:
    """Ensure embedding length matches expected Gemini dimension."""
    global _last_known_dim
    if not vector:
        return _zero_embedding(EXPECTED_DIM)

    actual_dim = len(vector)
    _last_known_dim = actual_dim
    if actual_dim == EXPECTED_DIM:
        return vector

    if actual_dim > EXPECTED_DIM:
        logger.warning(
            "Gemini embedding dimension %d exceeds expected %d; truncating.",
            actual_dim,
            EXPECTED_DIM,
        )
        return vector[:EXPECTED_DIM]

    logger.warning(
        "Gemini embedding dimension %d below expected %d; padding with zeros.",
        actual_dim,
        EXPECTED_DIM,
    )
    return vector + [0.0] * (EXPECTED_DIM - actual_dim)


def _request_embedding_with_backoff(url: str, payload: dict) -> List[float]:
    """Call Gemini embeddings endpoint with exponential backoff for rate limits."""
    delay = 0.5
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=_GEMINI_TIMEOUT_SECS)
            if response.status_code == 429:
                logger.warning("Gemini 429 rate-limit hit, retry %d/5 after %.1fs", attempt + 1, delay)
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            body = response.json()
            vector = body.get("embedding", {}).get("values", [])
            if isinstance(vector, list) and vector:
                return vector
        except Exception as exc:
            logger.warning("Gemini embed attempt %d failed: %s", attempt + 1, exc)
            time.sleep(delay)
            delay *= 2
    return _zero_embedding()


async def _embed_with_gemini(text: str) -> List[float]:
    """Robust Gemini embedding with minimal retries and instant fallback."""
    global _last_known_dim
    if USE_LOCAL_EMBEDDINGS:
        seed = abs(hash(text)) % (2**32)
        vec = np.random.default_rng(seed).random(LOCAL_EMBED_DIM).tolist()
        _last_known_dim = LOCAL_EMBED_DIM
        return vec

    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY missing; returning zero embedding.")
        return _zero_embedding()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": EMBEDDING_MODEL, "content": {"parts": [{"text": text}]}}

    def _fast_request():
        import time
        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=_GEMINI_TIMEOUT_SECS)
                if response.status_code == 429:
                    logger.warning(f"⚠️ Gemini rate-limit (429) — skipping after {attempt+1}/2 attempts")
                    if attempt == 0:
                        time.sleep(0.4)
                        continue
                    break
                response.raise_for_status()
                body = response.json()
                vec = body.get("embedding", {}).get("values", [])
                if vec:
                    return vec
            except Exception as exc:
                logger.warning(f"⚠️ Gemini embedding attempt {attempt+1} failed: {exc}")
                if attempt == 0:
                    time.sleep(0.2)
        logger.warning("⚠️ Gemini embedding failed — returning zero vector instantly.")
        return _zero_embedding()

    vector = await asyncio.to_thread(_fast_request)
    if vector:
        _last_known_dim = len(vector)
    return vector


def _embed_with_gemini_sync(text: str) -> List[float]:
    """Synchronous Gemini embedding call."""
    global _last_known_dim
    if not text.strip():
        return _zero_embedding()

    if USE_LOCAL_EMBEDDINGS:
        seed = abs(hash(text)) % (2**32)
        vec = np.random.default_rng(seed).random(LOCAL_EMBED_DIM).tolist()
        _last_known_dim = LOCAL_EMBED_DIM
        return vec

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": EMBEDDING_MODEL, "content": {"parts": [{"text": text}]}}

    try:
        vector = _request_embedding_with_backoff(url, payload)
        return _normalize_embedding(vector)
    except Exception as exc:
        logger.warning(f"Gemini embedding sync error: {exc}")
        return _zero_embedding()


# ---------------------------------------------------------------------------
# Async / Batch Embedding Functions
# ---------------------------------------------------------------------------
async def embed_text(text: str) -> List[float]:
    """Generate a single Gemini embedding."""
    if USE_LOCAL_EMBEDDINGS:
        return await _embed_with_gemini(text)

    try:
        return await _embed_with_gemini(text)
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return _zero_embedding()


async def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts asynchronously (batch safe)."""
    if not texts:
        return []

    embeddings: List[List[float]] = []
    for text in texts:
        try:
            vector = await embed_text(text)
        except Exception as exc:
            logger.error(f"Gemini embedding failed for text length {len(text)}: {exc}")
            vector = _zero_embedding()
        embeddings.append(vector)
    return embeddings


async def hybrid_embed_text(text: str, *, legal_density: float | None = None) -> List[float]:
    """Hybrid embedding with simple legal domain weighting."""
    base_list = await embed_text(text)
    if not base_list:
        return _zero_embedding()

    base = np.array(base_list)
    if legal_density is None:
        return base.tolist()

    legal_weight = min(max(legal_density, 0.0), 1.0) * 0.3
    general_weight = 1.0 - legal_weight
    legal = np.zeros_like(base)
    legal[-50:] = 1.0
    if np.linalg.norm(legal) > 0:
        legal = legal / np.linalg.norm(legal)
    vec = general_weight * base + legal_weight * legal
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Synchronous batch embedding helper for Gemini."""
    if not texts:
        return []

    embeddings: List[List[float]] = []
    for text in texts:
        try:
            vector = _embed_with_gemini_sync(text)
        except Exception as exc:
            logger.warning(f"Gemini sync embedding error: {exc}")
            vector = _zero_embedding()
        embeddings.append(vector)
    return embeddings
