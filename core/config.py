"""
Core Configuration and Services
Consolidated configuration settings and service wiring for the Law Firm Chatbot
"""

import logging
import os
from typing import Optional, Callable, List, Any
from pydantic_settings import BaseSettings
from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
import requests

logger = logging.getLogger(__name__)

def env_bool(key: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def env_int(key: str, default: int) -> int:
    """Parse integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def env_float(key: str, default: float) -> float:
    """Parse float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

class Settings(BaseSettings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Qdrant
    QDRANT_MODE: str = "cloud"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "law_docs_v1"       # physical
    QDRANT_COLLECTION_ALIAS: str = "law_docs_v1" # same as physical to avoid alias ops/warnings
    QDRANT_VECTOR_NAME: str = ""                 # "" or e.g. "text"
    
    # Legacy Qdrant settings for backward compatibility
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    LOG_QDRANT_HTTP: str = "0"
    
    # LLM (Gemini-focused configuration)
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_MODEL_FALLBACK: str = "gemini-2.5-flash"
    LLM_MODEL_GEMINI_FAST: str = "gemini-2.5-flash"
    LLM_MODEL_GEMINI_HEAVY: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    
    # LLM Behavior Settings (optimized for speed and quality)
    LLM_TEMPERATURE_DEFAULT: float = 0.3
    LLM_TEMPERATURE_LEGAL: float = 0.3
    LLM_MAX_TOKENS: int = 700
    LLM_MAX_OUTPUT_TOKENS: int = 700
    LLM_TOP_P: float = 1.0
    LLM_PRESENCE_PENALTY: float = 0.1
    LLM_FREQUENCY_PENALTY: float = 0.1
    
    # Qdrant Configuration (optimized for fast, high-quality retrieval)
    QDRANT_SEARCH_TIMEOUT_SECS: int = 2  # Tight timeout for faster responses
    QDRANT_TOP_K_DEFAULT: int = 6  # Focused retrieval: 6 high-quality chunks
    QDRANT_TOP_K_LONG_QUERY: int = 18  # Fetch more for MMR diversity (3x top_k)
    QDRANT_SCORE_THRESHOLD: float = 0.40  # Lowered to include more relevant chunks (was 0.55, too strict)
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "text-embedding-004"
    
    # RAG Debug Configuration
    DEBUG_RAG: bool = False  # Enable detailed RAG pipeline logging (set to True for debugging)
    
    # Performance Tuning Parameters
    QDRANT_USE_GRPC: bool = env_bool("QDRANT_USE_GRPC", True)
    QDRANT_HNSW_EF: int = env_int("QDRANT_HNSW_EF", 128)
    QDRANT_EXACT: bool = env_bool("QDRANT_EXACT", False)
    QDRANT_TIMEOUT_S: float = env_float("QDRANT_TIMEOUT_S", 3.0)
    EMBED_CACHE_TTL_S: int = env_int("EMBED_CACHE_TTL_S", 86400)
    EMBED_CACHE_MAX: int = env_int("EMBED_CACHE_MAX", 10000)
    EMBED_MODEL_DIM: int = env_int("EMBED_MODEL_DIM", 1536)
    RETRIEVAL_TOP_K: int = env_int("RETRIEVAL_TOP_K", 6)
    RETRIEVAL_SCORE_THRESH: float = env_float("RETRIEVAL_SCORE_THRESH", 0.18)

    # Web Search
    WEB_SEARCH_ENABLED: bool | None = None  # default: enabled if TAVILY_API_KEY exists
    TAVILY_API_KEY: str | None = None
    
    # CORS Configuration
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://localhost:3000"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()


def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client based on settings configuration."""
    if settings.QDRANT_MODE == "embedded":
        return QdrantClient(path="./qdrant_data")
    else:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_SEARCH_TIMEOUT_SECS
        )


def get_embedder():
    """
    Returns an embedding function compatible with Gemini embeddings.
    Replaces AsyncOpenAI from the old version.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", settings.LLM_PROVIDER or "gemini")
    if provider.lower() == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        model = os.getenv("EMBEDDING_MODEL", settings.EMBEDDING_MODEL)

        def embed_text(text: str):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={gemini_key}"
            payload = {"model": model, "content": {"parts": [{"text": text}]}}
            try:
                res = requests.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("embedding", {}).get("values", [])
            except Exception as e:
                print(f"⚠️ Gemini embedding error: {e}")
                return []

        return embed_text
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def get_llm_client() -> Optional[Any]:
    """
    Legacy hook for older code paths expecting an AsyncOpenAI client.
    Gemini routes interact via dedicated service modules, so we return None by default.
    """
    provider = (os.getenv("LLM_PROVIDER") or settings.LLM_PROVIDER or "").strip().lower()

    if provider == "openai":
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER is set to 'openai'")
        timeout = float(os.getenv("OPENAI_TIMEOUT_SECS", "30"))
        return AsyncOpenAI(
            api_key=api_key,
            timeout=timeout
        )

    return None


def wire_services(app: FastAPI) -> None:
    """Wire all singleton services into app.state on startup."""
    logger.info("Wiring global services...")
    
    # Wire settings
    app.state.settings = settings
    
    # Wire Qdrant client
    app.state.qdrant = get_qdrant_client()
    
    # Wire embedding client
    app.state.embedder = get_embedder()
    
    # Wire LLM client
    app.state.llm_client = get_llm_client()
    
    # Ensure collection (fail-soft)
    try:
        from app.modules.lawfirmchatbot.services.vector_store import ensure_collection
        ensure_collection(app.state.qdrant, dim=1536)
        logger.info("Qdrant collection initialized successfully")
            
    except Exception as e:
        logger.warning(f"Qdrant collection initialization failed: {str(e)}")
        logger.info("Application will continue without Qdrant initialization")
    
    logger.info("Service container wiring completed successfully")


async def perform_warmup(app: FastAPI) -> None:
    """Perform async warmup operations (call this from startup event)."""
    try:
        from app.modules.lawfirmchatbot.services.embeddings import embed_text
        from app.modules.lawfirmchatbot.services.vector_store import get_runtime_collection_name
        
        v = await embed_text("warmup")
        app.state.qdrant.search(
            collection_name=get_runtime_collection_name(),
            query_vector=v,
            limit=1
        )
        logger.info("Qdrant warmup completed successfully")
    except Exception as warmup_error:
        logger.info(f"Qdrant warmup failed (non-critical): {warmup_error}")


# Legacy compatibility - maintain the old Services class structure
class Services:
    """Legacy Services container for backward compatibility."""
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embed_client: Callable[[str], List[float]],
        llm_client: Optional[Any],
    ):
        self.qdrant_client = qdrant_client
        self.embed_client = embed_client
        self.llm_client = llm_client


def get_legacy_services(request: Request) -> Services:
    """
    Get services in the legacy format for backward compatibility.
    
    Args:
        request: FastAPI request object containing app.state
        
    Returns:
        Services: Legacy services container
    """
    state = request.app.state
    return Services(
        qdrant_client=state.qdrant,
        embed_client=state.embedder,
        llm_client=state.llm_client
    )
