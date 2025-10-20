from typing import List, Dict, Any, Optional
import time
import hashlib
import logging
import os
import anyio
import requests

from core.config import settings
from core.utils.perf import profile_stage
from app.modules.lawfirmchatbot.services.embeddings import embed_text as provider_embed_text

logger = logging.getLogger(__name__)

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or getattr(settings, "GOOGLE_API_KEY", None)
)
LLM_MODEL_GEMINI_FAST = getattr(settings, "LLM_MODEL_GEMINI_FAST", "gemini-2.5-flash")
LLM_MODEL_GEMINI_HEAVY = getattr(settings, "LLM_MODEL_GEMINI_HEAVY", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_TIMEOUT_SECS = float(os.getenv("GEMINI_TIMEOUT_SECS", "40"))

_EMBED_CACHE: dict[str, tuple[float, list[float]]] = {}
_EMBED_TTL = settings.EMBED_CACHE_TTL_S
_EMBED_MAX = settings.EMBED_CACHE_MAX


def _ekey(model: str, text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return f"{model}:{h}"


def _embed_cache_get(model: str, text: str):
    key = _ekey(model, text)
    item = _EMBED_CACHE.get(key)
    if not item:
        return None
    ts, vec = item
    if time.time() - ts > _EMBED_TTL:
        _EMBED_CACHE.pop(key, None)
        return None
    return vec


def _embed_cache_put(model: str, text: str, vec: list[float]):
    if len(_EMBED_CACHE) >= _EMBED_MAX:
        _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
    _EMBED_CACHE[_ekey(model, text)] = (time.time(), vec)


def _format_messages_for_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages into a plain prompt for Gemini's text endpoint."""
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _infer_task(messages: List[Dict[str, str]], explicit: Optional[str] = None) -> str:
    """Lightweight inference of task type (qa | docgen)."""
    if explicit:
        return explicit

    docgen_markers = ("docgen", "document", "draft", "petition", "affidavit", "legal notice")

    system = (messages[0].get("content") or "").lower() if messages else ""
    user = (messages[-1].get("content") or "").lower() if messages else ""

    if any(marker in system for marker in docgen_markers) or any(marker in user for marker in docgen_markers):
        return "docgen"
    return "qa"


def _select_model(task: str, override: Optional[str] = None) -> str:
    if override:
        return override
    return LLM_MODEL_GEMINI_HEAVY if task == "docgen" else LLM_MODEL_GEMINI_FAST


def _build_generation_config(temperature: float, max_tokens: int) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if temperature is not None:
        config["temperature"] = float(max(0.0, min(temperature, 1.0)))
    if max_tokens:
        config["maxOutputTokens"] = int(max_tokens)
    return config


def _gemini_generate(
    model: str,
    prompt: str,
    intent: str = "general",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """Send text prompt to Gemini API and return response text."""
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY missing; cannot generate text.")
        return "⚠️ Gemini key missing; please configure environment."

    url = f"{GEMINI_URL}/{model}:generateContent?key={GEMINI_API_KEY}"
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    config = _build_generation_config(temperature, max_tokens)
    if config:
        payload["generationConfig"] = config

    try:
        resp = requests.post(url, json=payload, timeout=_GEMINI_TIMEOUT_SECS)
        if resp.status_code == 429:
            logger.warning("⚠️ Gemini 429 rate-limit, skipping generation.")
            return "⚠️ Gemini temporarily overloaded. Please retry shortly."
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if not text:
            logger.warning("⚠️ Gemini returned empty output for intent '%s'.", intent)
            return "⚠️ No output from Gemini."
        return text
    except Exception as exc:
        logger.error("⚠️ Gemini generation error: %s", exc)
        return "⚠️ Gemini temporarily unavailable."


def looks_like_citations_only(text: str) -> bool:
    """
    Check if response is citations-only without substantive content.

    Returns True only if:
    - Response is empty or very short (<50 chars)
    - OR has citations but lacks substantive content
    """
    t = text.strip()
    if not t or len(t) < 50:
        return True

    # Count actual content vs citations/references
    lines = [line.strip() for line in t.split("\n") if line.strip()]
    content_lines = 0
    citation_lines = 0

    for line in lines:
        line_lower = line.lower()
        # Skip empty lines and pure formatting
        if not line or line in ["---", "***", ""]:
            continue
        # Count citation/reference lines
        if (
            line_lower.startswith(("citations:", "references:", "**references**", "## references", "[1]", "[2]", "[3]", "[4]", "[5]"))
            or line_lower.startswith("reference pages:")
            or line_lower.startswith("**reference pages**")
            or (line_lower.startswith("[") and "]" in line_lower[:10])
        ):
            citation_lines += 1
        # Count substantive content lines
        elif len(line) > 15 and not line_lower.startswith(("[", "page ", "see ", "- [", "* [")):
            content_lines += 1

    has_sufficient_content = content_lines >= 2 or (len(t) >= 100 and content_lines >= 1)
    is_citations_only = not has_sufficient_content and (citation_lines > 0 or len(t) < 100)

    return is_citations_only


@profile_stage("embedding")
async def embed_text_async(text: str) -> list[float]:
    """Cached async embedding using Gemini-backed helper."""
    model = settings.EMBEDDING_MODEL
    hit = _embed_cache_get(model, text)
    if hit is not None:
        return hit

    vec = await provider_embed_text(text)
    if not vec:
        raise RuntimeError("Embedding provider returned empty vector.")

    _embed_cache_put(model, text, vec)
    return vec


def embed_text(text: str) -> list[float]:
    return anyio.run(embed_text_async, text)


async def chat_completion(
    messages: List[Dict[str, str]],
    *,
    task: Optional[str] = None,
    is_legal_query: bool = False,
    temperature: float = 0.3,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> str:
    """
    Gemini-native chat completion used by RAG, QA, and DocGen flows.
    """
    effective_task = _infer_task(messages, task)
    if is_legal_query and effective_task != "docgen":
        effective_task = "qa"

    prompt = _format_messages_for_prompt(messages)
    selected_model = _select_model(effective_task, override=model)

    logger.info("[LLM] Gemini call | model=%s task=%s tokens<=%s", selected_model, effective_task, max_tokens)

    response_text = await anyio.to_thread.run_sync(
        _gemini_generate,
        selected_model,
        prompt,
        effective_task,
        temperature,
        max_tokens,
    )
    return response_text.strip()


@profile_stage("llm_response")
async def run_llm_chat(system_prompt: str, user_message: str, history=None):
    """
    Backwards-compatible wrapper that routes all chat to Gemini.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_message})

    system_lower = (system_prompt or "").lower()
    task_type = "docgen" if any(
        marker in system_lower for marker in ("docgen", "document", "draft", "petition", "affidavit")
    ) else "qa"

    response_text = await chat_completion(
        messages,
        task=task_type,
        is_legal_query=task_type != "qa",
        temperature=0.4 if task_type == "docgen" else 0.3,
        max_tokens=1200 if task_type == "docgen" else 700,
    )
    return response_text.strip()
