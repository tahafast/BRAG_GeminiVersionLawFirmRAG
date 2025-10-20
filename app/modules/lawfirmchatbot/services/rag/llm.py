import os
import logging
import requests

# === Gemini-only configuration ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FAST_MODEL = os.getenv("LLM_MODEL_GEMINI_FAST", "gemini-2.5-flash")
GEMINI_HEAVY_MODEL = os.getenv("LLM_MODEL_GEMINI_HEAVY", "gemini-2.5-flash")  # same for now

logger = logging.getLogger(__name__)


def call_gemini(prompt: str, mode: str = "fast") -> str:
    """
    Call Gemini model via Google AI Studio REST API.
    mode: 'fast' -> flash | 'heavy' -> pro (if available)
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key missing; cannot fulfill request.")
        return "⚠️ Gemini API temporarily unavailable. Please retry shortly."

    model = GEMINI_FAST_MODEL if mode == "fast" else GEMINI_HEAVY_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        res = requests.post(url, json=body, timeout=60)
        res.raise_for_status()
        data = res.json()
        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0].get("text", "")
        return "[No response from Gemini model]"
    except Exception as e:
        logger.error("⚠️ Gemini API error: %s", e)
        return "⚠️ Gemini API temporarily unavailable. Please retry shortly."


def generate_llm_response(prompt: str, task_type: str = "qa") -> str:
    """
    Unified entry point for RAG/DocGen tasks.
    'qa' → light/fast model, 'docgen' → heavy model (same for now).
    """
    mode = "fast" if task_type == "qa" else "heavy"
    return call_gemini(prompt, mode)


def detect_active_llm():
    """Return active LLM provider and Gemini model configuration."""
    provider = os.getenv("LLM_PROVIDER", "gemini")
    fast = os.getenv("LLM_MODEL_GEMINI_FAST", GEMINI_FAST_MODEL)
    heavy = os.getenv("LLM_MODEL_GEMINI_HEAVY", GEMINI_HEAVY_MODEL)
    return {"provider": provider, "fast": fast, "heavy": heavy}
