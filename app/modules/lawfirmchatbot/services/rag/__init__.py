"""RAG orchestration services."""

import os

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.18"))

__all__ = ["RAG_TOP_K", "RAG_MIN_SCORE"]
