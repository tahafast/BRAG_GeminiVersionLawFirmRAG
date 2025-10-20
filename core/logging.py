"""Unified logging configuration for the Law Firm chatbot."""

import logging
import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "brag_activity.log"

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setFormatter(formatter)

root_logger = logging.getLogger("RAGLogger")
if not root_logger.handlers:
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

root_logger.info("🪶 Unified Gemini RAG logger initialized")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    return logging.getLogger(name)
