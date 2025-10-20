# BRAG AI (Gemini Edition)

Modernised Retrieval-Augmented Generation for legal teams, now powered end-to-end by Google Gemini. This release replaces every OpenAI dependency with Gemini-native generation and embeddings, tightens Qdrant bootstrap, and adds local development fallbacks so you can iterate without rate-limit anxiety.

---

## What's New in the Gemini Edition

- **Gemini-native LLM pipeline** – `app/modules/lawfirmchatbot/services/llm.py` now calls the Gemini REST API for chat and doc generation, returning friendly fallback messages when the API is unavailable.
- **Gemini embeddings + fast retries** – Embedding requests run through the lightweight `_fast_request` helper with deterministic local-mode support.
- **Local embedding bypass** – Set `USE_LOCAL_EMBEDDINGS=true` to generate deterministic NumPy vectors (seeded per text) when you want to work offline or avoid billing during development.
- **Automatic Qdrant alias bootstrap** – `core/config.get_qdrant_client()` recreates the `law_docs_gemini_v1` collection if missing and ensures the `law_docs_gemini_current` alias always points to it.
- **Safer routing** – RAG orchestration now guarantees the retriever is active for QA/DocGen modes and disables it automatically if the alias check fails.
- **Frontend label update** – The landing page proudly displays **“BRAG AI (Gemini Version)”** so users know which stack they are on.

---

## Feature Overview

- **Document ingestion** – Upload PDFs/DOCX/TXT via the FastAPI endpoints or the lightweight frontend.
- **Gemini-powered answers** – QA and document drafting run through Gemini (flash/pro) with mode-aware prompts.
- **Qdrant vector store** – `law_docs_gemini_current` alias targets the active collection for ingestion/search.
- **Context-aware routing** – The orchestrator inspects every request, toggles retrieval, and enforces placeholder rules for DocGen.
- **Observability hooks** – Startup logs provide alias status, LLM models in use, and warm-up progress.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI (app/main.py)                                            │
│  ├── core.config               → settings + Qdrant bootstrap     │
│  ├── modules.lawfirmchatbot    → ingestion / RAG / docgen        │
│  └── services.vector_store     → Qdrant helpers & retries        │
│                                                                  │
│ Gemini APIs                                                       │
│  ├── Text generation (LLM)                                       │
│  └── Embeddings                                                  │
│                                                                  │
│ Qdrant                                                            │
│  ├── Collection: law_docs_gemini_v1                              │
│  └── Alias:      law_docs_gemini_current                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.9 or newer
- pip
- Qdrant instance (cloud or self-hosted) reachable from the backend
- Google Gemini API key (unless you intend to run in local embedding mode)

---

## Environment Configuration

1. Copy the example environment file:
   ```bash
   cp env.template .env
   ```
2. Update the following keys (minimum requirements):

   | Variable | Description |
   |----------|-------------|
   | `GEMINI_API_KEY` | Primary Gemini key used for chat + embeddings. |
   | `QDRANT_URL` | Base URL for your Qdrant instance. |
   | `QDRANT_API_KEY` | (Optional) API key for managed Qdrant. |
   | `QDRANT_COLLECTION` | Physical collection name (defaults to `law_docs_gemini_v1`). |
   | `QDRANT_COLLECTION_ALIAS` | Alias consumed by the app (defaults to `law_docs_gemini_current`). |

3. Optional developer toggles:

   ```env
   USE_LOCAL_EMBEDDINGS=true   # bypass Gemini embeddings with deterministic vectors
   LOCAL_EMBED_DIM=768         # dimension for local vectors (matches Gemini)
   ```

---

## Backend Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server boot log will confirm:

- Qdrant collection + alias status
- Gemini model configuration
- Warm-up embedding/search health

### Useful URLs

- API root: `http://127.0.0.1:8000/`
- Docs (Swagger): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/healthz`

---

## Frontend Preview (Static)

The frontend is a static HTML/CSS bundle under `frontend/`.

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500/index.html` to view the Gemini-branded landing page and navigation shortcuts for ingest/chat.

---

## Document Workflow

1. **Ingest** a PDF/DOCX/TXT via `/api/v1/lawfirm/upload-document` or the UI.
2. The pipeline:
   - Extracts text into chunks.
   - Generates Gemini embeddings (or deterministic vectors if local mode is enabled).
   - Upserts into Qdrant using the active alias.
3. **Query** with `/api/v1/lawfirm/query` or the chat UI. The orchestrator:
   - Detects QA vs DocGen intent.
   - Pulls relevant context (unless retrieval is intentionally disabled).
   - Routes the prompt to the Gemini model tier.

---

## Local Embedding Mode

Use this when you need to work offline or want predictable embeddings in tests.

```env
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBED_DIM=768
```

Effects:

- `_embed_with_gemini` returns a deterministic vector seeded by the text hash.
- Qdrant writes still occur (handy for smoke tests).
- LLM responses continue to rely on Gemini unless you add your own guardrails.

Reset the flag to `false` before deploying to production.

---

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| **“Collection alias doesn’t exist”** | Restart the backend; `core.config.get_qdrant_client()` will recreate the collection and alias. |
| **Gemini 429 rate limit** | The embedding helper already retries once with backoff. For repeated issues, enable `USE_LOCAL_EMBEDDINGS` temporarily. |
| **Retriever unexpectedly disabled** | Check logs for `[routing] Retriever disabled — missing Qdrant alias.` Confirm alias settings in `.env` and ensure Qdrant is reachable. |
| **Frontend still shows legacy branding** | Hard refresh the browser (`CTRL+F5`) or clear cache to pick up `frontend/index.html` changes. |

---

## Contributing & Support

- Open issues or feature requests through your team’s tracker.
- For operational questions, review the logs emitted on startup and during ingestion.
- Always document new environment variables in this README to keep the Gemini edition aligned.

---

**Status:** Active  
**Last Updated:** 2025-10-20  
**Maintainers:** BRAG AI Platform Team
