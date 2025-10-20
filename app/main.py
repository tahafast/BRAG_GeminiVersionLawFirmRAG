# app/main.py
import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
from app.modules.router import router as modules_router
from core.config import settings, wire_services, perform_warmup
from core.logging import get_logger
from app.modules.lawfirmchatbot.services.vector_store import (
    get_qdrant_client,
    get_active_collection,
    ensure_alias,
)
from app.modules.lawfirmchatbot.services.rag.llm import detect_active_llm

logger = get_logger("RAGLogger")


def create_app():
    app = FastAPI(title="Law Firm Chatbot")
    wire_services(app)
    app.include_router(modules_router)

    # --- CORS (allow local static servers & file:// testing) ---
    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = ["*"] if allow_origins.strip() == "*" else [o.strip() for o in allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,          # e.g. ["http://127.0.0.1:5500","http://localhost:5173","*"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )

    # --- Optional: serve the frontend from FastAPI to avoid CORS completely ---
    # Set SERVE_FRONTEND=1 (default) to mount /ui -> ./frontend
    if os.getenv("SERVE_FRONTEND", "1") == "1":
        # mounting at /ui ensures /api/* keeps working
        FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
        if FRONTEND_DIR.exists():
            app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    
    # (append in app/main.py after include_router)
    for r in app.routes:
        try:
            logging.getLogger("router.map").info("ROUTE %s %s", ",".join(sorted(r.methods or [])), r.path)
        except Exception:
            pass
    
    return app


app = create_app()


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    # Redirect root to static UI if present, otherwise return a tiny OK
    FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/ui/")
    return PlainTextResponse("ok", status_code=200)


@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
async def healthz():
    return PlainTextResponse("ok", status_code=200)


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("🚀 Starting LawFirm Gemini RAG backend initialization...")

    try:
        client = get_qdrant_client()
        ensure_alias(client)
        active_alias = get_active_collection()
        logger.info(f"✅ Qdrant alias confirmed — Active collection: {active_alias}")

        llm_config = detect_active_llm()
        logger.info(
            "🧠 LLM provider: %s | Fast: %s | Heavy: %s",
            llm_config.get("provider"),
            llm_config.get("fast"),
            llm_config.get("heavy"),
        )

        logger.info("✨ Gemini RAG system successfully initialized and ready.")
    except Exception as e:
        logger.exception(f"❌ Startup initialization failed: {e}")

    # Initialize database tables
    try:
        from app.services.memory.init_db import init_database
        logger.info("Initializing chat memory database...")
        await init_database()
        logger.info("Chat memory database initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize chat memory database: {e}")
        logger.warning("Chat memory features may not work properly")
    
    # Optional HTTP wire logging for debugging
    if settings.LOG_QDRANT_HTTP == "1":
        logging.getLogger("httpx").setLevel(logging.INFO)
    
    # Perform async warmup operations
    await perform_warmup(app)
    
    logger.info("Application startup completed successfully")
