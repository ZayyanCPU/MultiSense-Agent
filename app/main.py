"""
MultiSense Agent - Main FastAPI Application Entry Point.

A multi-modal AI chatbot with WhatsApp integration supporting
text, voice, images, and PDFs using RAG and n8n automation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
import logging

from app.config import get_settings
from app.routes import chat, webhook, health


# ─── Logging Configuration ──────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


# ─── App Lifespan ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    settings = get_settings()
    logger.info(
        "multisense_agent_starting",
        version="1.0.0",
        debug=settings.app_debug,
        model=settings.hf_chat_model,
    )

    # Startup: verify connections
    try:
        from app.services.rag_engine import get_rag_engine
        rag = get_rag_engine()
        stats = await rag.get_collection_stats()
        logger.info("chromadb_connected", **stats)
    except Exception as e:
        logger.warning("chromadb_connection_warning", error=str(e))

    yield

    # Shutdown
    logger.info("multisense_agent_shutdown")


# ─── FastAPI App ─────────────────────────────────────────

app = FastAPI(
    title="MultiSense Agent",
    description=(
        "🤖 Multi-Modal AI Chatbot with WhatsApp Integration\n\n"
        "Supports text, voice messages, image analysis, and PDF document processing "
        "using RAG (Retrieval Augmented Generation) and n8n workflow automation.\n\n"
        "Powered by FREE open-source models via HuggingFace Inference API.\n\n"
        "**Capabilities:**\n"
        "- 💬 Text chat with conversation memory\n"
        "- 🎤 Voice message transcription (Whisper)\n"
        "- 📸 Image analysis (BLIP + LLM)\n"
        "- 📄 PDF document ingestion (RAG)\n"
        "- 📱 WhatsApp Business API integration\n"
        "- 🔄 n8n workflow automation\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS Middleware ─────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routes ────────────────────────────────────

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(webhook.router)


# ─── Root Endpoint ───────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with project information."""
    return {
        "project": "MultiSense Agent",
        "version": "1.0.0",
        "description": "Multi-Modal AI Chatbot with WhatsApp Integration",
        "docs": "/docs",
        "health": "/health",
        "capabilities": [
            "text_chat",
            "voice_transcription",
            "image_analysis",
            "pdf_rag_ingestion",
            "whatsapp_integration",
            "n8n_automation",
        ],
    }


# ─── Main Entry Point ───────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.app_log_level,
    )
