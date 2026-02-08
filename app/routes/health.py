"""
Health and utility routes.
"""

from fastapi import APIRouter
import structlog

from app.models import HealthResponse
from app.services.rag_engine import get_rag_engine
from app.services.memory_service import get_memory_service

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Health & Utilities"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all connected services:
    - ChromaDB vector store connection
    - Active conversation sessions
    """
    rag = get_rag_engine()
    memory = get_memory_service()

    # Check ChromaDB
    chroma_stats = await rag.get_collection_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "chromadb": chroma_stats,
            "active_sessions": len(memory.get_active_sessions()),
        },
    )


@router.get("/api/v1/sessions")
async def list_sessions():
    """List all active conversation sessions."""
    memory = get_memory_service()
    sessions = memory.get_active_sessions()
    return {"active_sessions": sessions, "count": len(sessions)}


@router.delete("/api/v1/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    memory = get_memory_service()
    cleared = memory.clear_session(session_id)
    if cleared:
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@router.get("/api/v1/knowledge-base/stats")
async def knowledge_base_stats():
    """Get statistics about the RAG knowledge base."""
    rag = get_rag_engine()
    stats = await rag.get_collection_stats()
    return stats
