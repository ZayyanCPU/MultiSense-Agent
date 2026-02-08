"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ─── Enums ───────────────────────────────────────────────

class MessageType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    DOCUMENT = "document"


class ProcessingStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


# ─── Chat Schemas ────────────────────────────────────────

class ChatRequest(BaseModel):
    """Direct chat API request."""
    message: str = Field(..., description="User message text")
    session_id: str = Field(default="default", description="Session identifier for conversation memory")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context retrieval")


class ChatResponse(BaseModel):
    """Chat API response."""
    response: str
    session_id: str
    sources: List[str] = Field(default_factory=list, description="Source documents used in RAG")
    processing_time: float = Field(description="Processing time in seconds")
    message_type: MessageType = MessageType.TEXT


# ─── WhatsApp Webhook Schemas ────────────────────────────

class WhatsAppMessage(BaseModel):
    """Parsed WhatsApp incoming message."""
    message_id: str
    from_number: str
    timestamp: str
    message_type: MessageType
    text: Optional[str] = None
    media_id: Optional[str] = None
    mime_type: Optional[str] = None
    caption: Optional[str] = None


class WhatsAppWebhookVerification(BaseModel):
    """WhatsApp webhook verification query params."""
    hub_mode: str = Field(alias="hub.mode")
    hub_verify_token: str = Field(alias="hub.verify_token")
    hub_challenge: str = Field(alias="hub.challenge")

    model_config = {"populate_by_name": True}


# ─── Document Upload Schemas ─────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response after PDF/document upload for RAG ingestion."""
    filename: str
    chunks_created: int
    status: ProcessingStatus
    message: str


# ─── Health Check ────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    services: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Conversation Memory ────────────────────────────────

class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: MessageType = MessageType.TEXT


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
