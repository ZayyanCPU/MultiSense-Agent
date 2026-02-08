"""
Chat API routes - Direct chat interface (non-WhatsApp).
Useful for testing and web-based integrations.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import structlog

from app.models import ChatRequest, ChatResponse, DocumentUploadResponse, ProcessingStatus
from app.services.processor import get_processor
from app.services.rag_engine import get_rag_engine

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Direct text chat endpoint.
    
    Send a text message and get an AI response with optional RAG augmentation.
    
    - **message**: Your text message
    - **session_id**: Session identifier for conversation memory (default: "default")
    - **use_rag**: Whether to retrieve from document knowledge base (default: true)
    """
    try:
        processor = get_processor()
        response = await processor.process_text(
            text=request.message,
            session_id=request.session_id,
            use_rag=request.use_rag,
        )
        return response
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to ingest into RAG knowledge base"),
):
    """
    Upload a PDF document to the RAG knowledge base.
    
    The document will be:
    1. Parsed and text-extracted
    2. Split into overlapping chunks
    3. Embedded using HuggingFace sentence-transformers
    4. Stored in ChromaDB for semantic search
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    try:
        content = await file.read()
        rag = get_rag_engine()
        chunks = await rag.ingest_pdf(content, file.filename)

        return DocumentUploadResponse(
            filename=file.filename,
            chunks_created=chunks,
            status=ProcessingStatus.COMPLETED,
            message=f"Successfully ingested {file.filename} ({chunks} chunks created)",
        )
    except Exception as e:
        logger.error("upload_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/chat/voice", response_model=ChatResponse)
async def chat_voice(
    audio: UploadFile = File(..., description="Audio file (ogg, mp3, wav, m4a)"),
    session_id: str = Form(default="default"),
):
    """
    Voice chat endpoint.
    
    Upload an audio file to be transcribed by Whisper (via HuggingFace) and processed as text.
    Supported formats: ogg, mp3, wav, m4a, webm.
    """
    allowed_types = ["audio/ogg", "audio/mpeg", "audio/wav", "audio/mp4", "audio/webm"]
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio.content_type}. Supported: {allowed_types}",
        )

    try:
        audio_data = await audio.read()
        processor = get_processor()
        response = await processor.process_voice(
            audio_data=audio_data,
            session_id=session_id,
            filename=audio.filename or "audio.ogg",
        )
        return response
    except Exception as e:
        logger.error("voice_chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")


@router.post("/chat/image", response_model=ChatResponse)
async def chat_image(
    image: UploadFile = File(..., description="Image file (jpg, png, gif, webp)"),
    caption: str = Form(default="", description="Optional question about the image"),
    session_id: str = Form(default="default"),
):
    """
    Image analysis endpoint.
    
    Upload an image for AI vision analysis (BLIP + LLM pipeline).
    Optionally include a caption/question about the image.
    """
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {image.content_type}. Supported: {allowed_types}",
        )

    try:
        image_data = await image.read()
        processor = get_processor()
        response = await processor.process_image(
            image_data=image_data,
            session_id=session_id,
            caption=caption or None,
            mime_type=image.content_type or "image/jpeg",
        )
        return response
    except Exception as e:
        logger.error("image_chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
