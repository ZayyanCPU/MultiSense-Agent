"""
Multi-Modal Processor - Central orchestrator for all input types.
Routes text, voice, image, and PDF inputs to appropriate handlers.
Uses FREE HuggingFace Inference API for all AI capabilities.
"""

import time
import structlog
from typing import Optional

from app.models import MessageType, ChatResponse
from app.services.hf_service import get_hf_service
from app.services.rag_engine import get_rag_engine
from app.services.memory_service import get_memory_service

logger = structlog.get_logger(__name__)

# System prompt that defines the agent's personality and capabilities
SYSTEM_PROMPT = """You are MultiSense Agent, an intelligent multi-modal AI assistant. You can:
- Answer questions based on your knowledge
- Analyze images and describe what you see  
- Transcribe and respond to voice messages
- Extract information from PDF documents using RAG (Retrieval Augmented Generation)
- Maintain conversation context across messages

Guidelines:
- Be helpful, accurate, and concise
- When answering from document context, cite the source
- If you don't know something, say so honestly
- For image analysis, be detailed and descriptive
- Format responses clearly with bullet points or paragraphs as appropriate
- Keep responses under 1500 characters when possible (WhatsApp message limit friendly)

You are powered by open-source AI models via the HuggingFace Inference API.
"""


class MultiModalProcessor:
    """
    Central processor that orchestrates all multi-modal interactions.
    
    Routing logic:
    - TEXT → RAG context retrieval + LLM completion (HF chat model)
    - VOICE → Whisper ASR transcription → TEXT pipeline
    - IMAGE → HF image-to-text + LLM follow-up
    - DOCUMENT (PDF) → RAG ingestion + confirmation
    """

    def __init__(self):
        self.hf = get_hf_service()
        self.rag = get_rag_engine()
        self.memory = get_memory_service()

    async def process_text(
        self,
        text: str,
        session_id: str,
        use_rag: bool = True,
    ) -> ChatResponse:
        """
        Process a text message with optional RAG augmentation.
        
        Args:
            text: User's text message
            session_id: Conversation session ID
            use_rag: Whether to retrieve context from vector store
            
        Returns:
            ChatResponse with generated reply
        """
        start_time = time.time()
        sources = []

        # Add user message to memory
        self.memory.add_turn(session_id, "user", text, MessageType.TEXT)

        # Retrieve RAG context if enabled
        augmented_prompt = text
        if use_rag:
            context, sources = await self.rag.retrieve_context(text)
            if context:
                augmented_prompt = (
                    f"Use the following context to answer the user's question. "
                    f"If the context is not relevant, answer from your general knowledge.\n\n"
                    f"--- Retrieved Context ---\n{context}\n--- End Context ---\n\n"
                    f"User Question: {text}"
                )
                logger.info("rag_context_applied", sources=sources, context_length=len(context))

        # Build conversation messages
        history = self.memory.get_chat_messages(session_id)
        # Replace the last user message with augmented prompt
        if history:
            history[-1]["content"] = augmented_prompt

        # Generate response via HuggingFace
        response = await self.hf.chat_completion(
            messages=history,
            system_prompt=SYSTEM_PROMPT,
        )

        # Store assistant response in memory
        self.memory.add_turn(session_id, "assistant", response, MessageType.TEXT)

        elapsed = time.time() - start_time
        logger.info("text_processed", session_id=session_id, elapsed=round(elapsed, 2))

        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources,
            processing_time=round(elapsed, 2),
            message_type=MessageType.TEXT,
        )

    async def process_voice(
        self,
        audio_data: bytes,
        session_id: str,
        filename: str = "audio.ogg",
    ) -> ChatResponse:
        """
        Process a voice message: transcribe → then process as text.
        
        Args:
            audio_data: Raw audio bytes
            session_id: Conversation session ID
            filename: Audio filename with extension
            
        Returns:
            ChatResponse with transcription + reply
        """
        start_time = time.time()

        # Step 1: Transcribe audio with Whisper via HuggingFace
        transcription = await self.hf.transcribe_audio(audio_data, filename)
        logger.info("voice_transcribed", session_id=session_id, text_length=len(transcription))

        # Step 2: Process transcribed text through the text pipeline
        result = await self.process_text(transcription, session_id)

        elapsed = time.time() - start_time
        result.processing_time = round(elapsed, 2)
        result.message_type = MessageType.VOICE
        result.response = f"🎤 *Transcription:* _{transcription}_\n\n{result.response}"

        return result

    async def process_image(
        self,
        image_data: bytes,
        session_id: str,
        caption: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> ChatResponse:
        """
        Process an image using GPT-4 Vision.
        
        Args:
            image_data: Raw image bytes
            session_id: Conversation session ID
            caption: Optional user caption/question about the image
            mime_type: Image MIME type
            
        Returns:
            ChatResponse with image analysis
        """
        start_time = time.time()

        prompt = caption or "Analyze this image in detail. Describe what you see, any text visible, and any notable elements."

        # Analyze with HuggingFace Vision pipeline (image-to-text + LLM)
        analysis = await self.hf.vision_analysis(
            image_data=image_data,
            prompt=prompt,
            mime_type=mime_type,
        )

        # Store in conversation memory
        self.memory.add_turn(session_id, "user", f"[Image sent] {caption or 'Image analysis requested'}", MessageType.IMAGE)
        self.memory.add_turn(session_id, "assistant", analysis, MessageType.IMAGE)

        elapsed = time.time() - start_time
        logger.info("image_processed", session_id=session_id, elapsed=round(elapsed, 2))

        return ChatResponse(
            response=f"📸 *Image Analysis:*\n\n{analysis}",
            session_id=session_id,
            sources=[],
            processing_time=round(elapsed, 2),
            message_type=MessageType.IMAGE,
        )

    async def process_document(
        self,
        doc_data: bytes,
        session_id: str,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a document (PDF) for RAG ingestion.
        
        Args:
            doc_data: Raw document bytes
            session_id: Conversation session ID
            filename: Document filename
            mime_type: Document MIME type
            
        Returns:
            ChatResponse confirming ingestion
        """
        start_time = time.time()

        if mime_type and "pdf" in mime_type.lower() or filename.lower().endswith(".pdf"):
            chunks = await self.rag.ingest_pdf(doc_data, filename)
            response_text = (
                f"📄 *Document Processed:* _{filename}_\n\n"
                f"✅ Successfully ingested into knowledge base.\n"
                f"📊 Created *{chunks}* searchable chunks.\n\n"
                f"You can now ask me questions about this document!"
            )
        else:
            response_text = (
                f"⚠️ Currently, I only support PDF documents for RAG ingestion.\n"
                f"Received: {mime_type or 'unknown type'}\n\n"
                f"Please send a PDF file to add it to my knowledge base."
            )

        # Store in memory
        self.memory.add_turn(session_id, "user", f"[Document sent: {filename}]", MessageType.DOCUMENT)
        self.memory.add_turn(session_id, "assistant", response_text, MessageType.DOCUMENT)

        elapsed = time.time() - start_time

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=[filename],
            processing_time=round(elapsed, 2),
            message_type=MessageType.DOCUMENT,
        )


# Singleton
_processor: Optional[MultiModalProcessor] = None


def get_processor() -> MultiModalProcessor:
    """Get or create multi-modal processor singleton."""
    global _processor
    if _processor is None:
        _processor = MultiModalProcessor()
    return _processor
