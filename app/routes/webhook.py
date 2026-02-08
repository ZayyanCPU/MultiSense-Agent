"""
WhatsApp Webhook routes - Handles incoming WhatsApp messages.
Implements webhook verification and message processing.
"""

from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
import structlog

from app.models import MessageType
from app.services.whatsapp_service import get_whatsapp_service
from app.services.processor import get_processor

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/webhook", tags=["WhatsApp Webhook"])


@router.get("/whatsapp")
async def verify_webhook(
    request: Request,
):
    """
    WhatsApp webhook verification endpoint (GET).
    
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge
    to verify your webhook URL during setup.
    """
    params = request.query_params
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")

    whatsapp = get_whatsapp_service()
    result = whatsapp.verify_webhook(mode, token, challenge)

    if result:
        return PlainTextResponse(content=result, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("/whatsapp")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    WhatsApp message webhook endpoint (POST).
    
    Receives incoming messages from WhatsApp and processes them asynchronously.
    Returns 200 immediately to acknowledge receipt, then processes in background.
    
    Supported message types:
    - text: Direct text messages → RAG-augmented AI response
    - audio: Voice messages → Whisper transcription → AI response
    - image: Photos → HF Vision analysis (BLIP + LLM)
    - document: PDF files → RAG knowledge base ingestion
    """
    payload = await request.json()

    # Parse the incoming message
    whatsapp = get_whatsapp_service()
    message = whatsapp.parse_webhook_payload(payload)

    if message is None:
        # Not a user message (could be status update, etc.)
        return {"status": "ok"}

    # Process message in background to return 200 quickly
    background_tasks.add_task(process_whatsapp_message, message)

    return {"status": "received", "message_id": message.message_id}


async def process_whatsapp_message(message):
    """
    Background task to process an incoming WhatsApp message.
    
    Routes to the appropriate processor based on message type,
    then sends the response back via WhatsApp.
    """
    whatsapp = get_whatsapp_service()
    processor = get_processor()
    session_id = message.from_number  # Use phone number as session ID

    try:
        # Mark message as read (blue ticks)
        await whatsapp.mark_as_read(message.message_id)

        # Send processing reaction
        await whatsapp.send_reaction(message.from_number, message.message_id, "⏳")

        # Route based on message type
        if message.message_type == MessageType.TEXT:
            result = await processor.process_text(
                text=message.text,
                session_id=session_id,
            )

        elif message.message_type == MessageType.VOICE:
            audio_data = await whatsapp.download_media(message.media_id)
            result = await processor.process_voice(
                audio_data=audio_data,
                session_id=session_id,
                filename="voice.ogg",
            )

        elif message.message_type == MessageType.IMAGE:
            image_data = await whatsapp.download_media(message.media_id)
            result = await processor.process_image(
                image_data=image_data,
                session_id=session_id,
                caption=message.caption,
                mime_type=message.mime_type or "image/jpeg",
            )

        elif message.message_type == MessageType.DOCUMENT:
            doc_data = await whatsapp.download_media(message.media_id)
            result = await processor.process_document(
                doc_data=doc_data,
                session_id=session_id,
                filename=message.caption or "document.pdf",
                mime_type=message.mime_type,
            )

        else:
            result = None

        # Send response back via WhatsApp
        if result:
            # WhatsApp has a 4096 char limit per message
            response_text = result.response
            if len(response_text) > 4000:
                # Split into multiple messages
                chunks = [response_text[i : i + 4000] for i in range(0, len(response_text), 4000)]
                for chunk in chunks:
                    await whatsapp.send_text_message(message.from_number, chunk)
            else:
                await whatsapp.send_text_message(message.from_number, response_text)

            # Update reaction to done
            await whatsapp.send_reaction(message.from_number, message.message_id, "✅")

            logger.info(
                "whatsapp_response_sent",
                to=message.from_number,
                type=message.message_type,
                processing_time=result.processing_time,
            )

    except Exception as e:
        logger.error(
            "whatsapp_processing_error",
            message_id=message.message_id,
            type=message.message_type,
            error=str(e),
        )
        # Send error message to user
        try:
            await whatsapp.send_text_message(
                message.from_number,
                "❌ Sorry, I encountered an error processing your message. Please try again.",
            )
            await whatsapp.send_reaction(message.from_number, message.message_id, "❌")
        except Exception:
            pass
