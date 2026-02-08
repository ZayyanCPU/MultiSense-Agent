"""
WhatsApp Service - Handles WhatsApp Cloud API interactions.
Sends/receives messages, downloads media, and manages webhooks.
"""

import httpx
import structlog
from typing import Optional, Dict, Any

from app.config import get_settings
from app.models import WhatsAppMessage, MessageType

logger = structlog.get_logger(__name__)


class WhatsAppService:
    """
    WhatsApp Business Cloud API service.
    
    Handles:
    - Sending text messages
    - Sending media messages (images, documents, audio)
    - Downloading media from WhatsApp servers
    - Parsing incoming webhook payloads
    - Marking messages as read
    """

    def __init__(self):
        settings = get_settings()
        self.api_token = settings.whatsapp_api_token
        self.phone_number_id = settings.whatsapp_phone_number_id
        self.api_version = settings.whatsapp_api_version
        self.verify_token = settings.whatsapp_verify_token
        self.base_url = f"https://graph.facebook.com/{self.api_version}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def send_text_message(self, to: str, message: str) -> Dict[str, Any]:
        """
        Send a text message via WhatsApp.
        
        Args:
            to: Recipient phone number (with country code)
            message: Text message body
            
        Returns:
            WhatsApp API response dict
        """
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("whatsapp_message_sent", to=to, message_id=result.get("messages", [{}])[0].get("id"))
            return result

    async def send_reaction(self, to: str, message_id: str, emoji: str = "👍") -> Dict[str, Any]:
        """Send a reaction to a message."""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "reaction",
            "reaction": {"message_id": message_id, "emoji": emoji},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()

    async def mark_as_read(self, message_id: str) -> None:
        """Mark a message as read (blue ticks)."""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json=payload, headers=self.headers, timeout=10)
                logger.debug("message_marked_read", message_id=message_id)
        except Exception as e:
            logger.warning("mark_read_failed", message_id=message_id, error=str(e))

    async def download_media(self, media_id: str) -> bytes:
        """
        Download media from WhatsApp servers.
        
        Two-step process:
        1. Get the media URL from the media ID
        2. Download the actual media content
        
        Args:
            media_id: WhatsApp media ID
            
        Returns:
            Raw media bytes
        """
        # Step 1: Get media URL
        url = f"{self.base_url}/{media_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            media_info = response.json()
            media_url = media_info["url"]

            # Step 2: Download media content
            media_response = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=60,
            )
            media_response.raise_for_status()
            logger.info("media_downloaded", media_id=media_id, size=len(media_response.content))
            return media_response.content

    def parse_webhook_payload(self, payload: Dict[str, Any]) -> Optional[WhatsAppMessage]:
        """
        Parse an incoming WhatsApp webhook payload into a structured message.
        
        Args:
            payload: Raw webhook JSON payload
            
        Returns:
            Parsed WhatsAppMessage or None if not a valid user message
        """
        try:
            entry = payload.get("entry", [])
            if not entry:
                return None

            changes = entry[0].get("changes", [])
            if not changes:
                return None

            value = changes[0].get("value", {})
            messages = value.get("messages", [])
            if not messages:
                return None

            msg = messages[0]
            msg_type = msg.get("type", "text")

            # Map WhatsApp message types to our enum
            type_mapping = {
                "text": MessageType.TEXT,
                "audio": MessageType.VOICE,
                "image": MessageType.IMAGE,
                "document": MessageType.DOCUMENT,
            }

            parsed = WhatsAppMessage(
                message_id=msg.get("id", ""),
                from_number=msg.get("from", ""),
                timestamp=msg.get("timestamp", ""),
                message_type=type_mapping.get(msg_type, MessageType.TEXT),
            )

            # Extract type-specific fields
            if msg_type == "text":
                parsed.text = msg.get("text", {}).get("body", "")
            elif msg_type == "audio":
                audio = msg.get("audio", {})
                parsed.media_id = audio.get("id")
                parsed.mime_type = audio.get("mime_type", "audio/ogg")
            elif msg_type == "image":
                image = msg.get("image", {})
                parsed.media_id = image.get("id")
                parsed.mime_type = image.get("mime_type", "image/jpeg")
                parsed.caption = image.get("caption")
            elif msg_type == "document":
                doc = msg.get("document", {})
                parsed.media_id = doc.get("id")
                parsed.mime_type = doc.get("mime_type")
                parsed.caption = doc.get("caption")

            logger.info(
                "webhook_parsed",
                message_id=parsed.message_id,
                from_number=parsed.from_number,
                type=parsed.message_type,
            )
            return parsed

        except Exception as e:
            logger.error("webhook_parse_error", error=str(e), payload_keys=list(payload.keys()))
            return None

    def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """
        Verify WhatsApp webhook subscription.
        
        Args:
            mode: Should be 'subscribe'
            token: Verification token to match
            challenge: Challenge string to return
            
        Returns:
            Challenge string if verified, None otherwise
        """
        if mode == "subscribe" and token == self.verify_token:
            logger.info("webhook_verified")
            return challenge
        logger.warning("webhook_verification_failed", mode=mode)
        return None


# Singleton
_whatsapp_service: Optional[WhatsAppService] = None


def get_whatsapp_service() -> WhatsAppService:
    """Get or create WhatsApp service singleton."""
    global _whatsapp_service
    if _whatsapp_service is None:
        _whatsapp_service = WhatsAppService()
    return _whatsapp_service
