"""
Tests for MultiSense Agent API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app

client = TestClient(app)


# ─── Root & Health Tests ─────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_project_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["project"] == "MultiSense Agent"
        assert data["version"] == "1.0.0"
        assert "capabilities" in data
        assert "text_chat" in data["capabilities"]


class TestHealthEndpoint:
    @patch("app.routes.health.get_rag_engine")
    @patch("app.routes.health.get_memory_service")
    def test_health_check(self, mock_memory, mock_rag):
        mock_rag_instance = MagicMock()
        mock_rag_instance.get_collection_stats = AsyncMock(
            return_value={"collection_name": "test", "document_count": 0, "status": "connected"}
        )
        mock_rag.return_value = mock_rag_instance

        mock_memory_instance = MagicMock()
        mock_memory_instance.get_active_sessions.return_value = []
        mock_memory.return_value = mock_memory_instance

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# ─── Chat Endpoint Tests ────────────────────────────────

class TestChatEndpoint:
    @patch("app.routes.chat.get_processor")
    def test_chat_text_success(self, mock_processor):
        from app.models import ChatResponse, MessageType

        mock_proc = MagicMock()
        mock_proc.process_text = AsyncMock(
            return_value=ChatResponse(
                response="Hello! I'm MultiSense Agent.",
                session_id="test-session",
                sources=[],
                processing_time=0.5,
                message_type=MessageType.TEXT,
            )
        )
        mock_processor.return_value = mock_proc

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello",
                "session_id": "test-session",
                "use_rag": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test-session"

    def test_chat_empty_message_fails(self):
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "test"},
        )
        assert response.status_code == 422  # Validation error


# ─── WhatsApp Webhook Tests ─────────────────────────────

class TestWhatsAppWebhook:
    @patch("app.routes.webhook.get_whatsapp_service")
    def test_webhook_verification_success(self, mock_wa):
        mock_service = MagicMock()
        mock_service.verify_webhook.return_value = "challenge-token-123"
        mock_wa.return_value = mock_service

        response = client.get(
            "/api/v1/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test-token",
                "hub.challenge": "challenge-token-123",
            },
        )
        assert response.status_code == 200
        assert response.text == "challenge-token-123"

    @patch("app.routes.webhook.get_whatsapp_service")
    def test_webhook_verification_failure(self, mock_wa):
        mock_service = MagicMock()
        mock_service.verify_webhook.return_value = None
        mock_wa.return_value = mock_service

        response = client.get(
            "/api/v1/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong-token",
                "hub.challenge": "challenge-123",
            },
        )
        assert response.status_code == 403

    @patch("app.routes.webhook.get_whatsapp_service")
    def test_webhook_receives_message(self, mock_wa):
        mock_service = MagicMock()
        mock_service.parse_webhook_payload.return_value = None  # Status update, not message
        mock_wa.return_value = mock_service

        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {"phone_number_id": "123"},
                                "statuses": [{"status": "delivered"}],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        response = client.post("/api/v1/webhook/whatsapp", json=payload)
        assert response.status_code == 200


# ─── Model Tests ─────────────────────────────────────────

class TestModels:
    def test_message_type_enum(self):
        from app.models import MessageType

        assert MessageType.TEXT == "text"
        assert MessageType.VOICE == "voice"
        assert MessageType.IMAGE == "image"
        assert MessageType.DOCUMENT == "document"

    def test_chat_request_validation(self):
        from app.models import ChatRequest

        req = ChatRequest(message="Hello", session_id="test")
        assert req.message == "Hello"
        assert req.use_rag is True  # Default

    def test_conversation_turn(self):
        from app.models import ConversationTurn

        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"


# ─── Memory Service Tests ───────────────────────────────

class TestMemoryService:
    def test_add_and_retrieve(self):
        from app.services.memory_service import MemoryService

        memory = MemoryService()
        memory.add_turn("session-1", "user", "Hello")
        memory.add_turn("session-1", "assistant", "Hi there!")

        history = memory.get_history("session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_chat_message_format(self):
        from app.services.memory_service import MemoryService

        memory = MemoryService()
        memory.add_turn("session-2", "user", "What is AI?")
        memory.add_turn("session-2", "assistant", "AI is artificial intelligence.")

        messages = memory.get_chat_messages("session-2")
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is AI?"}

    def test_clear_session(self):
        from app.services.memory_service import MemoryService

        memory = MemoryService()
        memory.add_turn("session-3", "user", "Test")
        assert memory.clear_session("session-3") is True
        assert memory.clear_session("nonexistent") is False

    def test_max_history_limit(self):
        from app.services.memory_service import MemoryService

        memory = MemoryService()
        memory.max_history = 5
        for i in range(10):
            memory.add_turn("session-4", "user", f"Message {i}")

        history = memory.get_history("session-4")
        assert len(history) == 5
