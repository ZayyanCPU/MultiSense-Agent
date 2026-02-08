"""
Conversation Memory Service - In-memory conversation history management.
Tracks multi-turn conversations per session/user.
"""

import structlog
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from app.models import ConversationTurn, ConversationHistory, MessageType
from app.config import get_settings

logger = structlog.get_logger(__name__)


class MemoryService:
    """
    In-memory conversation history manager.
    
    Features:
    - Per-session conversation tracking
    - Configurable history length limits
    - TTL-based automatic cleanup
    - Format conversion for chat model message format
    """

    def __init__(self):
        settings = get_settings()
        self._conversations: Dict[str, ConversationHistory] = {}
        self.max_history = settings.max_conversation_history
        self.ttl_hours = settings.conversation_ttl_hours

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
    ) -> None:
        """
        Add a conversation turn.
        
        Args:
            session_id: Session/user identifier
            role: 'user' or 'assistant'
            content: Message content
            message_type: Type of message
        """
        if session_id not in self._conversations:
            self._conversations[session_id] = ConversationHistory(session_id=session_id)

        conv = self._conversations[session_id]
        conv.turns.append(
            ConversationTurn(role=role, content=content, message_type=message_type)
        )
        conv.updated_at = datetime.utcnow()

        # Trim to max history
        if len(conv.turns) > self.max_history:
            conv.turns = conv.turns[-self.max_history:]

        logger.debug("memory_turn_added", session_id=session_id, role=role, total_turns=len(conv.turns))

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Get conversation history for a session."""
        self._cleanup_expired()
        conv = self._conversations.get(session_id)
        if conv is None:
            return []
        return conv.turns

    def get_chat_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in chat completion message format.
        Compatible with HuggingFace, OpenAI, and other chat APIs.
        
        Returns:
            List of dicts with 'role' and 'content' keys
        """
        turns = self.get_history(session_id)
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.info("memory_session_cleared", session_id=session_id)
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        self._cleanup_expired()
        return list(self._conversations.keys())

    def _cleanup_expired(self) -> None:
        """Remove conversations that have exceeded TTL."""
        cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        expired = [
            sid
            for sid, conv in self._conversations.items()
            if conv.updated_at < cutoff
        ]
        for sid in expired:
            del self._conversations[sid]
            logger.info("memory_session_expired", session_id=sid)


# Singleton
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
