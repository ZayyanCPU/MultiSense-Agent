"""
Application configuration using pydantic-settings.
Loads from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Hugging Face (FREE Inference API) ---
    hf_api_token: str = ""
    hf_chat_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_whisper_model: str = "openai/whisper-large-v3"
    hf_vision_model: str = "Salesforce/blip-image-captioning-large"

    # --- WhatsApp Cloud API ---
    whatsapp_api_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_business_account_id: str = ""
    whatsapp_verify_token: str = "multisense-verify-token"
    whatsapp_api_version: str = "v21.0"

    # --- n8n ---
    n8n_webhook_url: str = "http://n8n:5678"

    # --- ChromaDB ---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection_name: str = "multisense_documents"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    app_debug: bool = False
    app_log_level: str = "info"

    # --- RAG ---
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5

    # --- Memory ---
    max_conversation_history: int = 20
    conversation_ttl_hours: int = 24

    @property
    def whatsapp_api_url(self) -> str:
        return f"https://graph.facebook.com/{self.whatsapp_api_version}/{self.whatsapp_phone_number_id}/messages"

    @property
    def whatsapp_media_url(self) -> str:
        return f"https://graph.facebook.com/{self.whatsapp_api_version}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
