"""
Hugging Face Inference Service - Handles all AI interactions via free HF Inference API.
Supports text completion, vision (image-to-text), speech-to-text, and embeddings
using open-source models — completely FREE with a HuggingFace account.
"""

import io
import base64
import structlog
from huggingface_hub import AsyncInferenceClient
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = structlog.get_logger(__name__)


class HFService:
    """
    Centralized Hugging Face Inference API interaction service.

    Uses the FREE HuggingFace Inference API (serverless) with open-source models:
    - Chat: Mistral-7B-Instruct (or any HF chat model)
    - Vision: BLIP image captioning + LLM follow-up
    - Audio: Whisper-large-v3 for speech-to-text
    - Embeddings: sentence-transformers for vector embeddings
    """

    def __init__(self):
        settings = get_settings()
        self.client = AsyncInferenceClient(token=settings.hf_api_token)
        self.chat_model = settings.hf_chat_model
        self.embedding_model = settings.hf_embedding_model
        self.whisper_model = settings.hf_whisper_model
        self.vision_model = settings.hf_vision_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a chat completion response via HF Inference API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system-level instruction
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        logger.info(
            "hf_chat_completion",
            model=self.chat_model,
            message_count=len(full_messages),
        )

        response = await self.client.chat_completion(
            messages=full_messages,
            model=self.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def vision_analysis(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail. What do you see?",
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Analyze an image using a two-stage HF pipeline:
            1. Image-to-Text model generates a descriptive caption
            2. Chat LLM refines or answers user-specific questions about the image

        Args:
            image_data: Raw image bytes
            prompt: Question/instruction about the image
            mime_type: Image MIME type

        Returns:
            Image analysis text
        """
        logger.info("hf_vision_analysis", image_size=len(image_data))

        # ── Stage 1: Generate image caption using vision model ──
        caption_result = await self.client.image_to_text(
            image=image_data,
            model=self.vision_model,
        )

        # Handle different response formats
        if hasattr(caption_result, "generated_text"):
            description = caption_result.generated_text
        elif isinstance(caption_result, str):
            description = caption_result
        elif isinstance(caption_result, list) and len(caption_result) > 0:
            item = caption_result[0]
            description = item.get("generated_text", str(item)) if isinstance(item, dict) else str(item)
        else:
            description = str(caption_result)

        # ── Stage 2: Enhance with chat model if user has a specific question ──
        default_prompts = {
            "Describe this image in detail. What do you see?",
            "Analyze this image in detail. Describe what you see, any text visible, and any notable elements.",
        }

        if prompt not in default_prompts:
            enhanced = await self.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"I have an image that was analyzed by a vision model. "
                            f"The vision model described it as: \"{description}\"\n\n"
                            f"Based on this description, please answer the user's question: {prompt}"
                        ),
                    }
                ],
                temperature=0.5,
            )
            return f"**Image Description:** {description}\n\n**Analysis:** {enhanced}"

        return description

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def transcribe_audio(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str:
        """
        Transcribe audio using HF Automatic Speech Recognition (Whisper).

        Args:
            audio_data: Raw audio bytes
            filename: Original filename with extension

        Returns:
            Transcribed text
        """
        logger.info("hf_whisper_transcription", audio_size=len(audio_data))

        result = await self.client.automatic_speech_recognition(
            audio=audio_data,
            model=self.whisper_model,
        )

        # Handle different response formats
        if hasattr(result, "text"):
            return result.text
        elif isinstance(result, dict):
            return result.get("text", str(result))
        elif isinstance(result, str):
            return result
        return str(result)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using HF sentence-transformers via Inference API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info("hf_embeddings", text_count=len(texts))

        embeddings = []
        for text in texts:
            result = await self.client.feature_extraction(
                text=text,
                model=self.embedding_model,
            )
            embedding = self._normalize_embedding(result)
            embeddings.append(embedding)

        return embeddings

    @staticmethod
    def _normalize_embedding(result) -> List[float]:
        """
        Normalize embedding result to a flat 1-D list of floats.

        HF feature_extraction can return:
        - 1D list: already pooled sentence embedding  → use as-is
        - 2D list: per-token embeddings                → mean-pool
        - 3D list: batch of per-token embeddings       → take first, mean-pool
        - numpy array                                  → convert to list
        """
        # Convert numpy to list if needed
        if hasattr(result, "tolist"):
            result = result.tolist()

        if not isinstance(result, list) or len(result) == 0:
            return list(result) if hasattr(result, "__iter__") else [float(result)]

        # 3D → take first element → 2D
        if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
            result = result[0]

        # 2D → mean-pool across tokens
        if isinstance(result[0], list):
            num_tokens = len(result)
            dim = len(result[0])
            return [
                sum(result[t][d] for t in range(num_tokens)) / num_tokens
                for d in range(dim)
            ]

        # 1D → already a flat embedding
        return [float(x) for x in result]


# ── Singleton ────────────────────────────────────────────

_hf_service: Optional[HFService] = None


def get_hf_service() -> HFService:
    """Get or create HuggingFace service singleton."""
    global _hf_service
    if _hf_service is None:
        _hf_service = HFService()
    return _hf_service
