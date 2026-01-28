"""
ElevenLabs service for audio processing.

Provides text-to-speech and speech-to-text capabilities using ElevenLabs API.
"""
import logging
from typing import List, Optional, Dict, Any, AsyncIterator
from pathlib import Path
import io

from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from elevenlabs import save

from app.core.config import settings

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_STT_MODEL = "scribe_v1"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - natural male voice
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsService:
    """Service for ElevenLabs text-to-speech and speech-to-text."""

    def __init__(self):
        """Initialize the ElevenLabs service."""
        if not settings.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs service")

        self._sync_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        self._async_client = AsyncElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    @property
    def client(self) -> ElevenLabs:
        """Get synchronous client."""
        return self._sync_client

    @property
    def async_client(self) -> AsyncElevenLabs:
        """Get asynchronous client."""
        return self._async_client

    # ==================== Speech-to-Text ====================

    async def transcribe_audio(
        self,
        audio_data: bytes,
        filename: str = "audio.mp3",
        model_id: str = DEFAULT_STT_MODEL,
        language_code: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes
            filename: Original filename for content type detection
            model_id: STT model to use (default: scribe_v1)
            language_code: Optional ISO language code (auto-detected if not provided)
            diarize: Whether to identify different speakers
            num_speakers: Expected number of speakers (helps with diarization)

        Returns:
            Dict with transcription results:
                - text: Transcribed text
                - language: Detected language code
                - duration: Audio duration in seconds
                - segments: List of timed segments (if available)
                - words: List of timed words (if available)
        """
        try:
            # Create file-like object from bytes
            audio_file = io.BytesIO(audio_data)
            audio_file.name = filename

            # Call the API
            result = await self._async_client.speech_to_text.convert(
                file=audio_file,
                model_id=model_id,
                language_code=language_code,
                diarize=diarize,
                num_speakers=num_speakers,
            )

            # Build response
            response = {
                "text": result.text if hasattr(result, "text") else str(result),
                "language": getattr(result, "language_code", None) or getattr(result, "language", None),
                "duration": getattr(result, "duration", None),
                "model": model_id,
            }

            # Add segments if available
            if hasattr(result, "segments") and result.segments:
                response["segments"] = [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                    }
                    for seg in result.segments
                ]

            # Add words if available
            if hasattr(result, "words") and result.words:
                response["words"] = [
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                    }
                    for word in result.words
                ]

            logger.info(f"Transcribed audio: {len(response['text'])} characters")
            return response

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise

    async def transcribe_file(
        self,
        file_path: str,
        model_id: str = DEFAULT_STT_MODEL,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file from path.

        Args:
            file_path: Path to audio file
            model_id: STT model to use
            **kwargs: Additional arguments for transcribe_audio

        Returns:
            Transcription results dict
        """
        path = Path(file_path)
        with open(path, "rb") as f:
            audio_data = f.read()

        return await self.transcribe_audio(
            audio_data=audio_data,
            filename=path.name,
            model_id=model_id,
            **kwargs,
        )

    # ==================== Text-to-Speech ====================

    async def text_to_speech(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_TTS_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert
            voice_id: Voice ID to use
            model_id: TTS model to use
            output_format: Audio output format

        Returns:
            Audio bytes
        """
        try:
            audio = await self._async_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
            )

            # Collect all chunks into bytes
            audio_bytes = b""
            async for chunk in audio:
                audio_bytes += chunk

            logger.info(f"Generated TTS audio: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise

    async def text_to_speech_stream(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_TTS_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech audio.

        Args:
            text: Text to convert
            voice_id: Voice ID to use
            model_id: TTS model to use
            output_format: Audio output format

        Yields:
            Audio chunks as bytes
        """
        try:
            audio_stream = await self._async_client.text_to_speech.stream(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
            )

            async for chunk in audio_stream:
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream speech: {e}")
            raise

    def text_to_speech_sync(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_TTS_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> bytes:
        """
        Synchronous text to speech conversion.

        Args:
            text: Text to convert
            voice_id: Voice ID to use
            model_id: TTS model to use
            output_format: Audio output format

        Returns:
            Audio bytes
        """
        try:
            audio = self._sync_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
            )

            # Collect all chunks into bytes
            audio_bytes = b"".join(audio)

            logger.info(f"Generated TTS audio (sync): {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"Failed to generate speech (sync): {e}")
            raise

    def save_audio(self, audio: bytes, file_path: str) -> None:
        """
        Save audio bytes to file.

        Args:
            audio: Audio bytes
            file_path: Path to save file
        """
        with open(file_path, "wb") as f:
            f.write(audio)
        logger.info(f"Saved audio to: {file_path}")

    # ==================== Voice Management ====================

    async def list_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voices.

        Returns:
            List of voice information dicts
        """
        try:
            response = await self._async_client.voices.search()

            voices = []
            for voice in response.voices:
                voices.append({
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": getattr(voice, "category", None),
                    "description": getattr(voice, "description", None),
                    "labels": getattr(voice, "labels", {}),
                })

            return voices

        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise

    async def get_voice(self, voice_id: str) -> Dict[str, Any]:
        """
        Get details for a specific voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            Voice information dict
        """
        try:
            voice = await self._async_client.voices.get(voice_id=voice_id)

            return {
                "voice_id": voice.voice_id,
                "name": voice.name,
                "category": getattr(voice, "category", None),
                "description": getattr(voice, "description", None),
                "labels": getattr(voice, "labels", {}),
                "settings": getattr(voice, "settings", None),
            }

        except Exception as e:
            logger.error(f"Failed to get voice {voice_id}: {e}")
            raise

    # ==================== Model Information ====================

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dicts
        """
        try:
            models = await self._async_client.models.list()

            return [
                {
                    "model_id": model.model_id,
                    "name": getattr(model, "name", model.model_id),
                    "description": getattr(model, "description", None),
                    "can_do_text_to_speech": getattr(model, "can_do_text_to_speech", None),
                    "can_do_voice_conversion": getattr(model, "can_do_voice_conversion", None),
                    "languages": getattr(model, "languages", []),
                }
                for model in models
            ]

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise


# Singleton instance
_elevenlabs_service: Optional[ElevenLabsService] = None


def get_elevenlabs_service() -> ElevenLabsService:
    """Get or create ElevenLabs service singleton."""
    global _elevenlabs_service
    if _elevenlabs_service is None:
        _elevenlabs_service = ElevenLabsService()
    return _elevenlabs_service
