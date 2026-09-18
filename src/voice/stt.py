import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set")


DEEPGRAM_STT_URL = "https://api.deepgram.com/v1/listen"


def transcribe_audio(audio_path: str | Path) -> str:
    """
    Transcribe an audio file using Deepgram Nova-3.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Transcribed text.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    content_type = _get_content_type(audio_path)

    params = {
        "model": "nova-3",
        "smart_format": "true",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type,
    }

    with open(audio_path, "rb") as audio:
        response = requests.post(
            DEEPGRAM_STT_URL,
            params=params,
            headers=headers,
            data=audio,
        )

    response.raise_for_status()

    result = response.json()

    return (
        result["results"]
        ["channels"][0]
        ["alternatives"][0]
        ["transcript"]
    )


def _get_content_type(audio_path: Path) -> str:
    """Return the MIME type expected by Deepgram."""

    extension = audio_path.suffix.lower()

    content_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }

    return content_types.get(
        extension,
        "application/octet-stream",
    )