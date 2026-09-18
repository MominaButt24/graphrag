import os

import requests
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set")

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v2/speak"

TTS_MODEL = "flux-alexis-en"


def synthesize_speech(text: str) -> bytes:

    params = {
        "model": TTS_MODEL,
        "encoding": "mp3",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
    }

    response = requests.post(
        DEEPGRAM_TTS_URL,
        params=params,
        headers=headers,
        json=payload,
    )

    response.raise_for_status()

    return response.content