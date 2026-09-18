# import os
# from pathlib import Path

# import requests
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("DEEPGRAM_API_KEY")

# if not API_KEY:
#     raise RuntimeError("DEEPGRAM_API_KEY is not set")


# audio_file = Path(__file__).parent / "test.mp3"

# url = "https://api.deepgram.com/v1/listen"

# params = {
#     "model": "nova-3",
#     "smart_format": "true",
# }

# headers = {
#     "Authorization": f"Token {API_KEY}",
#     "Content-Type": "audio/mpeg",
# }


# with open(audio_file, "rb") as audio:
#     response = requests.post(
#         url,
#         params=params,
#         headers=headers,
#         data=audio,
#     )

# response.raise_for_status()

# result = response.json()

# transcript = (
#     result["results"]
#     ["channels"][0]
#     ["alternatives"][0]
#     ["transcript"]
# )

# print("\nTranscript:")
# print(transcript)
from src.voice.stt import transcribe_audio

audio_path = "test.mp3"

text = transcribe_audio(audio_path)

print("\nTranscript:")
print(text)