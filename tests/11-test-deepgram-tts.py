from src.voice.tts import synthesize_speech


output_path = "test_tts.mp3"

synthesize_speech(
    "Hello, this is a test of Flux text to speech.",
    output_path,
)

print(f"TTS audio saved to: {output_path}")