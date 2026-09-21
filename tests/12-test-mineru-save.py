from pathlib import Path

from src.preprocessing.mineru_processor import (
    run_mineru,
    clean_mineru_markdown,
)


PDF_PATH = "/home/momina/Downloads/Physics_9th_Ch1_Physical_Quantities.pdf"
OUTPUT_DIR = "data/mineru_test"


output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

print("[TEST] Running MinerU...")

markdown_path = run_mineru(
    PDF_PATH,
    OUTPUT_DIR,
)

print(f"[TEST] Original Markdown: {markdown_path}")

markdown = markdown_path.read_text(
    encoding="utf-8"
)

cleaned = clean_mineru_markdown(markdown)

cleaned_path = output_dir / "Physics_9th_Ch1_cleaned.md"

cleaned_path.write_text(
    cleaned,
    encoding="utf-8",
)

print(f"[TEST] Cleaned Markdown: {cleaned_path}")
print(f"[TEST] Characters: {len(cleaned)}")
print(f"[TEST] Base64 remaining: {'data:image' in cleaned}")