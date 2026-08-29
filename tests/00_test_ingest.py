from src.ingest import load_and_chunk

chunks = load_and_chunk("data/raw/sample1.pdf")
print(f"Total chunks: {len(chunks)}")
print("--- First chunk preview ---")
print(chunks[0].page_content)
print("--- Metadata ---")
print(chunks[0].metadata)