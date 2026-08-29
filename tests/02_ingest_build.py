from src.ingest import load_and_chunk
from src.graph_builder import build_graph_from_chunks

chunks = load_and_chunk("data/raw/sample1.pdf")
print(f"Extracting from {len(chunks)} chunks...")

failed = build_graph_from_chunks(chunks)  # start with just 3 chunks, not the whole file
print("Done. Failed chunks:", len(failed))