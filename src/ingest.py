from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # swap per file type
from src.milvus_client import get_collection, get_embedder

def load_and_chunk(filepath: str, chunk_size: int = 800, chunk_overlap: int = 100):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # smaller than your vanilla RAG chunk size on purpose
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    print(f"{filepath}: {len(docs)} pages -> {len(chunks)} extraction-ready chunks")
    return chunks


# --- Phase 1: Milvus ingestion additions below ---
# load_and_chunk above is untouched — this reuses its output, it doesn't
# replace or touch whatever function feeds chunks into LLMGraphTransformer
# for the graph side. Call this as a separate, additive step.

def embed_and_ingest(chunks, source: str):
    """
    Embeds chunk texts and inserts them into Milvus.
    Call this with the same chunks you already feed to graph_builder.py —
    same source documents, second storage backend, nothing shared or
    overwritten.
    """
    collection = get_collection()
    embedder = get_embedder()

    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    sources = [source] * len(texts)

    collection.insert([texts, sources, embeddings])
    collection.flush()
    print(f"[ingest] {source}: inserted {len(texts)} chunks into Milvus")


def ingest_file_to_milvus(filepath: str):
    """
    Convenience wrapper: chunk a file and push it straight into Milvus.
    Use this for testing the vector side in isolation. For your real
    ingestion run, call embed_and_ingest() with the same chunks you're
    already passing to your graph ingestion step, so both stores see
    identical text.
    """
    chunks = load_and_chunk(filepath)
    embed_and_ingest(chunks, source=filepath)
    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <filepath>")
        sys.exit(1)
    ingest_file_to_milvus(sys.argv[1])