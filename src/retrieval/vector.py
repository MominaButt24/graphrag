from src.storage.milvus_client import get_collection, get_embedder
from src.config.logging import get_logger
import sentry_sdk

from src.generation.llm import get_llm

logger = get_logger(__name__)

llm = get_llm()


def vector_search(question: str, top_k: int = 5) -> list[dict]:
    """Milvus similarity search — the vector-side counterpart to smart_query."""
    span = sentry_sdk.get_current_span()

    collection = get_collection()
    collection.load()

    embedder = get_embedder()
    query_vec = embedder.encode([question]).tolist()

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=[
            "text",
            "filename",
            "source_key",
            "document_id",
            "chunk_id",
            "page_number",
        ],
    )

    hits = [
        {
            "text": hit.entity.get("text"),
            "source": hit.entity.get("source_key"),
            "filename": hit.entity.get("filename"),
            "document_id": hit.entity.get("document_id"),
            "chunk_id": hit.entity.get("chunk_id"),
            "page_number": hit.entity.get("page_number"),
            "score": hit.distance,
            "origin": "vector",
        }
        for hit in results[0]
    ]

    if span:
        span.set_tag("vector_hits", len(hits))

    logger.info(
        f"[vector_search] question='{question}' | hits found: {len(hits)}"
    )

    return hits


def kb_relevance_score(question: str) -> float:
    hits = vector_search(question, top_k=1)
    return hits[0]["score"] if hits else 0.0
