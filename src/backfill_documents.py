"""
One-off backfill: your chat_documents tracking collection only started
getting written to once the docs-sidebar feature was added — anything
ingested before that has no row here, even though it's fully present
in the main graphrag_docs collection. This script reads what's already
there and creates matching chat_documents rows retroactively.

Run once: uv run python -m src.backfill_documents
"""

import time
from collections import Counter
from src.milvus_client import get_collection
from src.milvus_data_layer import get_documents_collection

MAX_QUERY_LIMIT = 16384  # Milvus's hard ceiling on offset + limit combined


def backfill():
    main_collection = get_collection()
    main_collection.load()

    # A single query at Milvus's max limit is enough for a corpus this size
    # (a handful of PDFs, not hundreds of thousands of chunks). If you ever
    # exceed 16384 total chunks, this needs Milvus's query_iterator API
    # instead — offset+limit pagination hits the same ceiling either way.
    rows = main_collection.query(
        expr="id >= 0",
        output_fields=["source"],
        limit=MAX_QUERY_LIMIT,
    )

    counts = Counter(r["source"] for r in rows)
    print(f"Found {len(counts)} distinct source files across {len(rows)} chunks")

    docs_collection = get_documents_collection()
    now = int(time.time())

    for i, (source, chunk_count) in enumerate(counts.items()):
        import os
        filename = os.path.basename(source)
        doc_id = f"backfill-{i}"
        docs_collection.upsert([{
            "id": doc_id,
            "filename": filename,
            "status": "done",
            "chunk_count": chunk_count,
            "uploaded_by": "unknown",  # not recorded pre-backfill, no way to recover this
            "uploaded_at": now,
            "embedding": [0.0] * 8,
        }])
        print(f"  Backfilled: {filename} ({chunk_count} chunks)")

    docs_collection.flush()
    print("Done.")


if __name__ == "__main__":
    backfill()