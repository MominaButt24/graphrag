import os
import tempfile

from src.config.logging import get_logger
from src.queue.redis_queue import dequeue_ingestion_job
from src.storage.storage import download_file
from src.storage.milvus_data_layer import finish_document
from src.ingestion.ingest import load_and_chunk, embed_and_ingest
from src.graph.graph_builder import build_graph_from_chunks
from src.graph.community import (
    load_graph_from_neo4j,
    run_leiden,
    write_communities_to_neo4j,
    build_all_community_summaries,
)

logger = get_logger(__name__)


def process_ingestion_job(job: dict):
    document_id = job["document_id"]
    filename = job["filename"]
    source_key = job["source_key"]
    uploaded_by = job["uploaded_by"]
    uploaded_at = job["uploaded_at"]

    temp_path = None

    logger.info(
        f"[worker] processing document_id='{document_id}' "
        f"filename='{filename}'"
    )

    try:
        # ---------------------------------------------
        # 1. Download permanent original from MinIO
        # ---------------------------------------------

        temp_dir = tempfile.mkdtemp(prefix="graphrag_")

        temp_path = os.path.join(
            temp_dir,
            filename,
        )

        download_file(
            source_key=source_key,
            local_path=temp_path,
        )

        # ---------------------------------------------
        # 2. Load + chunk
        # ---------------------------------------------

        chunks = load_and_chunk(temp_path)

        # ---------------------------------------------
        # 3. Build graph
        # ---------------------------------------------

        failed = build_graph_from_chunks(
            chunks,
            document_id=document_id,
        )

        if failed:
            raise RuntimeError(
                "Graph extraction failed for "
                f"{len(failed)} chunk(s)."
            )

        # ---------------------------------------------
        # 4. Embed + Milvus
        # ---------------------------------------------

        embed_and_ingest(
            chunks,
            document_id=document_id,
            filename=filename,
            source_key=source_key,
        )

        # ---------------------------------------------
        # 5. Communities
        # ---------------------------------------------

        G = load_graph_from_neo4j()

        community_map = run_leiden(G)

        write_communities_to_neo4j(
            community_map
        )

        build_all_community_summaries()

        # ---------------------------------------------
        # 6. Mark document complete
        # ---------------------------------------------

        finish_document(
            doc_id=document_id,
            filename=filename,
            uploaded_by=uploaded_by,
            uploaded_at=uploaded_at,
            chunk_count=len(chunks),
            status="done",
        )

        logger.info(
            f"[worker] completed document_id='{document_id}' "
            f"chunks={len(chunks)}"
        )

    except Exception:
        logger.exception(
            f"[worker] failed document_id='{document_id}'"
        )

        finish_document(
            doc_id=document_id,
            filename=filename,
            uploaded_by=uploaded_by,
            uploaded_at=uploaded_at,
            chunk_count=0,
            status="failed",
        )

    finally:
        # ---------------------------------------------
        # 7. Delete temporary processing copy
        # ---------------------------------------------

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)

                logger.info(
                    f"[worker] deleted temporary file: {temp_path}"
                )

            except OSError:
                logger.exception(
                    f"[worker] failed to delete: {temp_path}"
                )


def run_worker():
    logger.info("[worker] ingestion worker started")

    while True:
        job = dequeue_ingestion_job()

        if job is None:
            continue

        process_ingestion_job(job)


if __name__ == "__main__":
    run_worker()