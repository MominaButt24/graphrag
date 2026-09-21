import os
import shutil
import tempfile

from src.config.logging import get_logger
from src.queue.redis_queue import dequeue_ingestion_job
from src.storage.storage import (
    download_file,
    upload_processed_markdown,
)
from src.storage.milvus_data_layer import finish_document
from src.ingestion.ingest import chunk_documents, embed_and_ingest
from src.preprocessing.mineru_processor import mineru_to_documents
from src.preprocessing.document_processor import process_document
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

    temp_dir = None
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

        # # ---------------------------------------------
        # # 2.1. MinerU preprocessing
        # # ---------------------------------------------

        # docs, processed_markdown = mineru_to_documents(
        #     temp_path
        # )

        # # ---------------------------------------------
        # # 2.2. Store processed Markdown in MinIO
        # # ---------------------------------------------

        # processed_md_path = os.path.join(
        #     temp_dir,
        #     f"{os.path.splitext(filename)[0]}.md",
        # )

        # with open(
        #     processed_md_path,
        #     "w",
        #     encoding="utf-8",
        # ) as f:
        #     f.write(processed_markdown)

        # processed_key = upload_processed_markdown(
        #     local_path=processed_md_path,
        #     document_id=document_id,
        #     filename=filename,
        # )

        # logger.info(
        #     f"[worker] uploaded processed Markdown: "
        #     f"{processed_key}"
        # )

        # # ---------------------------------------------
        # # 2.3. Chunk MinerU documents
        # # ---------------------------------------------

        # chunks = chunk_documents(docs)

        # logger.info(
        #     f"[worker] {filename}: "
        #     f"{len(docs)} pages -> "
        #     f"{len(chunks)} chunks"
        # )


        # ---------------------------------------------
        # 2.1. Preprocessing
        # ---------------------------------------------

        processed = process_document(temp_path)

        if processed["type"] == "mineru":
            docs, processed_markdown = mineru_to_documents(
                processed["path"]
            )

        else:
            docs = processed["documents"]

            # Directly extracted formats do not currently
            # have MinerU Markdown output.
            processed_markdown = "\n\n".join(
                doc.page_content.strip()
                for doc in docs
                if doc.page_content.strip()
            )

        # ---------------------------------------------
        # 2.2. Store processed Markdown in MinIO
        # ---------------------------------------------

        processed_md_path = os.path.join(
            temp_dir,
            f"{os.path.splitext(filename)[0]}.md",
        )

        with open(
            processed_md_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(processed_markdown)

        processed_key = upload_processed_markdown(
            local_path=processed_md_path,
            document_id=document_id,
            filename=filename,
        )

        logger.info(
            f"[worker] uploaded processed Markdown: "
            f"{processed_key}"
        )

        # ---------------------------------------------
        # 2.3. Chunk extracted documents
        # ---------------------------------------------

        chunks = chunk_documents(docs)

        logger.info(
            f"[worker] {filename}: "
            f"{len(docs)} documents -> "
            f"{len(chunks)} chunks"
        )



        # ---------------------------------------------
        # 3. Build graph
        # ---------------------------------------------

        failed = build_graph_from_chunks(
            chunks,
            document_id=document_id,
        )

        if failed:
            logger.warning(
                f"[worker] graph extraction skipped/failed for "
                f"{len(failed)} chunk(s), continuing with Milvus ingestion"
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
    # 7. Delete temporary processing directory
    # ---------------------------------------------

        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)

                logger.info(
                    f"[worker] deleted temporary directory: {temp_dir}"
                )

            except OSError:
                logger.exception(
                    f"[worker] failed to delete temporary directory: "
                    f"{temp_dir}"
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