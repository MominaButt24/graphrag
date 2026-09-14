import os
import uuid
import shutil

from fastapi import HTTPException, UploadFile

from src.config.logging import get_logger
from src.services.errors import map_pipeline_error
from src.storage.milvus_data_layer import start_document, finish_document
from src.storage.storage import upload_file
from src.ingestion.ingest import load_and_chunk, embed_and_ingest
from src.graph.graph_builder import build_graph_from_chunks
from src.graph.community import (
    load_graph_from_neo4j,
    run_leiden,
    write_communities_to_neo4j,
    build_all_community_summaries,
)

logger = get_logger(__name__)


def upload_document(file: UploadFile):
    """Application workflow for a PDF upload request.

    This function owns the document metadata lifecycle and the ingestion
    orchestration. The route file only translates the HTTP request into
    this service call.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(f"[/upload] filename='{file.filename}'")

    os.makedirs("data/raw", exist_ok=True)
    save_path = f"data/raw/{file.filename}"

    doc_id = str(uuid.uuid4())
    uploaded_by = "api-upload"
    uploaded_at = start_document(
        doc_id=doc_id,
        filename=file.filename,
        uploaded_by=uploaded_by,
    )

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        logger.exception("Failed to save uploaded file")
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            0,
            status="failed",
        )
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

    try:
        source_key = upload_file(
            local_path=save_path,
            document_id=doc_id,
            filename=file.filename,
        )
    except Exception as e:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            0,
            status="failed",
        )
        raise map_pipeline_error(e)

    try:
        chunks = load_and_chunk(save_path)
    except Exception as e:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            0,
            status="failed",
        )
        raise map_pipeline_error(e)

    try:
        failed = build_graph_from_chunks(chunks, document_id=doc_id)
    except Exception as e:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            len(chunks),
            status="failed",
        )
        raise map_pipeline_error(e)

    if failed:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            len(chunks),
            status="failed",
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Graph extraction failed for one or more chunks — "
                f"{len(failed)} chunk(s) produced zero graph nodes or relationships."
            ),
        )

    try:
        embed_and_ingest(
            chunks,
            document_id=doc_id,
            filename=file.filename,
            source_key=source_key,
        )
    except Exception as e:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            len(chunks),
            status="failed",
        )
        raise map_pipeline_error(e)

    try:
        G = load_graph_from_neo4j()
        community_map = run_leiden(G)
        write_communities_to_neo4j(community_map)
        build_all_community_summaries()
    except Exception as e:
        finish_document(
            doc_id,
            file.filename,
            uploaded_by,
            uploaded_at,
            len(chunks),
            status="failed",
        )
        raise map_pipeline_error(e)

    finish_document(
        doc_id=doc_id,
        filename=file.filename,
        uploaded_by=uploaded_by,
        uploaded_at=uploaded_at,
        chunk_count=len(chunks),
        status="done",
    )

    try:
        os.remove(save_path)
        logger.info(f"[/upload] deleted temporary local file: {save_path}")
    except Exception:
        logger.exception(f"[/upload] failed to delete temporary file: {save_path}")

    logger.info(
        f"[/upload] completed filename='{file.filename}' "
        f"chunks={len(chunks)}"
    )

    return {
        "status": "processed",
        "filename": file.filename,
        "chunks": len(chunks),
        "failed_chunks": len(failed),
        "document_id": doc_id,
    }
