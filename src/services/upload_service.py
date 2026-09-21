import os
import uuid
import shutil

from fastapi import HTTPException, UploadFile

from src.config.logging import get_logger
from src.storage.milvus_data_layer import start_document, finish_document
from src.storage.storage import upload_file
from src.queue.redis_queue import enqueue_ingestion_job

logger = get_logger(__name__)


def upload_documents(files: list[UploadFile]):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided.",
        )

    allowed_extensions = { ".pdf", ".docx", ".pptx", ".xlsx", }

    results = []
    os.makedirs("data/raw", exist_ok=True)

    for file in files:

        extension = os.path.splitext( file.filename )[1].lower() 
        if extension not in allowed_extensions: 
            raise HTTPException( status_code=400, detail=( "Supported file types are: " "PDF, DOCX, PPTX, XLSX. " f"Unsupported file: {file.filename}" ), )

        doc_id = str(uuid.uuid4())
        uploaded_by = "api-upload"

        save_path = os.path.join(
            "data/raw",
            f"{doc_id}_{file.filename}",
        )

        logger.info(
            f"[/upload] receiving filename='{file.filename}' "
            f"document_id='{doc_id}'"
        )

        uploaded_at = start_document(
            doc_id=doc_id,
            filename=file.filename,
            uploaded_by=uploaded_by,
        )

        # --------------------------------------------------
        # 1. Save temporary local copy
        # --------------------------------------------------

        try:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        except Exception:
            logger.exception(
                f"Failed to save uploaded file: {file.filename}"
            )

            finish_document(
                doc_id,
                file.filename,
                uploaded_by,
                uploaded_at,
                0,
                status="failed",
            )

            raise HTTPException(
                status_code=500,
                detail=f"Failed to save {file.filename}.",
            )

        # --------------------------------------------------
        # 2. Upload permanent original to MinIO
        # --------------------------------------------------

        try:
            source_key = upload_file(
                local_path=save_path,
                document_id=doc_id,
                filename=file.filename,
            )

        except Exception:
            logger.exception(
                f"Failed to upload to MinIO: {file.filename}"
            )

            finish_document(
                doc_id,
                file.filename,
                uploaded_by,
                uploaded_at,
                0,
                status="failed",
            )

            # Local file is no longer needed.
            try:
                os.remove(save_path)
            except OSError:
                pass

            raise HTTPException(
                status_code=500,
                detail=f"Failed to store {file.filename}.",
            )

        # --------------------------------------------------
        # 3. Local copy is no longer needed
        # --------------------------------------------------

        try:
            os.remove(save_path)

            logger.info(
                f"[/upload] deleted temporary local file: {save_path}"
            )

        except OSError:
            logger.exception(
                f"Failed to delete temporary local file: {save_path}"
            )

        # --------------------------------------------------
        # 4. Create Redis ingestion job
        # --------------------------------------------------

        job = {
            "document_id": doc_id,
            "filename": file.filename,
            "source_key": source_key,
            "uploaded_by": uploaded_by,
            "uploaded_at": uploaded_at,
        }

        try:
            enqueue_ingestion_job(job)

        except Exception:
            logger.exception(
                f"Failed to enqueue ingestion job: {file.filename}"
            )

            finish_document(
                doc_id,
                file.filename,
                uploaded_by,
                uploaded_at,
                0,
                status="failed",
            )

            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue {file.filename}.",
            )

        results.append(
            {
                "document_id": doc_id,
                "filename": file.filename,
                "status": "queued",
            }
        )

    return {
        "status": "queued",
        "documents": results,
    }