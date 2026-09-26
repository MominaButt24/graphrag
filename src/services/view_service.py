from pathlib import Path

from fastapi import HTTPException

from src.storage.milvus_data_layer import get_document
from src.storage.storage import get_presigned_url


def get_content_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    return content_types.get(
        extension,
        "application/octet-stream",
    )


def view_document(document_id: str):
    document = get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    filename = document["filename"]

    original_key = (
        f"{document_id}/original/{filename}"
    )

    processed_filename = (
        f"{Path(filename).stem}.md"
    )

    processed_key = (
        f"{document_id}/processed/{processed_filename}"
    )

    try:
        original_url = get_presigned_url(
            original_key,
            content_type=get_content_type(filename),
            content_disposition="inline",
        )

        processed_markdown_url = get_presigned_url(
            processed_key,
            content_type="text/markdown; charset=utf-8",
            content_disposition="inline",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate document URLs: {str(e)}",
        )

    return {
        "status": "ok",
        "document_id": document_id,
        "filename": filename,
        "original_url": original_url,
        "processed_markdown_url": processed_markdown_url,
    }