from fastapi import HTTPException

from src.storage.milvus_data_layer import get_document
from src.storage.storage import get_presigned_url


def view_document(document_id: str):
    """Generate a temporary URL for viewing an uploaded document."""

    document = get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    filename = document["filename"]

    object_key = (
        f"{document_id}/original/{filename}"
    )

    try:
        url = get_presigned_url(object_key)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate document URL: {str(e)}",
        )

    return {
        "status": "ok",
        "document_id": document_id,
        "filename": filename,
        "url": url,
    }