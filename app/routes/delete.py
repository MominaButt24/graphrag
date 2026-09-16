from fastapi import APIRouter

from src.services.delete_service import delete_document

router = APIRouter()


@router.delete("/documents/{document_id}")
def delete_document_route(document_id: str):
    return delete_document(document_id)
