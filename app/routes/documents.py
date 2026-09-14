from fastapi import APIRouter

from src.services.document_service import get_documents, delete_document

router = APIRouter()


@router.get("/documents")
def get_documents_route():
    return get_documents()


@router.delete("/documents/{document_id}")
def delete_document_route(document_id: str):
    return delete_document(document_id)
