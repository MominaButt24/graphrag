from fastapi import APIRouter

from src.services.document_service import get_documents

router = APIRouter()


@router.get("/documents")
def get_documents_route():
    return get_documents()
