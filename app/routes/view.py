from fastapi import APIRouter

from src.services.view_service import view_document


router = APIRouter()


@router.get("/documents/{document_id}/view")
def view_document_route(document_id: str):
    return view_document(document_id)