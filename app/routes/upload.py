from fastapi import APIRouter, UploadFile, File

from src.services.upload_service import upload_document

router = APIRouter()


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    return upload_document(file)
