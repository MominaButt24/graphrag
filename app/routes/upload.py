from fastapi import APIRouter, UploadFile, File

from src.services.upload_service import upload_documents

router = APIRouter()


@router.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    return upload_documents(files)
