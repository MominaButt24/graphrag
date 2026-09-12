import os
import boto3
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")


s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)


def upload_file(local_path: str, document_id: str, filename: str) -> str:
    """
    Upload original document to MinIO.
    Returns the object key.
    """

    object_key = f"{document_id}/original/{filename}"

    s3_client.upload_file(
        local_path,
        MINIO_BUCKET,
        object_key,
    )

    return object_key