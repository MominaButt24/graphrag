import boto3

from src.config.settings import settings

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
)


def upload_file(local_path: str, document_id: str, filename: str) -> str:
    """
    Upload original document to MinIO.
    Returns the object key.
    """

    object_key = f"{document_id}/original/{filename}"

    s3_client.upload_file(
        local_path,
        settings.minio_bucket,
        object_key,
    )

    return object_key