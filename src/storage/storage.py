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


def delete_document_files(document_id: str):
    """Remove every object stored under one uploaded document id in MinIO."""
    prefix = f"{document_id}/"
    response = s3_client.list_objects_v2(
        Bucket=settings.minio_bucket,
        Prefix=prefix,
    )
    objects = response.get("Contents", [])
    if not objects:
        return

    keys = [{"Key": obj["Key"]} for obj in objects]
    s3_client.delete_objects(
        Bucket=settings.minio_bucket,
        Delete={"Objects": keys},
    )


def delete_file(object_key: str):
    """Delete one object key from MinIO."""
    s3_client.delete_object(
        Bucket=settings.minio_bucket,
        Key=object_key,
    )