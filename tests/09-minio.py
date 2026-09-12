from src.storage import s3_client, MINIO_BUCKET


response = s3_client.list_objects_v2(
    Bucket=MINIO_BUCKET
)

print("Connected to MinIO successfully!")
print("Bucket:", MINIO_BUCKET)
print("Objects:", response.get("Contents", []))