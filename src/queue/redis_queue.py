import json
import redis
from src.config import settings
from src.config.settings import settings

redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

def enqueue_ingestion_job(job: dict):
    redis_client.rpush(
        settings.redis_queue_name,
        json.dumps(job),
    )

def dequeue_ingestion_job():
    payload = redis_client.lpop(settings.redis_queue_name)
    if payload is None:
        return None
    return json.loads(payload)