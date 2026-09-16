import json
import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "graphrag:ingestion"

# redis_client = redis.Redis.from_url(
#     REDIS_URL,
#     decode_responses=True,
# )

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

def enqueue_ingestion_job(job: dict):
    redis_client.rpush(
        QUEUE_NAME,
        json.dumps(job),
    )


# def dequeue_ingestion_job():
#     result = redis_client.blpop(QUEUE_NAME, timeout=5)

#     if result is None:
#         return None

#     _, payload = result

#     return json.loads(payload)

def dequeue_ingestion_job():
    payload = redis_client.lpop(QUEUE_NAME)
    if payload is None:
        return None
    return json.loads(payload)