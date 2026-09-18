import json
import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


def plan_channel(run_id: str) -> str:
    return f"graphrag:plan:{run_id}"


def publish_plan_event(
    run_id: str,
    event_type: str,
    **payload,
):
    event = {
        "type": event_type,
        **payload,
    }

    redis_client.publish(
        plan_channel(run_id),
        json.dumps(event),
    )