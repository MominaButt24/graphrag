import json
import os

import redis.asyncio as aioredis


def plan_channel(run_id: str) -> str:
    return f"graphrag:plan:{run_id}"


async def subscribe_plan_events(
    run_id: str,
    ready_event=None,
):
    """
    Subscribe to Deep Agent plan events for one run.

    Yields parsed event dictionaries.
    This layer knows about Redis only.
    It does NOT know anything about Chainlit.
    """

    redis_client = aioredis.from_url(
        os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        ),
        decode_responses=True,
    )

    pubsub = redis_client.pubsub()

    channel = plan_channel(run_id)

    await pubsub.subscribe(channel)

    if ready_event is not None:
        ready_event.set()

    try:
        async for message in pubsub.listen():

            if message["type"] != "message":
                continue

            event = json.loads(message["data"])

            yield event

            if event.get("type") == "finished":
                break

    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_client.close()