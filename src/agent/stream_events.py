from typing import Any

from src.queue.plan_events import publish_plan_event


def parse_agent_stream(question: str, run_id: str | None = None):
    """
    Run the Deep Agent and convert LangGraph stream
    events into UI-friendly events.

    If run_id is provided, publish each event to Redis.
    """
    from src.agent.agent import agent

    stream = agent.stream(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        stream_mode="updates",
    )

    for chunk in stream:
        for event in _parse_chunk(chunk):
            # Keep terminal event local
            # and publish everything when run_id exists.
            if run_id:
                publish_plan_event(
                    run_id,
                    event["type"],
                    **{
                        key: value
                        for key, value in event.items()
                        if key != "type"
                    },
                )

            yield event


def _parse_chunk(chunk: dict[str, Any]):
    """
    Convert one Deep Agent stream chunk
    into our clean event format.
    """

    # -------------------------
    # TODO / PLAN EVENTS
    # -------------------------
    if "tools" in chunk:
        data = chunk["tools"]

        todos = data.get("todos")

        if todos is not None:
            yield {
                "type": "todo_updated",
                "todos": todos,
            }

        for message in data.get("messages", []):
            name = getattr(message, "name", None)

            if name == "write_todos":
                continue

            if name:
                yield {
                    "type": "tool_completed",
                    "tool": name,
                    "tool_call_id": getattr(
                        message,
                        "tool_call_id",
                        None,
                    ),
                }

    # -------------------------
    # MODEL EVENTS
    # -------------------------
    if "model" in chunk:
        for message in chunk["model"].get("messages", []):

            # Tool calls = tool started
            for call in getattr(message, "tool_calls", []):
                tool_name = call.get("name")

                if tool_name == "write_todos":
                    continue

                yield {
                    "type": "tool_started",
                    "tool": tool_name,
                    "tool_call_id": call.get("id"),
                    "args": call.get("args", {}),
                }

            # Final model response
            response_metadata = getattr(
                message,
                "response_metadata",
                {},
            )

            if response_metadata.get("stop_reason") == "end_turn":
                yield {
                    "type": "finished",
                }