
import os

import gradio as gr
import httpx


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8002/api/v1/query/agent",
)


def query_graphrag(
    message: str,
    history: list,
    request: gr.Request,
) -> str:
    """Send a question to GraphRAG and return a readable answer."""

    if not message or not message.strip():
        return "Please enter a question."

    # Keep each Gradio session's conversation separate.
    thread_id = f"gradio-{request.session_hash}"

    try:
        response = httpx.post(
            API_URL,
            json={
                "question": message.strip(),
                "thread_id": thread_id,
            },
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()

        # Your API returns the answer inside data["answer"]["answer"].
        result = data.get("answer", data)

        if isinstance(result, dict):
            answer = result.get("answer", "No answer was returned.")
            retrieval = result.get("retrieval") or {}
        else:
            answer = str(result)
            retrieval = data.get("retrieval") or {}

        # Add source information when available.
        sources = []
        if isinstance(retrieval, dict):
            vector_results = retrieval.get("vector_results", [])

            for item in vector_results[:5]:
                filename = item.get("filename", "Unknown source")
                page = item.get("page_number")

                label = f"- {filename}"
                if page is not None:
                    label += f" (page {page})"

                if label not in sources:
                    sources.append(label)

        if sources:
            answer += "\n\n### Retrieved sources\n" + "\n".join(sources)

        return str(answer)

    except httpx.TimeoutException:
        return "The GraphRAG request timed out. Please try again."

    except httpx.HTTPStatusError as exc:
        return (
            f"GraphRAG API returned HTTP {exc.response.status_code}.\n\n"
            f"{exc.response.text[:1000]}"
        )

    except Exception as exc:
        return f"Could not get an answer: {exc}"


demo = gr.ChatInterface(
    fn=query_graphrag,
    type="messages",
    title="GraphRAG Assistant",
    description="Ask questions about your knowledge base.",
    textbox=gr.Textbox(
        placeholder="Ask GraphRAG a question...",
        lines=2,
    ),
)
