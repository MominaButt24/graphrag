import os

import gradio as gr
import requests
from dotenv import load_dotenv
load_dotenv("/workspace/.env", override=True)


API_BASE = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8002/api/v1",
).rstrip("/")


def upload_document(file_path):
    if not file_path:
        return "Please select a file first."

    try:
        with open(file_path, "rb") as file:
            response = requests.post(
                f"{API_BASE}/upload",
                files={
                    "file": (
                        os.path.basename(file_path),
                        file,
                    )
                },
                timeout=300,
            )

        response.raise_for_status()
        return response.text

    except requests.RequestException as exc:
        return f"Upload failed: {exc}"

def ask_question(question):
    if not question or not question.strip():
        return "Please enter a question."

    try:
        response = requests.post(
            f"{API_BASE}/query/agent",
            json={"question": question.strip()},
            timeout=300,
        )

        if not response.ok:
            return (
                f"API error ({response.status_code}):\n"
                f"{response.text}"
            )

        data = response.json()

        # Your response contains answer.answer.
        result = data.get("answer", data)

        if isinstance(result, dict):
            return (
                result.get("answer")
                or result.get("response")
                or str(result)
            )

        return str(result)

    except requests.RequestException as exc:
        return f"Request failed: {exc}"
    except ValueError as exc:
        return f"Invalid JSON response: {exc}"

with gr.Blocks(title="GraphRAG") as demo:
    gr.Markdown("# GraphRAG")
    gr.Markdown("Upload a document and ask questions.")

    with gr.Tab("Upload Document"):
        document = gr.File(
            label="Select document",
            type="filepath",
        )
        upload_button = gr.Button("Upload")
        upload_output = gr.Textbox(
            label="Upload Result",
            lines=5,
        )

        upload_button.click(
            fn=upload_document,
            inputs=document,
            outputs=upload_output,
        )

    with gr.Tab("Ask a Question"):
        question = gr.Textbox(
            label="Question",
            placeholder="Ask something about your knowledge base...",
            lines=3,
        )
        ask_button = gr.Button("Ask")
        answer = gr.Textbox(
            label="Answer",
            lines=10,
        )

        ask_button.click(
            fn=ask_question,
            inputs=question,
            outputs=answer,
        )
        question.submit(
            fn=ask_question,
            inputs=question,
            outputs=answer,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )