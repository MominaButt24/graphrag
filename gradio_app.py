# gradio_app.py
import gradio as gr
import requests

API_BASE = "http://localhost:8000/api/v1"


def upload_doc(file):
    try:
        with open(file.name, "rb") as f:
            response = requests.post(f"{API_BASE}/upload", files={"file": f}, timeout=120)
    except requests.exceptions.ConnectionError:
        return "Error: could not reach the API server. Is it running?"
    except requests.exceptions.Timeout:
        return "Error: upload timed out — the document may be too large or a backend service is slow to respond."

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", f"HTTP {response.status_code}")
        except ValueError:
            detail = f"HTTP {response.status_code}"
        return f"Error: {detail}"

    result = response.json()
    return f"Processed {result['filename']}: {result['chunks']} chunks, {result['failed_chunks']} failed"


def ask(question):
    try:
        response = requests.post(f"{API_BASE}/query/agent", json={"question": question}, timeout=60)
    except requests.exceptions.ConnectionError:
        return "Error: could not reach the API server. Is it running?"
    except requests.exceptions.Timeout:
        return "Error: request timed out — the LLM backend may be slow or unreachable."

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", f"HTTP {response.status_code}")
        except ValueError:
            detail = f"HTTP {response.status_code}"
        return f"Error: {detail}"

    return response.json().get("answer", "Error: no answer returned")


with gr.Blocks(title="GraphRAG") as demo:
    gr.Markdown("# GraphRAG")

    with gr.Row():
        file_input = gr.File(label="Upload Document")
        upload_status = gr.Textbox(label="Upload Status")
    file_input.upload(upload_doc, inputs=file_input, outputs=upload_status)

    gr.Markdown("---")
    question_input = gr.Textbox(label="Ask a question")
    ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=5)
    ask_button.click(ask, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)