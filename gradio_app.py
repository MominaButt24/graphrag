# gradio_app.py
import gradio as gr
import requests

API_BASE = "http://localhost:8000/api/v1"

def upload_doc(file):
    with open(file.name, "rb") as f:
        response = requests.post(f"{API_BASE}/upload", files={"file": f})
    result = response.json()
    return f"Processed {result['filename']}: {result['chunks']} chunks, {result['failed_chunks']} failed"

def ask(question):
    response = requests.post(f"{API_BASE}/query", json={"question": question})
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