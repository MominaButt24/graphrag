from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_groq import ChatGroq
from src.neo4j_client import get_graph  # Neo4jGraph wrapper
import time
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv(override=True)
# llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai", 
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# transformer = LLMGraphTransformer(llm=llm)
transformer = LLMGraphTransformer(
    llm=llm,
    ignore_tool_usage=True,
)

def build_graph_from_chunks(chunks, batch_size: int = 5, delay_seconds: float = 1.0):
    graph = get_graph()
    failed_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            graph_documents = transformer.convert_to_graph_documents([chunk])
            graph.add_graph_documents(graph_documents)
            print(f"[{i+1}/{len(chunks)}] extracted + written OK")
        except Exception as e:
            print(f"[{i+1}/{len(chunks)}] FAILED: {e}")
            failed_chunks.append((i, chunk, str(e)))

        # rate limiting — avoid hammering Groq and tripping limits
        if (i + 1) % batch_size == 0:
            time.sleep(delay_seconds)

    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunks failed — see failed_chunks for details")
    return failed_chunks