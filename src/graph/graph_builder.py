from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.storage.neo4j_client import get_graph  # Neo4jGraph wrapper
import time

from src.generation.llm import get_llm

llm = get_llm()

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# transformer = LLMGraphTransformer(llm=llm)
transformer = LLMGraphTransformer(
    llm=llm,
    ignore_tool_usage=True,
)


def build_graph_from_chunks(chunks, batch_size: int = 5, delay_seconds: float = 1.0, document_id: str = None):
    graph = get_graph()
    failed_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            graph_documents = transformer.convert_to_graph_documents([chunk])
            if document_id:
                for gd in graph_documents:
                    for node in gd.nodes:
                        node.properties["document_id"] = document_id
                    for rel in gd.relationships:
                        rel.properties["document_id"] = document_id

            node_count = sum(len(gd.nodes) for gd in graph_documents)
            rel_count = sum(len(gd.relationships) for gd in graph_documents)

            # A graph extraction with zero nodes and zero relationships is not
            # a successful extraction. Surface it as a chunk failure so callers
            # can stop the upload metadata lifecycle from reporting "done".
            if node_count == 0 and rel_count == 0:
                message = (
                    f"[{i+1}/{len(chunks)}] extracted 0 nodes, 0 relationships "
                    "— graph conversion produced an empty graph for this chunk"
                )
                print(message)
                failed_chunks.append((i, chunk, message))
                continue

            graph.add_graph_documents(graph_documents)
            print(f"[{i+1}/{len(chunks)}] extracted {node_count} nodes, {rel_count} relationships — written OK")
        except Exception as e:
            print(f"[{i+1}/{len(chunks)}] FAILED: {e}")
            failed_chunks.append((i, chunk, str(e)))

        # rate limiting — avoid hammering Groq and tripping limits
        if (i + 1) % batch_size == 0:
            time.sleep(delay_seconds)

    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunks failed — see failed_chunks for details")
    return failed_chunks