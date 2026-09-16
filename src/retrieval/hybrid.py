from src.config.logging import get_logger
import sentry_sdk
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.generation.llm import get_llm
from src.retrieval.graph import smart_query
from src.retrieval.vector import vector_search
from src.retrieval.reranker import rerank_and_merge
from src.retrieval.relevance import _set_retrieval_metadata
from src.retrieval.query_enhancer import enhance_query

logger = get_logger(__name__)

llm = get_llm()



def hybrid_search(question: str, vector_top_k: int = 5) -> dict:
    span = sentry_sdk.get_current_span()
    output = {
        "graph_result": None,
        "vector_results": [],
        "graph_error": None,
        "vector_error": None,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(smart_query, question): "graph",
            executor.submit(vector_search, question, vector_top_k): "vector",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                if source == "graph":
                    output["graph_result"] = result
                else:
                    output["vector_results"] = result
            except Exception as e:
                logger.error(f"[hybrid_search] {source} search failed: {e}")
                output[f"{source}_error"] = str(e)

    if span:
        span.set_tag("hybrid_vector_hits", len(output["vector_results"]))
        span.set_tag("hybrid_graph_error", bool(output["graph_error"]))
    logger.info(f"[hybrid_search] question='{question}' | vector hits: {len(output['vector_results'])} | graph_error: {output['graph_error']}")
    return output


def hybrid_answer(question: str, vector_top_k=5, rerank_top_k=5):
    retrieval_question = enhance_query(question)
    logger.info(
        f"[hybrid_answer] original_question='{question}' "
        f"retrieval_question='{retrieval_question}'"
    )

    hybrid_result = hybrid_search(
        retrieval_question,
        vector_top_k=vector_top_k
    )

    ranked = rerank_and_merge(
        retrieval_question,
        hybrid_result,
        top_k=rerank_top_k
    )

    if not ranked:
        _set_retrieval_metadata({
            "vector_results": hybrid_result.get("vector_results", []),
            "graph_result": hybrid_result.get("graph_result"),
            "ranked_results": [],
            "stats": {
                "vector_candidates": len(hybrid_result.get("vector_results", [])),
                "graph_available": bool(hybrid_result.get("graph_result")),
                "reranked_candidates": 0,
                "final_results": 0,
            },
        })

        return "I couldn't find relevant information in the knowledge base."

    _set_retrieval_metadata({
        "vector_results": hybrid_result.get("vector_results", []),
        "graph_result": hybrid_result.get("graph_result"),
        "ranked_results": ranked,
        "stats": {
            "vector_candidates": len(hybrid_result.get("vector_results", [])),
            "graph_available": bool(hybrid_result.get("graph_result")),
            "reranked_candidates": len(hybrid_result.get("vector_results", [])) + (1 if hybrid_result.get("graph_result") else 0),
            "final_results": len(ranked),
        },
    })

    context = "\n\n".join(
        f"[{c['origin']}] {c['text']}"
        for c in ranked
    )

    prompt = f"""Based on this combined context from a knowledge graph and a vector search:
{context}

Answer this question: {question}"""

    response = llm.invoke(prompt)

    return response.content